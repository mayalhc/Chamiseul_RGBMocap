# Copyright (c) 2026 Chamiseul. All rights reserved.
# Proprietary: redistribution prohibited. See LICENSE-CHAMISEUL.md in the repository root.
"""Check the Cascadeur live link end to end against a running Cascadeur.

  runtime/python.exe tools/verify-cascadeur-live.py

This connects the link the way the File menu does - posting the receiver's
source to Cascadeur's script server and letting it listen on 9566 - then
streams a real take, and finally asks Cascadeur what it actually holds at the
written frames. The read-back is the point: a receiver that binds to nothing
still counts frames it never wrote, so the frames are compared joint by joint
against the orientations that were sent.

It writes keyframes into whatever scene Cascadeur has open. That scene is not
saved, and the playhead is put back where it started, but the take in the open
document will have extra frames after this runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from mocap.cascadeur_link import CascadeurLink, MCP_PORT, PORT, _get, _post, _url, _yaw  # noqa: E402
from mocap.live_body import for_project, yawed_points  # noqa: E402

DEFAULT_PROJECT = BASE / 'projects/Download (2)_23dd0cf4a379e7_project'
FRAMES = (0, 30, 60)


def angle_between(a, b):
    """Degrees between two rotation matrices."""
    r = np.asarray(a) @ np.asarray(b).T
    trace = max(-1.0, min(3.0, float(np.trace(r))))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))))


def rest_geometry(spots, frame):
    """Is the character standing in a humanoid rest pose, in the world?

    The per-joint comparison below is relative and cannot see a rest that is
    itself wrong: a receiver that read its rest from a keyed frame - or a
    banked rest that did not fit - writes self-consistent nonsense. These
    absolute checks are what catches that: a standing figure has its head
    above its hips, its feet on the floor and its arms out level. Cascadeur's
    world is Y-up, so height is the Y component.
    """
    import math
    frame_key = str(frame)
    needed = ('Hips', 'Head', 'LeftArm', 'LeftForeArm', 'RightArm',
              'RightForeArm', 'LeftFoot', 'RightFoot')
    missing = [name for name in needed if frame_key not in (spots.get(name) or {})]
    if missing:
        return {'ok': False, 'error': 'joints missing from the scene: %s'
                % ', '.join(missing)}

    def at(name):
        return np.asarray(spots[name][frame_key], dtype=np.float64)

    def elevation(name, child):
        direction = at(child) - at(name)
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return None
        return math.degrees(math.asin(max(-1.0, min(1.0, direction[1] / length))))

    hips = at('Hips')
    head = at('Head')
    toes = at('LeftToeBase') - at('LeftFoot')
    toes_length = float(np.linalg.norm(toes))
    facing = round(float(toes[2] / toes_length), 3) if toes_length > 1e-6 else None
    checks = {
        'hips_height_cm': round(float(hips[1]), 1),
        'head_above_hips_cm': round(float(head[1] - hips[1]), 1),
        'left_foot_height_cm': round(float(at('LeftFoot')[1]), 1),
        'right_foot_height_cm': round(float(at('RightFoot')[1]), 1),
        'left_arm_elevation_deg': elevation('LeftArm', 'LeftForeArm'),
        'right_arm_elevation_deg': elevation('RightArm', 'RightForeArm'),
        'facing_forward': facing,
    }
    checks['ok'] = bool(
        70.0 <= checks['hips_height_cm'] <= 140.0
        and 40.0 <= checks['head_above_hips_cm'] <= 85.0
        and -15.0 <= checks['left_foot_height_cm'] <= 30.0
        and -15.0 <= checks['right_foot_height_cm'] <= 30.0
        and checks['left_arm_elevation_deg'] is not None
        and abs(checks['left_arm_elevation_deg']) <= 35.0
        and checks['right_arm_elevation_deg'] is not None
        and abs(checks['right_arm_elevation_deg']) <= 35.0
        # Either facing is a humanoid rest: the exported base body is authored
        # towards -Z while Cascadeur's front view looks from +Z. What matters
        # is that the character is not standing sideways, and that the stream
        # (no half turn by default) keeps whatever facing it was imported with.
        and facing is not None and abs(facing) > 0.5)
    if not checks['ok']:
        checks['error'] = ('the character is not standing in a humanoid rest '
                           'pose - the rest itself is wrong')
    return checks


def read_back(names, frames):
    """Ask Cascadeur for the pose it holds at each frame."""
    code = (
        "import json\n"
        "import csc\n"
        "NAMES = %r\n"
        "FRAMES = %r\n"
        "view = csc.app.get_application().get_scene_manager().current_scene()\n"
        "scene = view.domain_scene()\n"
        "mv = scene.model_viewer(); bv = mv.behaviour_viewer(); data = scene.data_viewer()\n"
        "objs = {mv.get_object_name(o): o for o in mv.get_objects()}\n"
        "out = {}\n"
        "places = {}\n"
        "for name in NAMES:\n"
        "    oid = objs.get(name)\n"
        "    if oid is None:\n"
        "        continue\n"
        "    tr = bv.get_behaviour_by_name(oid, 'Transform')\n"
        "    if tr.is_null():\n"
        "        continue\n"
        "    rot = bv.get_behaviour_data(tr, 'global_rotation')\n"
        "    pos = bv.get_behaviour_data(tr, 'global_position')\n"
        "    per = {}\n"
        "    where = {}\n"
        "    for f in FRAMES:\n"
        "        m = data.get_data_value(rot, int(f)).to_rotation_matrix()\n"
        "        per[int(f)] = [[float(m[r][c]) for c in range(3)] for r in range(3)]\n"
        "        p = data.get_data_value(pos, int(f))\n"
        "        where[int(f)] = [float(p[k]) for k in range(3)]\n"
        "    out[name] = per\n"
        "    places[name] = where\n"
        "print('READBACK', json.dumps(out))\n"
        "print('PLACES', json.dumps(places))\n"
        "print('CURRENT', scene.get_current_frame())\n" % (list(names), list(frames)))
    reply = _post(_url(MCP_PORT, '/run'), {'code': code}, timeout=60.0)
    if not reply.get('ok'):
        raise RuntimeError(reply.get('error') or 'Cascadeur refused the read-back')
    found = {}
    spots = {}
    current = None
    for message in reply.get('messages') or []:
        text = message.get('text') or ''
        if text.startswith('READBACK '):
            found = json.loads(text[len('READBACK '):])
        elif text.startswith('PLACES '):
            spots = json.loads(text[len('PLACES '):])
        elif text.startswith('CURRENT '):
            current = int(text.split(' ', 1)[1])
    return found, spots, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', nargs='?', default=str(DEFAULT_PROJECT))
    parser.add_argument('--frames', default=','.join(str(f) for f in FRAMES))
    parser.add_argument('--output', default='')
    options = parser.parse_args()
    frames = [int(v) for v in options.frames.split(',') if v.strip()]

    report = {'project': str(options.project), 'frames': frames}
    body = for_project(options.project)
    if body is None:
        print('No take export in %s' % options.project)
        return 2

    link = CascadeurLink()
    # A receiver left running by an earlier session answers the status check and
    # would be reused - so this report would describe that build, not the source
    # in the tree. Stop it, then load what is here now.
    if link.status():
        print('stopping the previous link:', link.disconnect())
    ok, message = link.connect()
    report['connect'] = message
    print('connect:', ok, message)
    if not ok:
        report['ok'] = False
        return 1

    state = link.status() or {}
    report['receiver'] = {k: state.get(k) for k in
                          ('version', 'bound', 'roots', 'written', 'errors')}
    print('receiver:', report['receiver'])

    ok, message = link.start(body, frames[0], publish_first=False)
    report['start'] = message
    print('start:', ok, message)
    if not ok:
        report['ok'] = False
        return 1
    report['matched'] = len(link.body_index)

    # Nothing has been written yet, so the character stands in its rest -
    # the one pose whose answer is not a matter of taste. Fail here, in
    # numbers, before a single take pose is judged.
    _rest_held, rest_spots, _rest_current = read_back(link.names, [frames[0]])
    geometry = rest_geometry(rest_spots, frames[0])
    report['rest_geometry'] = geometry
    print('rest:', json.dumps(geometry, ensure_ascii=False))
    if not geometry.get('ok'):
        report['ok'] = False
        Path('cascadeur-rest-failed.json').write_text(
            json.dumps(report, indent=1), encoding='utf-8')
        link.stop()
        return 1

    # One frame at a time, so a failure names the frame that failed rather
    # than "somewhere in the batch".
    expected = {frame: link.frame_payload(frame) for frame in frames}
    wrote = []
    for frame in frames:
        before = (link.status() or {}).get('written', 0)
        link.publish(frame)
        began = time.time()
        while time.time() - began < 10.0:
            now = link.status() or {}
            if now.get('written', 0) > before:
                break
            if now.get('errors'):
                break
            time.sleep(0.1)
        now = link.status() or {}
        wrote.append({'frame': frame, 'written': now.get('written', 0) - before,
                      'errors': now.get('errors') or []})
    report['writes'] = wrote
    print('writes:', wrote)
    report['status'] = link.status()
    # Hold the link for a moment so the last frame is definitely flushed.
    time.sleep(0.4)

    held, spots, current = read_back(link.names, frames)
    report['playhead'] = current
    # What each joint must point at over there: the turn that was sent, composed
    # onto the rest the receiver reports for that joint.
    try:
        rest_payload = _get(_url(PORT, '/rests'), timeout=6.0) or {}
    except (OSError, ValueError):
        rest_payload = {}
    rests = dict(rest_payload.get('rests') or {})
    report['rests'] = len(rests)
    report['rest_locals'] = dict(rest_payload.get('locals') or {})
    # Every joint is compared against its parent, not against the world. A
    # Cascadeur scene in Rig Mode lets its own rig drive the root joint - the
    # sample document open here has a spline IK doing exactly that - and that
    # one override turns every child's world orientation without a single
    # child rotation being wrong. The parent-relative turn is what the stream
    # actually writes, so that is what is measured. The root is reported
    # separately, as the world value it is.
    # The stream is aim-driven: a joint is turned so that its child points the
    # way the take's own joint positions do. That geometry is what is checked,
    # not the export skeleton's rotation table, which lives in its own basis.
    child_row = {}
    errors = {}
    # What each joint must hold: the turn that was sent, composed onto the
    # rest this character actually has. The measured value is the local
    # rotation Cascadeur derives - the child's world orientation turned into
    # its parent's frame - so the check sees exactly what the receiver wrote,
    # whatever rotations the imported nodes above the character carry. The
    # rest itself comes from the receiver (`locals`), because that is the
    # value its writes compose onto; the bind-derived world rests are only a
    # fallback for an older receiver.
    parents_all = np.asarray(body.parents)
    # Compare BONE DIRECTIONS, not reconstructed local rotations. The stream is
    # aim-driven, and a direction read back from Cascadeur is the same physical
    # quantity on both sides whatever rotation convention either skeleton uses.
    # (The old check rebuilt the receiver's formula and therefore agreed with it
    # even when the body was folded in half.)
    yaw = _yaw(link.facing)
    for row, name in enumerate(link.names):
        if row == link.root or name not in spots:
            continue
        app_index = link.body_index[row]
        parent_app = int(parents_all[app_index]) if app_index < len(parents_all) else -1
        parent_name = body.names[parent_app] if 0 <= parent_app < len(body.names) else None
        if parent_name is None or parent_name not in spots:
            continue
        per = []
        for frame in frames:
            key = str(frame)
            here = (spots.get(name) or {}).get(key)
            above = (spots.get(parent_name) or {}).get(key)
            if here is None or above is None:
                continue
            shown = np.asarray(here, dtype=np.float64) - np.asarray(above, dtype=np.float64)
            wish = (np.asarray(body.frames[frame])[app_index, :3, 3]
                    - np.asarray(body.frames[frame])[parent_app, :3, 3]) @ yaw.T
            if float(np.linalg.norm(shown)) < 1e-9 or float(np.linalg.norm(wish)) < 1e-9:
                continue
            cos = float(np.dot(shown, wish)
                        / (float(np.linalg.norm(shown)) * float(np.linalg.norm(wish))))
            per.append(round(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), 3))
        if per:
            errors[name] = per
    report['errors_deg'] = errors
    root = link.names[link.root] if link.root < len(link.names) and link.names[link.root] in spots else None
    report['root_shift_cm'] = None
    if root:
        index = link.body_index[link.root]
        shifts = []
        for frame in frames:
            want = np.asarray(body.frames[frame])[index, :3, 3] @ yaw.T * 100.0
            got = np.asarray(spots[root][str(frame)], dtype=np.float64)
            shifts.append(round(float(np.linalg.norm(want - got)), 2))
        report['root_shift_cm'] = shifts
    worst = max((max(v) for v in errors.values()), default=None)
    report['worst_deg'] = worst
    mean = (float(np.mean([v for rows in errors.values() for v in rows]))
            if errors else None)
    report['mean_deg'] = mean
    close = sum(1 for rows in errors.values() if max(rows) < 5.0)
    report['close_joints'] = close
    print('joints compared:', len(errors), 'worst', worst, 'mean', mean,
          'within 5 deg:', close, 'root shift cm', report['root_shift_cm'])
    # The per-frame increments race the receiver's own tick; the final counter
    # is the reliable number (a duplicate frame can also legitimately be a no-op).
    written = int((link.status() or {}).get('written', 0))
    report['written_total'] = written
    shifts = report['root_shift_cm']
    shift_ok = shifts is not None and all(float(s) < 5.0 for s in shifts)
    report['ok'] = bool(len(errors) >= 30 and written >= len(frames) - 1
                        and (mean is None or mean < 2.0)
                        and close >= len(errors) - 4 and shift_ok)
    print('written frames:', written, 'ok:', report['ok'])

    link.stop()
    if options.output:
        Path(options.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
        print('report:', options.output)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
