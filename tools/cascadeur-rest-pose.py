"""Send the character's own rest pose to Cascadeur and report what came back.

  runtime/python.exe tools/cascadeur-rest-pose.py [project] [--facing DEG]

This is the base check for the live link. A T-pose is the one pose where the
answer is not a matter of taste: if the character does not stand in its rest
pose, facing the same way, nothing that is animated afterwards can be trusted.
The chest normal is printed in world axes, so which way the character faces is
a number rather than an impression.
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

from mocap.cascadeur_link import MCP_PORT, CascadeurLink, _post, _url  # noqa: E402
from mocap.live_body import for_project  # noqa: E402

DEFAULT_PROJECT = BASE / 'projects/Download (2)_23dd0cf4a379e7_project'


def read_back(names):
    # Every joint carries its position under any name; the prefix a character
    # was imported with (mixamorig:, Armature|) is not part of the answer, so
    # nothing here filters on one.
    code = (
        'import csc, json\n'
        'view = csc.app.get_application().get_scene_manager().current_scene()\n'
        'scene = view.domain_scene()\n'
        'mv = scene.model_viewer(); bv = mv.behaviour_viewer(); dv = scene.data_viewer()\n'
        'out = {}\n'
        'for o in mv.get_objects():\n'
        '    n = mv.get_object_name(o)\n'
        '    tr = bv.get_behaviour_by_name(o, "Transform")\n'
        '    if tr.is_null():\n'
        '        continue\n'
        '    p = dv.get_data_value(bv.get_behaviour_data(tr, "global_position"), 0)\n'
        '    out[n] = [float(p[k]) for k in range(3)]\n'
        'print("POS", json.dumps(out))\n')
    reply = _post(_url(MCP_PORT, '/run'), {'code': code}, timeout=60.0)
    for message in reply.get('messages') or []:
        text = message.get('text') or ''
        if text.startswith('POS '):
            return json.loads(text[4:])
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', nargs='?', default=str(DEFAULT_PROJECT))
    parser.add_argument('--facing', type=float, default=None,
                        help='yaw in degrees; omit to use the link default')
    options = parser.parse_args()

    import os
    if options.facing is not None:
        os.environ['RGB_MOCAP_CASCADEUR_FACING'] = str(options.facing)

    body = for_project(options.project)
    if body is None:
        print('No take export in %s' % options.project)
        return 2
    link = CascadeurLink()
    # A link left running from an earlier version would answer the status check
    # and be reused, so this build's receiver would never be loaded. Stopping
    # it first is what makes the T-pose a check of THIS code.
    if link.status():
        print('stopping the previous link:', link.disconnect())
    ok, message = link.connect()
    print('connect:', ok, message)
    if not ok:
        return 1
    # Bind, then send ONLY the rest pose: the take's own frame would land
    # straight after it and the character would already be posed when the
    # facing and the height are measured.
    ok, message = link.start(body, 0, publish_first=False)
    print('start:', ok, message)
    if not ok:
        return 1
    print('facing:', link.facing, 'deg')
    link.publish_payload(link.rest_payload())
    time.sleep(1.2)
    print('receiver:', {k: (link.status() or {}).get(k) for k in
                        ('version', 'bound', 'written', 'errors')})

    got = read_back(link.names)
    by_short = {key.rsplit(':', 1)[-1]: value for key, value in got.items()}

    def at(name):
        return np.asarray(by_short.get(name, [0, 0, 0]), dtype=np.float64)
    hips, head = at('Hips'), at('Head')
    left, right = at('LeftShoulder'), at('RightShoulder')
    up = head - hips
    up = up / max(np.linalg.norm(up), 1e-9)
    across = right - left
    across = across - up * float(np.dot(across, up))
    across = across / max(np.linalg.norm(across), 1e-9)
    chest = np.cross(across, up)
    print('world height   : %.1f cm' % (head[1] - hips[1]))
    print('hip height     : %.1f cm' % hips[1])
    print('right shoulder : %s' % np.round(across, 2).tolist())
    print('chest normal   : %s' % np.round(chest, 2).tolist())
    print('facing         : %s' % ('-Y (towards the front view)' if chest[1] < -0.5
                                   else '+Y (away from the front view)' if chest[1] > 0.5
                                   else 'sideways'))
    link.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
