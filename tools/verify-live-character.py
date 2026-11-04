"""Verify the character live link against a real app character take.

  blender.exe -b --python tools/verify-live-character.py

The app first describes the character on screen (rest file, palette, placement)
and only then streams bone matrices. This test replays that order: it checks the
character is appended and standing at rest before any pose arrives, then compares
the posed bones with the take's own exported animation.
"""
from __future__ import annotations

import json
import math
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import bpy
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'blender_addon'))

import importlib  # noqa: E402

try:
    addon = importlib.import_module('bl_ext.user_default.rgbmocap_bridge')
except ImportError:
    addon = importlib.import_module('rgbmocap_bridge')
live = importlib.import_module(addon.__name__ + '.live')

TAKE = BASE / ('projects/Download_35b14c2294807c_project/characters/1d86e8a4c3dd4e8f8dd981709cc4e466'
               '/animation-988febee65e14e66a4ce17ed082dd3f7/take-0')
PORT = 9235
FPS = 30.0
A4 = np.eye(4)
A4[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
report = {}


def take_data():
    bake = json.loads((TAKE / 'bake.json').read_text(encoding='utf-8'))
    kind = json.loads((TAKE / 'import.json').read_text(encoding='utf-8'))['kind']
    rig = json.loads((TAKE / f'{kind}-rig.json').read_text(encoding='utf-8'))
    palette = sorted(bake['keyed_bones'])
    frames = np.load(TAKE / 'preview-frames.npy', mmap_mode='r')
    with np.load(TAKE.parents[1] / 'asset/preview.npz') as preview:
        vertices = preview['vertices']
    return dict(palette=palette, frames=frames, blend=TAKE / f'{kind}-rest.blend',
                armature=rig.get('armature', 'Armature'), kind=kind,
                height=float(vertices[:, 1].max() - vertices[:, 1].min()))


def to_blender(values):
    return np.einsum('ij,njk,kl->nil', A4, np.asarray(values, dtype=np.float64), A4.T)


def sender(stop, go, data, sent):
    """Describe the character, wait to be asked, then stream poses."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT))
    server.listen(1)
    server.settimeout(0.5)
    client = None
    meta = dict(version=2, kind='character', blend=str(data['blend']),
                palette=data['palette'], armature=data['armature'], transform=None,
                height=data['height'], fps=FPS, frames=int(data['frames'].shape[0]),
                frame=0, label='Person 1', rest_only=True)
    line = (json.dumps(meta) + '\n').encode('utf-8')
    index, started = 0, 0.0
    try:
        while not stop.is_set():
            if client is None:
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                client.sendall(line)
                continue
            if not go.is_set():
                time.sleep(0.02)
                continue
            if not started:
                started = time.time()
            index = int(round((time.time() - started) * FPS)) % data['frames'].shape[0]
            matrices = to_blender(data['frames'][index]).astype('<f4')
            packet = (live.MAGIC_CHARACTER
                      + struct.pack('<IfH', index, FPS, len(data['palette']))
                      + matrices.tobytes())
            try:
                client.sendall(packet)
                sent.append(index)
            except OSError:
                client = None
            time.sleep(1.0 / FPS)
    finally:
        if client is not None:
            client.close()
        server.close()


def pump(seconds):
    """Run the addon timer callback by hand for a while."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        live.tick()
        time.sleep(1.0 / 60.0)


def mesh_stats():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    low, high = None, None
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', coordinates)
            coordinates = coordinates.reshape(-1, 3)
            matrix = np.array(evaluated.matrix_world)
            heights = coordinates @ matrix[2, :3] + matrix[2, 3]
            low = heights.min() if low is None else min(low, heights.min())
            high = heights.max() if high is None else max(high, heights.max())
        finally:
            evaluated.to_mesh_clear()
    return float(low), float(high)


def ground_truth(frame):
    """Bone poses of the take's exported animation at this frame."""
    with tempfile.TemporaryDirectory(prefix='rgbmocap-gt-') as folder:
        target = Path(folder) / 'bones.json'
        result = subprocess.run([bpy.app.binary_path, '-b', '--python',
                                 str(TOOLS / 'dump-export-bones.py'), '--',
                                 str(TAKE), str(frame), str(target)],
                                capture_output=True, text=True)
        if not target.is_file():
            raise RuntimeError('could not read exported bones: ' + (result.stdout or result.stderr)[-400:])
        return json.loads(target.read_text(encoding='utf-8'))['poses']


def bone_errors(armature, truth):
    """Angle and head shift of every posed bone against the exported animation."""
    rows = []
    for name, values in truth.items():
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            continue
        expected = np.asarray(values, dtype=float).reshape(4, 4)
        actual = np.array(armature.matrix_world @ pose_bone.matrix)
        # The rest file's armature keeps a uniform import scale, so the rotation
        # blocks must be normalised before comparing their angle.
        left = expected[:3, :3] / np.linalg.norm(expected[:3, :3], axis=0).mean()
        right = actual[:3, :3] / np.linalg.norm(actual[:3, :3], axis=0).mean()
        delta = left.T @ right
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(delta) - 1) / 2))))
        shift = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
        rows.append((name, angle, shift))
    rows.sort(key=lambda row: -max(row[1] / 1.0, row[2] * 1000.0))
    return rows


def main():
    data = take_data()
    report['take'] = dict(palette=len(data['palette']), frames=int(data['frames'].shape[0]),
                          blend=str(data['blend']), blend_exists=data['blend'].is_file(),
                          height=round(data['height'], 3))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Scene, 'rgbmocap'):
        addon.register()

    stop, go, sent = threading.Event(), threading.Event(), []
    threading.Thread(target=sender, args=(stop, go, data, sent), daemon=True).start()
    time.sleep(0.4)
    if not live.start('127.0.0.1', PORT):
        raise SystemExit('no connection: ' + live.status()['error'])

    # Phase 1: description only. The character must be in the scene, at rest.
    deadline = time.time() + 6.0
    while time.time() < deadline:
        pump(0.1)
        if live.character_state().get('active'):
            break
    rest = live.status()
    rest_low, rest_high = mesh_stats()
    report['rest'] = dict(active=live.character_state().get('active'),
                          received=rest['received'], applied=rest['applied'],
                          meshes=len([o for o in bpy.context.scene.objects if o.type == 'MESH']),
                          armatures=len([o for o in bpy.context.scene.objects if o.type == 'ARMATURE']),
                          min_z=round(rest_low, 3), max_z=round(rest_high, 3),
                          error=rest['error'])

    # Phase 2: poses.
    go.set()
    pump(2.0)
    status = live.status()
    # Compare the pose that is actually in the scene, not the newest arrival.
    frame = status['applied_frame']
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
    rows = bone_errors(armature, ground_truth(frame))
    checked = len(rows)
    angle = max(row[1] for row in rows)
    shift = max(row[2] for row in rows)
    low, high = mesh_stats()
    hips = armature.matrix_world @ armature.pose.bones['mixamorig:Hips'].head
    report['pose'] = dict(frame=frame, received=status['received'], applied=status['applied'],
                          latest=status['frame'],
                          bones_checked=checked, worst_angle_deg=round(angle, 5),
                          worst_shift_mm=round(shift * 1000, 4), frames_sent=len(sent),
                          worst=[dict(bone=name, angle_deg=round(a, 4), shift_mm=round(s * 1000, 3))
                                 for name, a, s in rows[:8]])
    report['mesh'] = dict(min_z=round(low, 3), max_z=round(high, 3), height=round(high - low, 3),
                          hips_z=round(float(hips[2]), 3),
                          hips_xy=[round(float(hips[0]), 3), round(float(hips[1]), 3)])
    report['state'] = live.character_state()
    stop.set()
    time.sleep(0.2)
    pump(0.1)
    live.stop()

    ok = (report['rest']['active'] and report['rest']['received'] == 0
          and report['rest']['armatures'] == 1 and report['rest']['meshes'] == 1
          and checked == len(data['palette']) and angle < 0.05 and shift < 0.001
          and 1.4 < high - low < 2.2 and hips[2] > 0.7 and low > -0.05)
    report['pass'] = bool(ok)
    target = BASE / 'screenshots' / 'live-character-report.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFY-LIVE-CHARACTER', json.dumps(report), flush=True)
    if not ok:
        raise SystemExit(1)


main()
