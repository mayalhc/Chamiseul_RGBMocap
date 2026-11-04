"""Check that Blender loads the app's own base body over the live link.

  blender.exe -b --python tools/verify-live-base.py

Run tools/live-base-server.py (a real take, the app's own publisher) first, or
pass --serve to start it from here. The addon's **Load Base Character** must
bring in the character RGB MoCap shows - its SMPL body surface and its 52-bone
skeleton - and not a stand-in rig of its own.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import bpy

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'blender_addon'))

import importlib  # noqa: E402

try:
    addon = importlib.import_module('bl_ext.user_default.rgbmocap_bridge')
except ImportError:
    addon = importlib.import_module('rgbmocap_bridge')
live = importlib.import_module(addon.__name__ + '.live')

PORT = 9241
# A project workspace, not a take folder: the app resolves the newest take below
# it, which is the single-person view's path.
TAKE = BASE / 'projects/Download_35b14c2294807c_project'


def surface_gap(armature, body, frame):
    """Largest distance between Blender's deformed body and the app's surface."""
    sys.path.insert(0, str(BASE))
    from mocap.live_body import for_project, yawed_points
    import numpy as np

    stream = for_project(TAKE)
    skin = stream.skin
    vertices = yawed_points(skin['vertices'])
    matrices = stream.matrices(max(0, int(frame)))
    weights = np.zeros((len(vertices), len(stream.names)))
    owner = np.repeat(np.arange(len(vertices)), np.diff(skin['offsets']))
    weights[owner, skin['bones']] = skin['weights']
    expected = (np.einsum('vj,jik,vk->vi', weights, matrices[:, :3, :3], vertices)
                + np.einsum('vj,jk->vk', weights, matrices[:, :3, 3]))
    turn = np.eye(4)
    turn[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    expected = expected @ turn[:3, :3].T
    bpy.context.view_layer.update()
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    actual = np.array([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices])
    evaluated.to_mesh_clear()
    return float(np.abs(actual - expected).max())


def start_server():
    script = BASE / 'tools/live-base-server.py'
    runtime = BASE / 'runtime/python.exe'
    python = str(runtime) if runtime.is_file() else sys.executable
    return subprocess.Popen([python, '-B', str(script), str(TAKE), '--port', str(PORT)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main():
    server = start_server()
    report = {'pass': False}
    try:
        for _ in range(120):
            line = server.stdout.readline()
            if not line:
                break
            if line.startswith('SERVING'):
                report['server'] = line.strip()
                break
        armature = live.load_base('127.0.0.1', PORT, wait=6.0)
        report['loaded'] = armature.name if armature else None
        report['error'] = live.status()['error']
        if armature is not None:
            bodies = [obj for obj in armature.children if obj.type == 'MESH']
            body = bodies[0] if bodies else None
            report['bones'] = len(armature.data.bones)
            report['body'] = body.name if body else None
            report['vertices'] = len(body.data.vertices) if body else 0
            report['deform'] = sorted(group.name for group in body.vertex_groups)[:3]
            # A pose packet arrived after the rest file: the rig must follow it.
            samples = []
            for _ in range(90):
                live.tick()
                if armature.pose.bones:
                    samples.append((armature.pose.bones[0].head.copy(),
                                    live.status()['applied_frame']))
                time.sleep(1.0 / 60.0)
            moved = max((a[0] - b[0]).length for a, b in zip(samples, samples[1:])) if len(samples) > 1 else 0.0
            report['applied_frames'] = live.status()['applied_frame']
            report['max_step_m'] = round(moved, 5)
            # The imported surface must be the app's own surface for the frame
            # being shown, not just a rig near it.
            gap = surface_gap(armature, body, live.status()['applied_frame'])
            report['surface_gap_mm'] = round(gap * 1000, 4)
            report['pass'] = bool(body is not None and len(body.data.vertices) > 5000
                                  and len(armature.data.bones) == 52
                                  and report['applied_frames'] >= 0 and gap < 0.001)
            report['stand_in_present'] = 'RGB MoCap Base' in bpy.data.objects
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
    print('VERIFY-LIVE-BASE ' + json.dumps(report))


if __name__ == '__main__':
    main()
