"""Dump exported bone poses of a character take, for live-link comparison.

  blender.exe -b --python tools/dump-export-bones.py -- <take folder> <frame> <out.json>

The frame is the app's 0-based frame; the baked scene stores it as frame + 1.
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

args = sys.argv[sys.argv.index('--') + 1:]
take, frame, target = Path(args[0]), int(args[1]), Path(args[2])
palette = sorted(json.loads((take / 'bake.json').read_text(encoding='utf-8'))['keyed_bones'])

bpy.ops.wm.open_mainfile(filepath=str(take / 'animation.blend'))
scene = bpy.context.scene
armature = next(obj for obj in scene.objects if obj.type == 'ARMATURE')
scene.frame_set(frame + 1)
bpy.context.view_layer.update()
poses = {}
for name in palette:
    pose_bone = armature.pose.bones.get(name)
    if pose_bone is None:
        continue
    poses[name] = [float(value) for row in (armature.matrix_world @ pose_bone.matrix) for value in row]
target.write_text(json.dumps(dict(frame=frame, poses=poses)), encoding='utf-8')
print('DUMPED', len(poses), 'bones at', frame, flush=True)
