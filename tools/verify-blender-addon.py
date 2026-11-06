# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: GPL-3.0-or-later (see blender_addon/LICENSE)
"""Verify the RGB MoCap Bridge addon inside Blender, headless.

  blender.exe -b --python tools/verify-blender-addon.py

Imports a real Mixamo character, checks humanoid detection and renaming,
floats the clip off the floor, plants it back per frame, then round-trips FBX
and GLB through a fresh import and measures the result. Writes a JSON report
next to the repository screenshots.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy

BASE = Path(__file__).resolve().parents[1]
ADDON = BASE / 'blender_addon'
CHARACTER = Path('C:/Users/mayal/Downloads/Ch14_nonPBR.fbx')
OUT = BASE / 'screenshots'
# Prefer the installed extension, so packaging and registration are tested
# too; fall back to the checkout when Blender runs with --factory-startup.
import importlib  # noqa: E402

try:
    addon = importlib.import_module('bl_ext.user_default.rgbmocap_bridge')
except ImportError:
    sys.path.insert(0, str(ADDON))
    addon = importlib.import_module('rgbmocap_bridge')
ground = importlib.import_module(addon.__name__ + '.ground')

report = dict(character=str(CHARACTER), blender=bpy.app.version_string)


def step(message):
    print('VERIFY-STEP', message, flush=True)


def fresh():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def active():
    return bpy.context.active_object


def armature_of():
    return next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')


def min_z(armature):
    return ground.measure(armature, foot_only=False)['heights']


def float_clip(armature, amplitude=0.3, frames=60):
    """Lift the clip off the floor by a moving amount, like a bad export."""
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    armature.location = (0.0, 0.0, 0.0)
    for frame in (1, frames // 4, frames // 2, 3 * frames // 4, frames):
        bpy.context.scene.frame_set(frame)
        armature.location = (0.0, 0.0, 0.35 + amplitude * math.sin(frame / frames * math.tau))
        armature.keyframe_insert('location', frame=frame)
    scene.frame_set(1)


def main():
    fresh()
    if not hasattr(bpy.types.Scene, 'rgbmocap'):
        addon.register()
    report['addon'] = dict(module=addon.__name__, version=addon.__version__)

    # 1. Import through the addon, as a user would from the panel.
    step('import')
    bpy.ops.rgbmocap.import_character(filepath=str(CHARACTER))
    armature = armature_of()
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bones_before = sorted(bone.name for bone in armature.data.bones)
    report['import'] = dict(objects=len(bpy.context.scene.objects), bones=len(bones_before),
                            meshes=len([o for o in bpy.context.scene.objects if o.type == 'MESH']))
    report['sample_bones'] = bones_before[:3]

    # 2. Detection and canonical renaming.
    step('detect')
    found, missing, _ = addon.describe(armature)
    report['detected'] = dict(matched=len(found), missing=missing,
                              hips=found.get('Hips'), left_leg=found.get('LeftLeg'))
    bpy.ops.rgbmocap.rename()
    names = sorted(bone.name for bone in armature.data.bones)
    # Only the mapped body bones are renamed; fingers and helpers keep their
    # original names on purpose.
    body_prefixed = [name for name in names
                     if name.split(':')[-1] in addon.NAMES and ':' in name]
    report['renamed'] = dict(canonical_present=all(name in names for name in addon.NAMES),
                             body_prefixed=body_prefixed,
                             extras_untouched=sum(1 for name in names if ':' in name))
    groups = sorted({group.name for obj in bpy.context.scene.objects if obj.type == 'MESH'
                     for group in obj.vertex_groups})
    report['skin_groups_canonical'] = 'Hips' in groups and 'LeftUpLeg' in groups

    # 3. Per-frame flat-ground snapping on a deliberately floating clip.
    step('float')
    float_clip(armature)
    before = min_z(armature)
    settings = bpy.context.scene.rgbmocap
    settings.ground_mode = 'PLANT'
    settings.foot_only = True
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    step('snap')
    result = bpy.ops.rgbmocap.ground()
    after = min_z(armature)
    if before.max() - before.min() < 0.2:
        raise SystemExit('the test clip was not floating')
    report['ground'] = dict(operator=list(result), frames=len(after),
                            before_min=float(before.min()), before_max=float(before.max()),
                            after_min=float(after.min()), after_max=float(after.max()),
                            residual=float(abs(after).max()),
                            plant_per_frame=float(after.max() - after.min()))
    if abs(after).max() > 1e-4:
        raise SystemExit('feet did not stay on the floor: %.6f' % abs(after).max())

    # 4. Export and re-import both formats.
    exports = {}
    for fmt, suffix in (('FBX', '.fbx'), ('GLB', '.glb')):
        step('export ' + fmt)
        path = OUT / ('blender-addon-roundtrip' + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        if fmt == 'FBX':
            bpy.ops.rgbmocap.export_fbx(filepath=str(path))
        else:
            bpy.ops.rgbmocap.export_glb(filepath=str(path))
        size = path.stat().st_size
        step('reimport ' + fmt)
        fresh()
        if fmt == 'FBX':
            bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False,
                                     use_prepost_rot=True)
        else:
            bpy.ops.import_scene.gltf(filepath=str(path))
        imported = armature_of()
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = max(1, int(imported.animation_data.action.frame_range[1])
                                         if imported.animation_data and imported.animation_data.action else 1)
        heights = min_z(imported)
        exports[fmt] = dict(path=str(path), bytes=size,
                            bones=len(imported.data.bones),
                            meshes=len([o for o in bpy.context.scene.objects if o.type == 'MESH']),
                            vertices=sum(len(o.data.vertices) for o in bpy.context.scene.objects
                                         if o.type == 'MESH'),
                            frames=bpy.context.scene.frame_end,
                            floor_residual=float(abs(heights).max()))
        if exports[fmt]['floor_residual'] > 2e-3:
            raise SystemExit('%s did not keep the planted floor: %.6f'
                             % (fmt, exports[fmt]['floor_residual']))
    report['exports'] = exports

    OUT.mkdir(exist_ok=True)
    target = Path(os.environ.get('RGBMOCAP_ADDON_REPORT', OUT / 'blender-addon-report.json'))
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFY-ADDON', json.dumps(report))


try:
    main()
except BaseException as error:  # a failed check must not look like a pass
    import traceback
    traceback.print_exc()
    report['error'] = '%s: %s' % (type(error).__name__, error)
    OUT.mkdir(exist_ok=True)
    (OUT / 'blender-addon-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise SystemExit(1)
