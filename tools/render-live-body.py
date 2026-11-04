"""Render the live body the bridge builds, from the front and the side.

  blender.exe -b --python tools/render-live-body.py

Builds the app's base body exactly as the live link does, applies one streamed
frame, and renders it, so the character's facing and pose can be eyeballed
against the app.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'blender_addon'))

from mocap.live_body import for_project  # noqa: E402

import importlib  # noqa: E402

addon = importlib.import_module('bl_ext.user_default.rgbmocap_bridge')
body_import = importlib.import_module(addon.__name__ + '.body_import')
live = importlib.import_module(addon.__name__ + '.live')

TAKE = BASE / 'projects/Download_35b14c2294807c_project'
FRAME = int(__import__('os').environ.get('RGBMOCAP_FRAME') or 120)
OUT = BASE / 'screenshots'


def camera_view(name, location, target=(0.0, 0.0, 0.95)):
    camera = bpy.data.cameras.new(name)
    camera.lens = 50
    obj = bpy.data.objects.new(name, camera)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = Vector(location)
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return obj


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = for_project(TAKE)
    built = body_import.build(body.rest_file())
    armature = built['armature']
    # The app converts every streamed matrix to Blender's Z-up frame first.
    import numpy as np
    turn = np.eye(4)
    turn[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    matrices = np.einsum('ij,njk,kl->nil', turn, body.matrices(FRAME), turn.T)
    frames = [row.reshape(16).tolist() for row in matrices]
    applied, missing = live.apply_character(armature, body.names, Matrix.Identity(4), frames)
    print('RENDER applied %d bones, %d missing' % (applied, missing))

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.device = 'CPU'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.film_transparent = False

    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
    world.node_tree.nodes['Background'].inputs[1].default_value = 1.0
    scene.world = world

    light = bpy.data.lights.new('Key', 'AREA')
    light.energy = 900
    light.size = 3
    key = bpy.data.objects.new('Key', light)
    scene.collection.objects.link(key)
    key.location = (2.5, -3.0, 3.2)
    key.rotation_euler = (Vector((0, 0, 1.0)) - key.location).to_track_quat('-Z', 'Y').to_euler()

    ground = bpy.data.meshes.new('Ground')
    ground.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
    ground.update()
    floor = bpy.data.objects.new('Ground', ground)
    material = bpy.data.materials.new('Ground')
    material.diffuse_color = (0.22, 0.22, 0.24, 1)
    material.use_nodes = True
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.22, 0.22, 0.24, 1)
    ground.materials.append(material)
    scene.collection.objects.link(floor)

    views = {'front': (0.0, -4.2, 1.25), 'side': (4.2, 0.0, 1.25),
             'three-quarter': (2.9, -3.2, 1.5)}
    for name, location in views.items():
        scene.camera = camera_view('Camera ' + name, location)
        scene.render.filepath = str(OUT / ('live-body-%s.png' % name))
        bpy.ops.render.render(write_still=True)
        print('RENDER wrote', scene.render.filepath)

    # The bind pose too: its facing must match the app's own default view.
    for pose in armature.pose.bones:
        pose.matrix_basis.identity()
    bpy.context.view_layer.update()
    for name, location in (('front', (0.0, -4.2, 1.25)), ('side', (4.2, 0.0, 1.25))):
        scene.camera = camera_view('Camera rest ' + name, location)
        scene.render.filepath = str(OUT / ('live-body-rest-%s.png' % name))
        bpy.ops.render.render(write_still=True)
        print('RENDER wrote', scene.render.filepath)


if __name__ == '__main__':
    main()
