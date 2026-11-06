# Copyright (c) 2026 Chamiseul. All rights reserved.
# Proprietary: redistribution prohibited. See LICENSE-CHAMISEUL.md in the repository root.
"""Build the app's displayed body from the data it streams.

The app writes the surface it draws together with the skeleton it exports: rest
joints, the rest basis of every bone, a skinned surface and its influence
weights. Building the rig here from those numbers (instead of importing a file)
keeps the rest pose exactly the one the streamed matrices are relative to, so a
frame's ``pose @ rest^-1`` reproduces the app's surface.
"""
import bpy
import numpy as np

# App world is Y-up; Blender is Z-up. A @ (x, y, z) = (x, -z, y).
A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
A4 = np.eye(4)
A4[:3, :3] = A
NAME = 'RGBMoCap Body'
BODY = 'RGBMoCap BodyMesh'


def to_blender(points):
    return np.asarray(points, dtype=float) @ A.T


def _leaf_names(names, parents):
    used = set(int(parent) for parent in parents if int(parent) >= 0)
    return {index for index in range(len(names)) if index not in used}


def build(path, name=NAME):
    """Create the armature and skinned body from a saved rest payload."""
    data = np.load(str(path), allow_pickle=False)
    names = [str(value) for value in data['names']]
    parents = np.asarray(data['parents']).astype(int)
    # A body saved by an older app has no joint pivots and streams transforms
    # about a different origin; building it would tear the surface apart.
    if 'pivots' not in data.files or 'yaw' not in data.files:
        raise ValueError('RGB MoCap is streaming an old body format. Restart the app, '
                         'then press Start Live Link again.')
    # Bone heads sit on the joint each cluster turns the surface around, so a
    # streamed transform is exactly ``pose @ rest^-1`` for its cluster.
    rest = to_blender(data['pivots'])
    tip = np.asarray(data['tip'], dtype=float)
    vertices = to_blender(data['vertices'])
    faces = np.asarray(data['faces']).astype(int)
    weights = np.asarray(data['weights'], dtype=float)
    offsets = np.asarray(data['offsets']).astype(int)
    bones = np.asarray(data['bones']).astype(int)

    armature_data = bpy.data.armatures.new(name)
    armature = bpy.data.objects.new(name, armature_data)
    bpy.context.scene.collection.objects.link(armature)
    view_layer = bpy.context.view_layer
    view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    edit = {}
    for index, bone_name in enumerate(names):
        bone = armature_data.edit_bones.new(bone_name)
        bone.head = rest[index]
        children = [child for child, parent in enumerate(parents) if parent == index]
        if children:
            bone.tail = rest[children[0]]
        else:
            direction = tip[index] @ A.T
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0.0, 0.0, 0.05])
            bone.tail = rest[index] + direction
        if (np.asarray(bone.tail) - np.asarray(bone.head)).sum() == 0:
            bone.tail = np.asarray(bone.head) + np.array([0.0, 0.0, 0.05])
        edit[bone_name] = bone
    for index, bone_name in enumerate(names):
        parent = int(parents[index])
        if parent >= 0:
            edit[bone_name].parent = edit[names[parent]]
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = bpy.data.meshes.new(BODY)
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], faces.tolist())
    mesh.validate()
    mesh.update()
    body = bpy.data.objects.new(BODY, mesh)
    bpy.context.scene.collection.objects.link(body)
    groups = {}
    for index, bone_name in enumerate(names):
        groups[index] = body.vertex_groups.new(name=bone_name)
    owner = np.repeat(np.arange(len(vertices)), np.diff(offsets))
    for influence in range(len(weights)):
        bone = int(bones[influence])
        if bone not in groups or weights[influence] <= 0.0:
            continue
        groups[bone].add([int(owner[influence])], float(weights[influence]), 'REPLACE')
    modifier = body.modifiers.new('Armature', 'ARMATURE')
    modifier.object = armature
    body.parent = armature
    material = bpy.data.materials.get('RGB MoCap Body')
    if material is None:
        material = bpy.data.materials.new('RGB MoCap Body')
        material.diffuse_color = (0.72, 0.66, 0.60, 1.0)
        material.roughness = 0.6
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    view_layer.update()
    return dict(armature=armature, objects=[armature, body], heights=float(data['height']))


def find():
    obj = bpy.data.objects.get(NAME)
    return obj if obj is not None and obj.type == 'ARMATURE' else None
