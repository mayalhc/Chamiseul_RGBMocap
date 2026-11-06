# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: GPL-3.0-or-later (see blender_addon/LICENSE)
"""FBX and glTF binary export using the app's export conventions.

Only options this Blender supports are passed, so the same addon runs on the
Blender versions the app installs and on newer builds whose operators gained or
renamed properties.
"""
import bpy
from pathlib import Path

from .ground import skinned_meshes


def _supported(operator, options):
    """Drop options the running Blender does not expose."""
    properties = operator.get_rna_type().properties.keys()
    return {key: value for key, value in options.items() if key in properties}


def select(armature, include_mesh=True):
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    if include_mesh:
        for mesh in skinned_meshes(armature):
            mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature


def export_fbx(path, armature, include_mesh=True):
    """Bake the bone animation into one FBX; no object-level transform keys."""
    select(armature, include_mesh)
    types = {'ARMATURE', 'MESH'} if include_mesh else {'ARMATURE'}
    options = dict(
        filepath=str(path), use_selection=True, object_types=types,
        add_leaf_bones=True, bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
        bake_anim_simplify_factor=0.0, bake_anim_force_startend_keying=True,
        use_mesh_modifiers=True, path_mode='COPY', embed_textures=True,
        axis_forward='-Z', axis_up='Y', apply_scale_options='FBX_SCALE_NONE',
        use_space_transform=True, bake_space_transform=False)
    bpy.ops.export_scene.fbx(**_supported(bpy.ops.export_scene.fbx, options))
    return Path(path)


def export_glb(path, armature, include_mesh=True):
    """One binary glTF with the armature's rest joints and sampled animation."""
    select(armature, include_mesh)
    options = dict(
        filepath=str(path), export_format='GLB', use_selection=True, export_apply=True,
        export_animations=True, export_animation_mode='ACTIVE_ACTIONS',
        export_anim_single_armature=True, export_force_sampling=True,
        export_frame_range=True, export_anim_slide_to_zero=True,
        export_optimize_animation_size=False, export_reset_pose_bones=True,
        export_rest_position_armature=True, export_leaf_bone=False,
        export_def_bones=False, export_skins=include_mesh, export_yup=True,
        export_materials='EXPORT', export_cameras=False, export_lights=False,
        export_extras=False)
    bpy.ops.export_scene.gltf(**_supported(bpy.ops.export_scene.gltf, options))
    return Path(path)
