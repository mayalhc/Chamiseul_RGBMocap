"""Flat-ground snapping for an imported character clip.

The app plants the displayed sole on one common plane; this applies the same
idea inside Blender: measure the lowest skinned vertex of every frame, then
offset the root bone so the sole stays on the floor. Nothing else about the
pose is changed, so the motion keeps its shape and timing.
"""
import bpy
import numpy as np
from mathutils import Vector

FOOT_PREFIXES = ('leftfoot', 'rightfoot', 'lefttoebase', 'righttoebase',
                 'lefttoe', 'righttoe', 'foot', 'toe', 'ball')


def skinned_meshes(armature):
    """Meshes deformed by this armature, so props and helpers are ignored."""
    result = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for modifier in obj.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object == armature:
                result.append(obj)
                break
    return result


def foot_mask(obj):
    """Vertices driven mostly by a foot or toe bone."""
    groups = {group.index: group.name for group in obj.vertex_groups}
    foot = {index for index, name in groups.items()
            if any(name.split(':')[-1].lower().startswith(prefix) for prefix in FOOT_PREFIXES)}
    if not foot:
        return None
    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            if element.group in foot and element.weight > 0.3:
                mask[vertex.index] = True
                break
    return mask if mask.any() else None


def frame_range(armature):
    start, end = bpy.context.scene.frame_start, bpy.context.scene.frame_end
    if armature.animation_data and armature.animation_data.action:
        first, last = armature.animation_data.action.frame_range
        start, end = max(start, int(round(first))), min(end, int(round(last)))
    return start, max(start, end)


def _lowest(objects, masks):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lowest = None
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            count = len(mesh.vertices)
            if not count:
                continue
            coordinates = np.empty(count * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', coordinates)
            coordinates = coordinates.reshape(-1, 3)
            mask = masks.get(obj.name) if masks else None
            if mask is not None:
                coordinates = coordinates[mask]
                if not len(coordinates):
                    continue
            matrix = np.array(evaluated.matrix_world, dtype=np.float64)
            heights = coordinates @ matrix[2, :3] + matrix[2, 3]
            value = float(heights.min())
            lowest = value if lowest is None else min(lowest, value)
        finally:
            evaluated.to_mesh_clear()
    return lowest


def _bone_lowest(armature, names):
    lowest = None
    for name in names:
        bone = armature.pose.bones.get(name)
        if bone is None:
            continue
        for point in (bone.head, bone.tail):
            value = (armature.matrix_world @ point).z
            lowest = value if lowest is None else min(lowest, value)
    return lowest


def measure(armature, foot_only=True, step=1):
    """Lowest point of every frame, in world metres."""
    objects = skinned_meshes(armature)
    masks = {}
    if objects and foot_only:
        for obj in objects:
            mask = foot_mask(obj)
            if mask is not None:
                masks[obj.name] = mask
    start, end = frame_range(armature)
    values = []
    for frame in range(start, end + 1, max(1, step)):
        bpy.context.scene.frame_set(frame)
        value = _lowest(objects, masks)
        if value is None:
            value = _bone_lowest(armature, ('LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot'))
        values.append(float('nan') if value is None else value)
    heights = np.asarray(values, dtype=np.float64)
    if not np.isfinite(heights).all():
        raise ValueError('No skinned mesh or foot bone was found to measure the floor.')
    return dict(heights=heights, start=start, end=end, step=max(1, step),
                foot_only=bool(masks), meshes=len(objects))


def _root_bone(armature):
    for name in ('Hips', 'hips', 'Pelvis', 'pelvis'):
        if armature.pose.bones.get(name):
            return armature.pose.bones[name]
    roots = [bone for bone in armature.data.bones if bone.parent is None]
    return armature.pose.bones[roots[0].name] if roots else None


def _keyed_root(armature, offsets):
    """Add a world-Z offset to the root bone and bake it per frame.

    The base pose is read for every frame before anything is written, so
    inserting keys cannot change the motion that later frames sample.
    """
    bone = _root_bone(armature)
    scene = bpy.context.scene
    start, end = frame_range(armature)
    frames = list(range(start, end + 1))
    if len(frames) != len(offsets):
        raise ValueError('Offset count does not match the frame range.')
    if bone is not None:
        # A bone's location is expressed along its own rest axes, which the
        # armature object's rotation also turns; convert the world lift once.
        rest = armature.data.bones[bone.name].matrix_local
        axes = armature.matrix_world.to_3x3() @ rest.to_3x3()
        inverse = axes.inverted()
        bases = []
        for frame in frames:
            scene.frame_set(frame)
            bases.append(bone.location.copy())
        for index, (frame, base) in enumerate(zip(frames, bases)):
            scene.frame_set(frame)
            bone.location = base + inverse @ Vector((0.0, 0.0, float(offsets[index])))
            bone.keyframe_insert('location', frame=frame)
        scene.frame_set(start)
        return dict(target='bone:' + bone.name, frames=len(frames))

    bases = []
    for frame in frames:
        scene.frame_set(frame)
        bases.append(armature.location.copy())
    for index, (frame, base) in enumerate(zip(frames, bases)):
        scene.frame_set(frame)
        location = base.copy()
        location.z += float(offsets[index])
        armature.location = location
        armature.keyframe_insert('location', frame=frame)
    scene.frame_set(start)
    return dict(target='object:' + armature.name, frames=len(frames))


def snap(armature, floor=0.0, mode='PLANT', smooth=0, foot_only=True):
    """Move the character so its lowest point sits on the floor plane.

    PLANT  corrects every frame, so a contacted sole never floats or sinks.
    DROP   applies one offset from the lowest frame of the whole clip.
    AUTO   drops to a low percentile, keeping deliberately airborne frames
           above the plane instead of pulling the body down to them.
    """
    data = measure(armature, foot_only=foot_only)
    heights = data['heights']
    if mode == 'PLANT':
        offsets = floor - heights
    elif mode == 'DROP':
        offsets = np.full_like(heights, floor - heights.min())
    elif mode == 'AUTO':
        offsets = np.full_like(heights, floor - np.percentile(heights, 5))
    else:
        raise ValueError('Unknown ground mode: ' + mode)
    if smooth > 0:
        window = np.ones(int(smooth) * 2 + 1)
        window /= window.sum()
        padded = np.pad(offsets, (len(window) // 2, len(window) // 2), mode='edge')
        offsets = np.convolve(padded, window, mode='valid')
    receipt = _keyed_root(armature, offsets)
    after = measure(armature, foot_only=foot_only)
    residual = after['heights'] - floor
    return dict(
        mode=mode, floor=floor, count=len(heights), foot_only=data['foot_only'],
        meshes=data['meshes'], before_min=float(heights.min()), before_max=float(heights.max()),
        after_min=float(after['heights'].min()), after_max=float(after['heights'].max()),
        residual_low=float(residual.min()), residual_high=float(residual.max()),
        offset_min=float(offsets.min()), offset_max=float(offsets.max()), **receipt)
