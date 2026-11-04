"""Live link client: play what RGB MoCap shows on a rig in Blender.

The app normally streams the character it is displaying: the path of that
character's rest file, the bone palette behind the displayed poses, and one
matrix per palette bone and frame. Blender appends that same rest file, so the
skeleton, skin and textures are the app's own and the streamed matrices are
applied directly - no guessed bone axes. When no character is on screen the app
sends the 22 canonical body joints instead, and each mapped bone is aimed at the
streamed segment direction.
"""
import json
import socket
import struct
import threading
import os
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from . import NAMES
from .mapping import detect

MAGIC_CHARACTER = b'RGFC'
MAGIC_JOINTS = b'RGMF'
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
JOINTS = len(NAMES)

# One primary child per bone gives its segment direction in the joint stream.
CHAIN = {'Hips': 'Spine', 'Spine': 'Spine1', 'Spine1': 'Spine2', 'Spine2': 'Neck',
         'Neck': 'Head', 'LeftShoulder': 'LeftArm', 'RightShoulder': 'RightArm',
         'LeftArm': 'LeftForeArm', 'RightArm': 'RightForeArm',
         'LeftForeArm': 'LeftHand', 'RightForeArm': 'RightHand',
         'LeftUpLeg': 'LeftLeg', 'RightUpLeg': 'RightLeg',
         'LeftLeg': 'LeftFoot', 'RightLeg': 'RightFoot',
         'LeftFoot': 'LeftToeBase', 'RightFoot': 'RightToeBase'}


class LiveClient:
    """Receive the newest pose from the app over localhost."""

    def __init__(self):
        self.socket = None
        self.thread = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.buffer = b''
        self.latest = None
        self.kind = ''
        self.meta = {}
        self.frame = -1
        self.fps = 30.0
        self.received = 0
        self.error = ''

    @property
    def connected(self):
        return self.socket is not None

    def start(self, host='127.0.0.1', port=9187, timeout=2.0):
        self.close()
        self.stopping.clear()
        self.buffer = b''
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as error:
            self.error = 'Cannot reach %s:%d (%s)' % (host, port, error)
            return False
        sock.settimeout(0.5)
        self.socket = sock
        self.thread = threading.Thread(target=self._read, name='rgbmocap-live', daemon=True)
        self.thread.start()
        return True

    def close(self):
        self.stopping.set()
        thread, sock = self.thread, self.socket
        self.thread, self.socket = None, None
        if thread is not None:
            thread.join(1.0)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        with self.lock:
            self.latest = None

    def _read(self):
        while not self.stopping.is_set():
            sock = self.socket
            if sock is None:
                return
            try:
                chunk = sock.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError:
                self.error = 'Connection closed'
                return
            if not chunk:
                self.error = 'Connection closed'
                return
            self.buffer += chunk
            while True:
                if not self.buffer:
                    break
                if self.buffer[:1] == b'{':
                    line, _, rest = self.buffer.partition(b'\n')
                    if not _:
                        break
                    self.buffer = rest
                    try:
                        meta = json.loads(line.decode('utf-8'))
                    except (UnicodeDecodeError, ValueError):
                        continue
                    with self.lock:
                        self.meta = meta
                        self.kind = meta.get('kind', '')
                        self.fps = float(meta.get('fps') or 30.0)
                    continue
                if self.buffer[:4] == MAGIC_CHARACTER:
                    if len(self.buffer) < 14:
                        break
                    frame, fps, count = struct.unpack('<IfH', self.buffer[4:14])
                    size = 14 + count * 16 * 4
                    if len(self.buffer) < size:
                        break
                    values = struct.unpack('<%df' % (count * 16), self.buffer[14:size])
                    self.buffer = self.buffer[size:]
                    matrices = [values[i * 16:i * 16 + 16] for i in range(count)]
                    with self.lock:
                        self.latest = matrices
                        self.kind = 'character'
                        self.frame = int(frame)
                        self.fps = float(fps)
                        self.received += 1
                    continue
                if self.buffer[:4] == MAGIC_JOINTS:
                    size = 12 + JOINTS * 12
                    if len(self.buffer) < size:
                        break
                    frame, fps = struct.unpack('<If', self.buffer[4:12])
                    values = struct.unpack('<%df' % (JOINTS * 3), self.buffer[12:size])
                    self.buffer = self.buffer[size:]
                    points = [values[i * 3:i * 3 + 3] for i in range(JOINTS)]
                    with self.lock:
                        self.latest = points
                        self.kind = 'joints'
                        self.frame = int(frame)
                        self.fps = float(fps)
                        self.received += 1
                    continue
                # Unknown bytes: resync on the next known marker.
                marks = [self.buffer.find(marker) for marker in (MAGIC_CHARACTER, MAGIC_JOINTS, b'{')]
                marks = [index for index in marks if index > 0]
                self.buffer = self.buffer[min(marks):] if marks else b''
                break


_CLIENT = LiveClient()
_STATE = {'timer': False, 'applied': 0, 'bone_map': None, 'armature': None, 'error': '',
          'character': None, 'frame': -1}
_RECORD = {'enabled': False, 'armature': None, 'action': None, 'first_app': None,
           'start_scene': 1, 'keys': 0, 'last_frame': 0, 'previous': {}, 'fps': 30.0,
           'max_bones': 0, 'name': 'RGBMoCap Live'}


def recording():
    action = _RECORD['action']
    return dict(enabled=_RECORD['enabled'], action=action.name if action else '',
                keys=_RECORD['keys'], start=_RECORD['start_scene'],
                last=_RECORD['last_frame'], fps=_RECORD['fps'],
                bones=_RECORD['max_bones'], armature=_RECORD['armature'].name
                if _RECORD['armature'] else '')


def character_state():
    state = _STATE['character']
    if not state:
        return dict(active=False)
    return dict(active=True, armature=state['armature'].name, palette=len(state['palette']),
                blend=state['blend'], mesh_objects=len(state['objects']) - 1)


def start(host, port):
    ok = _CLIENT.start(host, port)
    _STATE['applied'] = 0
    _STATE['bone_map'] = None
    _STATE['character'] = None
    if ok and not _STATE['timer']:
        bpy.app.timers.register(tick, first_interval=1.0 / 60.0, persistent=True)
        _STATE['timer'] = True
    return ok


def load_base(host='127.0.0.1', port=9187, wait=2.0):
    """Build the character the app is showing, straight from its stream.

    The app sends its own base body (or the imported character) as a rest file
    plus bone matrices, so this only waits for that description and builds it -
    no stand-in rig of our own. Returns the armature once it exists, or None
    while the app has not described a character yet.
    """
    if not _STATE['timer'] and not start(host, port):
        return None
    deadline = time.time() + wait
    while time.time() < deadline:
        state = _STATE['character']
        if state is not None:
            return state['armature']
        with _CLIENT.lock:
            kind, meta = _CLIENT.kind, dict(_CLIENT.meta)
        if kind == 'character' and meta.get('blend'):
            try:
                state = _STATE['character'] = _append_character(meta)
                _place(state, meta)
                _STATE['armature'] = state['armature'].name
                _STATE['error'] = ''
            except Exception as error:
                _STATE['error'] = str(error)
                return None
            return state['armature']
        if kind == 'joints':
            _STATE['error'] = 'RGB MoCap is showing a video, not a character. Open a take first.'
            return None
        time.sleep(0.05)
    _STATE['error'] = _STATE['error'] or 'Waiting for RGB MoCap to send its character.'
    return None


def latest_points():
    """The newest joint packet, or None when the app is not sending joints."""
    with _CLIENT.lock:
        if _CLIENT.kind == 'joints' and _CLIENT.latest is not None:
            return list(_CLIENT.latest)
    return None


def current_armature():
    """The armature the live link is currently posing, if any."""
    name = _STATE['armature']
    return bpy.data.objects.get(name) if name else None


def stop():
    if _RECORD['enabled']:
        record_stop()
    _CLIENT.close()
    if _STATE['timer']:
        try:
            bpy.app.timers.unregister(tick)
        except ValueError:
            pass
        _STATE['timer'] = False
    _STATE['bone_map'] = None
    _STATE['armature'] = None
    _STATE['character'] = None


def running():
    return _STATE['timer']


def status():
    return dict(connected=_CLIENT.connected, received=_CLIENT.received, frame=_CLIENT.frame,
                applied_frame=_STATE['frame'], applied=_STATE['applied'],
                fps=_CLIENT.fps, error=_CLIENT.error,
                kind=_CLIENT.kind, meta=_CLIENT.meta, target=_STATE['armature'] or '',
                character=character_state())


def _target(context):
    obj = context.active_object
    if obj is not None and obj.type == 'ARMATURE':
        return obj
    for other in context.scene.objects:
        if other.type == 'ARMATURE':
            return other
    return None


def _matrix(values):
    """Build a 4x4 matrix from the 16 floats of one streamed bone."""
    return Matrix([values[0:4], values[4:8], values[8:12], values[12:16]])


def _depth(bone):
    depth = 0
    while bone.parent is not None:
        bone = bone.parent
        depth += 1
    return depth


def apply_character(armature, palette, base, matrices):
    """Apply one streamed frame of the character's own bone deltas.

    The matrices are pose @ rest^-1 in the rest file's world space, so a bone's
    target is matrix @ rest. The palette is alphabetical, so bases are solved in
    hierarchy order: a child must see its parent's *posed* matrix, not its rest.
    """
    inverse_base = base.inverted()
    bones = armature.data.bones
    targets = {}
    missing = 0
    for index, name in enumerate(palette):
        bone = bones.get(name)
        if bone is None:
            missing += 1
            continue
        targets[name] = (bone, inverse_base @ (_matrix(matrices[index]) @ (base @ bone.matrix_local)))
    posed = {}
    for name, (bone, target) in sorted(targets.items(), key=lambda item: _depth(item[1][0])):
        rest = bone.matrix_local
        parent = bone.parent
        if parent is None:
            basis = rest.inverted() @ target
        else:
            parent_pose = posed.get(parent.name, parent.matrix_local)
            basis = (parent_pose @ parent.matrix_local.inverted() @ rest).inverted() @ target
        pose_bone = armature.pose.bones[name]
        pose_bone.matrix_basis = basis
        posed[name] = target
    bpy.context.view_layer.update()
    return len(posed), missing


def apply_pose(armature, mapping, points):
    """Aim mapped bones at streamed joint directions and place the hips."""
    bones = armature.data.bones
    world = armature.matrix_world
    rest = {}
    for name in NAMES:
        bone = mapping.get(name)
        if bone and bone in bones:
            rest[name] = world.to_3x3() @ bones[bone].matrix_local.to_3x3()
    if not rest:
        raise ValueError('This armature has no mapped humanoid bones.')
    hips = mapping.get('Hips')
    if hips and hips in bones and 'Hips' in rest:
        pose_bone = armature.pose.bones[hips]
        rest_head = world @ bones[hips].head_local
        offset = Vector(points[0]) - rest_head
        pose_bone.location = rest['Hips'].inverted() @ offset
    posed = {}
    for index, name in enumerate(NAMES):
        child = CHAIN.get(name)
        if child is None or child not in mapping or name not in rest:
            continue
        direction = Vector(points[NAMES.index(child)]) - Vector(points[index])
        if direction.length < 1e-6:
            continue
        direction.normalize()
        axis = (rest[name] @ Vector((0.0, 1.0, 0.0))).normalized()
        desired = axis.rotation_difference(direction).to_matrix() @ rest[name]
        parent_index = PARENTS[index]
        parent = NAMES[parent_index]
        if parent_index >= 0 and parent in posed:
            base = posed[parent] @ rest[parent].inverted() @ rest[name]
        else:
            base = rest[name]
        pose_bone = armature.pose.bones[mapping[name]]
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = (base.inverted() @ desired).to_quaternion()
        posed[name] = desired
    bpy.context.view_layer.update()
    return len(posed)


def _world_matrix(obj):
    """World matrix without relying on a depsgraph evaluation yet.

    A freshly appended object keeps a stale ``matrix_world`` until the next
    dependency-graph pass, which in the background may never run before the
    first frame is applied.
    """
    matrix = obj.matrix_basis.copy()
    node = obj
    while node.parent is not None:
        matrix = node.parent.matrix_basis @ node.matrix_parent_inverse @ matrix
        node = node.parent
    return matrix


EXTENDED = chr(92) * 2 + '?' + chr(92)


def _append_character(meta):
    """Append the app's own rest file so rig, skin and rest pose match it."""
    path = Path(str(meta.get('blend') or ''))
    if not path.is_file() and os.name == 'nt' and not str(path).startswith(EXTENDED):
        # Paths past 260 characters need the extended prefix for Blender's Python.
        long_path = Path(EXTENDED + str(path))
        if long_path.is_file():
            path = long_path
    if not path.is_file():
        raise ValueError('Character rest file is missing: ' + str(path))
    name = str(meta.get('armature') or 'Armature')
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE' and obj.name == name and obj.get('rgbmocap_live_blend'):
            return dict(armature=obj, base=Matrix(obj['rgbmocap_live_base']),
                        palette=list(meta.get('palette') or []), objects=[obj],
                        blend=str(path))
    if str(meta.get('rest_format') or 'blend') == 'npz':
        # The app's own body surface and skeleton, built here so the rest pose
        # is exactly the one the streamed matrices are relative to.
        from . import body_import
        built = body_import.build(path)
        armature = built['armature']
        base = _world_matrix(armature)
        armature.matrix_world = base
        armature['rgbmocap_live_blend'] = str(path)
        armature['rgbmocap_live_base'] = [list(row) for row in base]
        bpy.context.view_layer.update()
        return dict(armature=armature, base=base, palette=list(meta.get('palette') or []),
                    objects=built['objects'], blend=str(path))
    if str(meta.get('rest_format') or 'blend') == 'fbx':
        # The app's base body arrives as a one-frame skinned FBX instead of a
        # rest scene: same skeleton, same surface, no Blender needed on its side.
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False,
                                 use_prepost_rot=True, ignore_leaf_bones=True)
        objects = [obj for obj in bpy.context.scene.objects if obj not in before]
        # The rest file carries one bind-pose keyframe; the live stream owns the
        # pose, so nothing may be re-applied on a depsgraph evaluation.
        for obj in objects:
            obj.animation_data_clear()
    else:
        with bpy.data.libraries.load(str(path), link=False) as (available, chosen):
            chosen.objects = list(available.objects)
        objects = []
        for obj in chosen.objects:
            if obj is None:
                continue
            bpy.context.scene.collection.objects.link(obj)
            objects.append(obj)
    armature = next((obj for obj in objects if obj.type == 'ARMATURE'), None)
    if armature is None:
        raise ValueError('No armature in the character rest file')
    # The app's rest file keeps its own import scale on the armature object;
    # its world matrix is the frame every streamed delta is expressed against.
    base = _world_matrix(armature)
    armature.matrix_world = base
    armature['rgbmocap_live_blend'] = str(path)
    armature['rgbmocap_live_base'] = [list(row) for row in base]
    bpy.context.view_layer.update()
    return dict(armature=armature, base=base, palette=list(meta.get('palette') or []),
                objects=objects, blend=str(path))


def _place(state, meta):
    """Match the app's scene placement for this take."""
    transform = meta.get('transform')
    if not transform:
        state['armature'].matrix_world = state['base']
        return
    extra = Matrix(transform)
    state['armature'].matrix_world = extra @ state['base']
    bpy.context.view_layer.update()


def tick():
    """Timer callback: apply the newest received pose."""
    if not _CLIENT.connected:
        _write_status()
        return 1.0
    with _CLIENT.lock:
        points, matrices, kind, frame, meta = (_CLIENT.latest, _CLIENT.latest, _CLIENT.kind,
                                               _CLIENT.frame, dict(_CLIENT.meta))
    if kind == 'character':
        return _tick_character(matrices, frame, meta)
    context = bpy.context
    armature = _target(context)
    if armature is None:
        # The app streams a character as itself; lone body joints arrive only
        # while it shows a video, and there is no character to stand in for it.
        _STATE['error'] = 'Waiting for a character from RGB MoCap. Open a take there.'
        _write_status()
        return 1.0 / 5.0
    if _STATE['armature'] != armature.name:
        mapping, missing, _ = detect(armature)
        _STATE['bone_map'] = mapping
        _STATE['armature'] = armature.name
        _STATE['error'] = '' if not missing else 'Missing bones: ' + ', '.join(missing)
    mapping = _STATE['bone_map'] or {}
    if points is None:
        _write_status()
        return 1.0 / 30.0
    try:
        apply_pose(armature, mapping, points)
        _STATE['applied'] += 1
        if _RECORD['enabled'] and _RECORD['armature'] is armature:
            _key_current(armature, frame, _CLIENT.fps, list(mapping.values()))
    except Exception as error:  # keep the viewport responsive on a bad rig
        _STATE['error'] = str(error)
    _write_status()
    return 1.0 / 60.0


def _tick_character(matrices, frame, meta):
    state = _STATE['character']
    if state is None or state['blend'] != str(meta.get('blend')):
        if not meta.get('blend'):
            return 1.0 / 30.0
        try:
            state = _STATE['character'] = _append_character(meta)
            _place(state, meta)
            _STATE['applied'] = 0
            _STATE['error'] = ''
        except Exception as error:
            _STATE['error'] = str(error)
            _write_status()
            return 1.0 / 5.0
    palette = state['palette']
    if matrices is None or len(matrices) != len(palette):
        _write_status()
        return 1.0 / 30.0
    if state.get('transform') != meta.get('transform'):
        _place(state, meta)
        state['transform'] = meta.get('transform')
    try:
        bones, missing = apply_character(state['armature'], palette, state['base'], matrices)
        _STATE['applied'] += 1
        _STATE['frame'] = int(frame)
        _STATE['armature'] = state['armature'].name
        if missing:
            _STATE['error'] = '%d palette bones are not in the rest file' % missing
        if _RECORD['enabled'] and _RECORD['armature'] is state['armature']:
            _key_current(state['armature'], frame, _CLIENT.fps, list(palette))
    except Exception as error:
        _STATE['error'] = str(error)
    _write_status()
    return 1.0 / 60.0


def record_start(armature, start_frame=None, name=None, fps=None):
    """Key every live pose into a new action on this armature."""
    record_stop()
    scene = bpy.context.scene
    start = int(scene.frame_current if start_frame is None else start_frame)
    if armature.animation_data is None:
        armature.animation_data_create()
    # Start from an empty action: the first key creates it, so Blender's action
    # slot handling stays consistent across 4.x and 5.x.
    armature.animation_data.action = None
    _RECORD.update(enabled=True, armature=armature, action=None, first_app=None,
                   start_scene=max(1, start), keys=0, last_frame=max(1, start) - 1,
                   previous={}, fps=float(fps or _CLIENT.fps or 30.0), max_bones=0)
    _RECORD['name'] = name or 'RGBMoCap Live'
    return recording()


def record_stop():
    """Finish the action and make its sampled keys play at a constant rate."""
    armature = _RECORD['armature']
    action = _RECORD['action']
    _RECORD['enabled'] = False
    summary = recording()
    if armature is not None and armature.animation_data is not None:
        action = armature.animation_data.action or action
        _RECORD['action'] = action
        summary = recording()
        if action is not None and _RECORD['keys']:
            try:
                for fcurve in _iter_fcurves(action):
                    for key in fcurve.keyframe_points:
                        key.interpolation = 'LINEAR'
            except (AttributeError, TypeError):
                pass
        if _RECORD['keys']:
            scene = bpy.context.scene
            scene.frame_start = min(scene.frame_start, _RECORD['start_scene'])
            scene.frame_end = max(_RECORD['last_frame'], _RECORD['start_scene'])
            fps = int(round(_RECORD['fps']))
            if fps > 0 and scene.render.fps != fps:
                scene.render.fps = fps
                summary['fps_changed_to'] = fps
    _RECORD['armature'] = None
    _RECORD['previous'] = {}
    return summary


def _iter_fcurves(action):
    """F-curves of an action, including Blender's slotted actions."""
    if hasattr(action, 'layers') and action.layers:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', ()):
                    for fcurve in bag.fcurves:
                        yield fcurve
        return
    for fcurve in action.fcurves:
        yield fcurve


def _key_current(armature, app_frame, fps, bone_names):
    """Insert one key set per incoming app frame at its matching scene frame."""
    if _RECORD['first_app'] is None:
        _RECORD['first_app'] = int(app_frame)
        _RECORD['fps'] = float(fps or _RECORD['fps'] or 30.0)
    scene_frame = _RECORD['start_scene'] + (int(app_frame) - _RECORD['first_app'])
    if scene_frame < 1:
        return
    keyed = 0
    for bone_name in bone_names:
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.rotation_mode = 'QUATERNION'
        quaternion = pose_bone.rotation_quaternion.copy()
        previous = _RECORD['previous'].get(bone_name)
        # Keep the short arc between sampled keys, otherwise interpolation takes
        # the long way round when the sign flips.
        if previous is not None and quaternion.dot(previous) < 0:
            quaternion.negate()
            pose_bone.rotation_quaternion = quaternion
        _RECORD['previous'][bone_name] = quaternion
        pose_bone.keyframe_insert('rotation_quaternion', frame=scene_frame, group=bone_name)
        translation = pose_bone.location.length
        if translation > 1e-6:
            pose_bone.keyframe_insert('location', frame=scene_frame, group=bone_name)
        keyed += 1
    if armature.animation_data and armature.animation_data.action is not None:
        action = armature.animation_data.action
        if _RECORD['action'] is not action:
            action.name = _RECORD['name']
            _RECORD['action'] = action
    _RECORD['keys'] += 1
    _RECORD['max_bones'] = max(_RECORD['max_bones'], keyed)
    _RECORD['last_frame'] = max(_RECORD['last_frame'], scene_frame)


def _write_status():
    state = status()
    character = state['character']
    detail = ''
    if character.get('active'):
        detail = ' · %s (%d bones)' % (character['armature'], character['palette'])
    text = ('Live: %s · frame %s · %d applied%s%s%s' % (
        'connected' if state['connected'] else 'waiting',
        state['frame'], state['applied'], detail,
        ' · ' + state['target'] if state['target'] and not detail else '',
        ' · ' + state['error'] if state['error'] else ''))
    if _RECORD['enabled']:
        text += ' · recording %d keys (frame %d)' % (_RECORD['keys'], _RECORD['last_frame'])
    try:
        for scene in bpy.data.scenes:
            scene.rgbmocap.live_status = text
    except AttributeError:
        pass
