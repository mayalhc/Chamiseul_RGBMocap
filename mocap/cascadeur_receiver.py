# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: MIT (see LICENSE in the repository root)
"""Live pose receiver that runs inside Cascadeur.

Nothing imports this file. `mocap.cascadeur_link` posts its source to
Cascadeur's script server (127.0.0.1:8765/run), which execs it once in a
namespace of its own; from then on an HTTP server on 127.0.0.1:9566 and a
QTimer on Cascadeur's Qt loop live inside the application until stopped.

Two threads, and the line between them is the whole design. The HTTP thread
only ever appends to a queue or answers from something the timer left behind.
The QTimer, on Cascadeur's main thread, is the ONLY thing that may touch the
scene - reading included: listing object names from the request thread has
taken Cascadeur down outright.

What the sender gives per joint is how far it has TURNED from its own rest,
measured in that joint's own rest frame. This side composes that turn onto the
character's own rest: `local_rotation = rest_local @ turn`. An identity turn
leaves the character in its own rest pose, so a T-pose is checkable before any
pose is trusted, and a character whose rest differs from the sender's is turned
by the same amount rather than being forced onto another skeleton's axes.
Everything below the root is rotation only; the root alone is placed.

The resulting orientation is turned into the local rotation its parent implies,
because `local_rotation` is the channel Cascadeur stores: `global_rotation` is
derived from it, and a write there is accepted, keyed, and quietly reshaped by
the next update.

A 33 ms timer measures out at about 21 Hz on 2026.2, with a scene write inside
it costing about 0.26 ms, which is why the tick can drive the character while
the app is still moving.
"""

import hashlib
import json
import tempfile
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import csc
import pycsc
from PySide6 import QtCore

HOST = "127.0.0.1"
PORT = 9566
VERSION = "2.4.0"

# Centimetres, as everything on this side is.
SCALE = 100.0

INTERVAL_MS = 33

# Frames waiting to be written. Bounded: if the writer falls behind the
# sender, dropping the oldest keeps the link live instead of building a lag
# that never clears.
QUEUE_LIMIT = 240

# The animation is lengthened a few seconds at a time. Growing is a modify
# session of its own, so doing it per frame would double the sessions.
GROW_BLOCK = 300

_state = None


def _rows(matrix):
    """A csc rotation matrix as plain rows, whatever shape it arrives in."""
    try:
        return [[float(matrix[r][c]) for c in range(3)] for r in range(3)]
    except TypeError:
        return [[float(matrix[r * 3 + c]) for c in range(3)] for r in range(3)]


def _multiply(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
            for r in range(3)]


def _transpose(matrix):
    return [[matrix[c][r] for c in range(3)] for r in range(3)]


def _orthonormal(matrix):
    """The rotation part alone, with the bind matrix's scale divided out."""
    out = [[0.0] * 3 for _ in range(3)]
    for column in range(3):
        length = sum(matrix[row][column] ** 2 for row in range(3)) ** 0.5
        for row in range(3):
            out[row][column] = (matrix[row][column] / length if length
                                else matrix[row][column])
    return out


def _invert(matrix):
    """The inverse of a 3x3, by cofactors. Only ever given bind matrices."""
    m = matrix
    a = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    b = m[1][2] * m[2][0] - m[1][0] * m[2][2]
    c = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    det = m[0][0] * a + m[0][1] * b + m[0][2] * c
    if abs(det) < 1e-12:
        return None
    adj = [
        [a, m[0][2] * m[2][1] - m[0][1] * m[2][2],
         m[0][1] * m[1][2] - m[0][2] * m[1][1]],
        [b, m[0][0] * m[2][2] - m[0][2] * m[2][0],
         m[0][2] * m[1][0] - m[0][0] * m[1][2]],
        [c, m[0][1] * m[2][0] - m[0][0] * m[2][1],
         m[0][0] * m[1][1] - m[0][1] * m[1][0]],
    ]
    return [[adj[r][c] / det for c in range(3)] for r in range(3)]


def _swing(source, target):
    """The rotation that carries one direction onto another, shortest way."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    if la < 1e-9 or lb < 1e-9:
        return None
    a = a / la
    b = b / lb
    cross = np.cross(a, b)
    cosine = float(np.dot(a, b))
    if cosine < -0.999999:
        # Exactly opposite: any perpendicular axis is half a turn.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        return (2.0 * np.outer(axis, axis) - np.eye(3)).tolist()
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    rotation = np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)
    return rotation.tolist()


def _separated(a, b, minimum=0.20):
    """Whether two directions define a plane well enough to fix a roll.

    0.20 is sin(11.5 degrees). Below that the plane spanned by the pair is
    dominated by noise: the roll computed from it flips by up to half a turn
    between frames (measured: the left forearm turned 131 degrees in one frame
    while its direction moved 15.8 degrees). When the pair is this close, the
    minimal swing is used instead and the character keeps its own roll.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return False
    return float(np.linalg.norm(np.cross(a / na, b / nb))) >= minimum


def _basis(along, across):
    """An orthonormal frame from two directions, or None if they are parallel."""
    a = np.asarray(along, dtype=np.float64)
    length = float(np.linalg.norm(a))
    if length < 1e-9:
        return None
    a = a / length
    b = np.asarray(across, dtype=np.float64)
    b = b - a * float(np.dot(a, b))
    length = float(np.linalg.norm(b))
    if length < 1e-6:
        return None
    b = b / length
    return np.stack((a, np.cross(b, a), b), axis=1)


def _plane(rest_along, rest_across, along, across):
    """Point the first direction at its target, and ROLL onto the second.

    Two directions cannot both be matched across characters: the angle
    between them is a body proportion (how far apart the legs sit), not a
    pose. The first fixes where the bone points; the second, projected flat
    against it, fixes only the roll, which is the part a character of another
    shape can share.
    """
    swing = _swing(rest_along, along)
    if swing is None:
        return None
    spin = np.asarray(along, dtype=np.float64)
    length = float(np.linalg.norm(spin))
    if length < 1e-9:
        return None
    spin = spin / length
    moved = np.asarray(swing, dtype=np.float64) @ np.asarray(rest_across, dtype=np.float64)
    moved = moved - spin * float(np.dot(moved, spin))
    wanted = np.asarray(across, dtype=np.float64)
    wanted = wanted - spin * float(np.dot(wanted, spin))
    if float(np.linalg.norm(moved)) < 1e-6 or float(np.linalg.norm(wanted)) < 1e-6:
        return swing
    moved = moved / float(np.linalg.norm(moved))
    wanted = wanted / float(np.linalg.norm(wanted))
    angle = np.arctan2(float(np.dot(np.cross(moved, wanted), spin)),
                       float(np.dot(moved, wanted)))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = spin
    turn = np.array([
        [cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin],
        [y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin],
        [z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)],
    ])
    return (turn @ np.asarray(swing, dtype=np.float64)).tolist()


def _align(rest_along, rest_across, along, across):
    """The rotation carrying one bone's frame onto another's, roll included.

    Direction alone leaves the roll free: a bone that points right but is
    twisted reads as a knot around the mesh. The second direction - the plane
    the joint bends in - fixes that, and both come from joint positions.
    """
    first = _basis(rest_along, rest_across)
    second = _basis(along, across)
    if first is None or second is None:
        return None
    return (second @ first.T).tolist()


def _rest_pose(scene, names):
    """Each joint's rest orientation, from the character's bind matrices.

    The sender measures every rotation as a turn AWAY from its own rest, so
    this is what turns that into where the joint must point here: compose the
    turn onto the orientation this character actually rests in. Sending world
    orientations instead assumes both characters rest the same way, and a
    Mixamo character does not - its skin twisted into a knot the moment that
    was tried.

    Joints the skin does not bind keep their parent's correction through the
    hierarchy and are left out; those are sent absolutely.
    """
    py_scene = pycsc.wrap(scene)
    viewer = scene.model_viewer()
    behaviours = viewer.behaviour_viewer()
    assets = scene.assets_manager()

    found = {}
    places = {}
    for mesh_obj in py_scene.get_scene_objects(with_behaviour="MeshObject"):
        behaviour = mesh_obj.get_behaviour("MeshObject")
        dependency = behaviours.get_behaviour_asset(behaviour.handle,
                                                    "mesh_dependency")
        if dependency.is_null():
            continue
        inverse = assets.at(dependency).inverse_bind_matrices()
        for index, joint in enumerate(behaviour.linked_objects.get()):
            if index >= len(inverse):
                break
            name = viewer.get_object_name(joint.unwrap())
            rows = [[float(inverse[index][r][c]) for c in range(3)]
                    for r in range(3)]
            back = _invert(rows)
            if back is None:
                continue
            rotation = _orthonormal(back)
            found[name] = rotation
            # The bind matrix's translation, undoing the inverse: this is where
            # the joint sits at rest, so bone directions can be compared.
            move = [float(inverse[index][r][3]) for r in range(3)]
            places[name] = [-sum(rotation[r][k] * move[k] for k in range(3))
                            for r in range(3)]
    kept = {name: found[name] for name in names if name in found}
    kept_places = {name: places[name] for name in names if name in places}
    return kept, kept_places


def _fixed_section():
    """A section whose keys interpolate the way Cascadeur's own takes do.

    `Section()` comes up on STEP, and STEP is a held pose: a take written into
    STEP sections plays but cannot be worked on, because AutoPosing has nothing
    to solve and never draws its controllers.
    """
    section = csc.layers.layer.Section()
    section.interval.interpolation = csc.layers.layer.Interpolation.FIXED
    return section


def _close_server(server):
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass


def _current_scene():
    view = csc.app.get_application().get_scene_manager().current_scene()
    return view.domain_scene() if view is not None else None


class _Link:

    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.server = None
        self.thread = None
        self.timer = None
        self.channels = None
        self.rests = None
        self.rest_positions = None
        self.rest_local = {}
        self.rest_children = {}
        self.names = None
        self.frame = 0
        self.written = 0
        self.received = 0
        self.dropped = 0
        self.errors = []
        self.notes = []
        self.debug_paths = {}
        self.last_write_ms = 0.0
        self.started = time.time()
        self.stopping = False
        self.length = 0
        self.last_at = 0
        self.follow = False
        self.names_seen = []
        self.names_at = 0.0
        self.bind_at = 0.0
        self.wanted_frame = None
        self.direct_names = set()
        self.root_map = None
        self.rest_cache = None
        self.rest_world = {}
        self.rest_world_positions = {}
        self.path_counts = {"positions": 0, "deltas": 0, "aim": 0}
        self.debug = {}
        self.app_rest_dirs = []
        self.finger_corr = {}
        self.bind_local_inv = {}
        self.delta_mode = False
        self.rig = None
        self.rig_rest = None

    # -- the sending side, on the HTTP thread ---------------------------
    def offer(self, payload):
        frames = payload.get("frames") or []
        with self.lock:
            for frame in frames:
                if len(self.queue) >= QUEUE_LIMIT:
                    self.queue.popleft()
                    self.dropped += 1
                self.queue.append(frame)
                self.received += 1
            depth = len(self.queue)
        return {"ok": True, "version": VERSION, "queued": depth,
                "notes": list(self.notes),
                "paths_by_joint": dict(self.debug_paths),
                "written": self.written, "dropped": self.dropped}

    # -- the writing side, on Cascadeur's main thread ------------------
    def _bind(self, scene):
        """Work out which joints of the open scene this stream can drive.

        Parents are resolved here too, because a local rotation only means
        something relative to the joint above it. A parent that is not part of
        the stream - the root object a character hangs under, say - is
        measured once and treated as the fixed frame it is.
        """
        viewer = scene.model_viewer()
        behaviours = viewer.behaviour_viewer()
        data = scene.data_viewer()
        frame = scene.get_current_frame()
        objects = {viewer.get_object_name(o): o for o in viewer.get_objects()}

        layers = scene.layers_viewer()
        channels = {}
        for name in self.names:
            object_id = objects.get(name)
            if object_id is None:
                continue
            transform = behaviours.get_behaviour_by_name(object_id, "Transform")
            if transform.is_null():
                continue
            local_rotation = behaviours.get_behaviour_data(transform, "local_rotation")
            local_position = behaviours.get_behaviour_data(transform, "local_position")
            if local_rotation.is_null():
                continue
            parent = None
            fixed_rotation = None
            fixed_position = None
            basic = behaviours.get_behaviour_by_name(object_id, "Basic")
            if not basic.is_null():
                parent_id = behaviours.get_behaviour_object(basic, "parent")
                if parent_id is not None and not parent_id.is_null():
                    parent_name = viewer.get_object_name(parent_id)
                    if parent_name in self.names:
                        parent = parent_name
                    elif parent_name in objects:
                        # Outside the stream: read where it stands once.
                        other = behaviours.get_behaviour_by_name(objects[parent_name], "Transform")
                        if not other.is_null():
                            try:
                                rotation = behaviours.get_behaviour_data(other, "global_rotation")
                                position = behaviours.get_behaviour_data(other, "global_position")
                                fixed_rotation = _rows(
                                    data.get_data_value(rotation, frame).to_rotation_matrix())
                                where = data.get_data_value(position, frame)
                                fixed_position = [float(where[0]), float(where[1]),
                                                  float(where[2])]
                            except Exception:
                                fixed_rotation = None
                                fixed_position = None
            # The joint's OWN layer. A take written into every layer at once
            # was accepted and then partly overridden: the sample document
            # here carries thirteen layers, and keying all of them at a frame
            # leaves the joint reading a layer this stream never meant to
            # touch.
            channels[name] = dict(rotation=local_rotation, position=local_position,
                                  layer=layers.layer_id_by_obj_id(object_id),
                                  object_id=object_id,
                                  parent=parent, fixed_rotation=fixed_rotation,
                                  fixed_position=fixed_position)
        self.channels = channels
        # A scene in Rig Mode owns its joints: the rig recomputes every joint
        # and box from its points on each update and silently puts a written
        # joint rotation back (measured: the value reads back unchanged). The
        # points are the inputs, so those are what gets written there.
        self.rig = self._find_rig(scene, objects, channels)
        # A character a previous live run keyed no longer rests at frame 0,
        # and a rest read out of a take pose twists every write that follows.
        # The one moment the rest is certainly known is the pristine single-
        # frame import, so it is banked then - in memory and in a
        # fingerprinted temp file - and put back here on any later bind.
        # The skin's inverse bind matrices are NOT a substitute: on an
        # imported hierarchy they live in the importer's own space, and a
        # restore built on them (receiver 2.1.0) put this character's arms
        # and legs wherever it liked.
        keyed = False if self.rig else self._animation_keyed(data)
        rest_frame = 0
        if keyed:
            cache = self.rest_cache or self._load_rest_cache(self._bone_marks(data))
            scratch = None
            if cache is not None:
                scratch = self._apply_rest_cache(scene, cache)
            if scratch is not None:
                self.rest_cache = cache
                rest_frame = scratch
            else:
                # No banked rest fitted. Writing anyway would compose every
                # pose onto a take pose read as a rest - the knot the user
                # sees. Refuse to bind; the queued frames are kept and the
                # bind is retried until a rest is available.
                self._note("this character is keyed and no banked rest fits "
                           "it: re-import it with "
                           "tools/send-base-body-to-cascadeur.py, or run the "
                           "link once on a fresh import and keep this "
                           "Cascadeur session")
                self.channels = None
                return None
        # The character's own bone offset for each joint, in its parent's
        # frame, read at the rest frame - now the rest, whether pristine or
        # put back on a scratch frame. This is the rig's true rest geometry,
        # which a Mixamo character's mesh bind pose does not agree with for
        # its branching joints.
        for name, channel in channels.items():
            offset = None
            try:
                value = data.get_data_value(channel['position'], rest_frame)
                offset = [float(value[0]), float(value[1]), float(value[2])]
            except Exception:
                offset = None
            channel['offset'] = offset
        if not keyed and not self.rig:
            self._capture_rest_cache(data)
        # Each joint's rest orientation in the WORLD, read from the character
        # itself at frame 0 now that the rest is back. World frames carry no
        # bone-roll convention - which is exactly what makes the world-turn
        # transfer frame-proof where the local form knotted.
        self.rest_world = {}
        self.rest_world_positions = {}
        for name, channel in channels.items():
            try:
                transform = behaviours.get_behaviour_by_name(channel['object_id'],
                                                             "Transform")
                rotation = behaviours.get_behaviour_data(transform,
                                                         "global_rotation")
                value = data.get_data_value(rotation, rest_frame).to_rotation_matrix()
                self.rest_world[name] = [[float(value[r][c]) for c in range(3)]
                                         for r in range(3)]
            except Exception:
                self.rest_world[name] = None
            try:
                position = behaviours.get_behaviour_data(transform,
                                                         "global_position")
                where = data.get_data_value(position, rest_frame)
                self.rest_world_positions[name] = [float(where[r])
                                                   for r in range(3)]
            except Exception:
                self.rest_world_positions[name] = None
        if self.rig:
            self._rig_rest(data, rest_frame)
        self.rests, self.rest_positions = _rest_pose(scene, self.names)
        # The channel semantics, measured - never assumed. An FBX whose Lcl
        # Rotation starts at zero carries the rest in PreRotation: the
        # channel is a DELTA from the bind, and the writes have to go
        # through the bind local. A channel whose rest value equals the
        # world-derived absolute local is absolute. The two FBX origins this
        # app feeds (the app's own writer and Blender's) differ, and writing
        # the wrong one knots the character the moment the stream plays.
        self.bind_local_inv = {}
        self.bone_rest_inv = {}
        self.delta_mode = False
        # The channel semantics, PROBED - never derived from the same values
        # it would be compared against. A known test rotation is written to
        # the root joint and the world rotation is read back: absolute
        # semantics answer world = parent @ R_test; delta semantics (the
        # app's own FBX, whose Lcl Rotation starts at zero with the rest in
        # PreRotation) answer world = parent @ bone_rest @ R_test - and the
        # probe hands back the bone rest itself.
        # In Rig Mode a rig-driven joint ignores the probe, which would read
        # as a delta channel, so a joint the rig leaves alone is probed.
        free = [name for name in self._order()
                if not self.rig or name not in self.rig['points']]
        probe_name = free[0] if free else None
        py_scene = pycsc.wrap(scene)
        if probe_name:
            probe_channel = channels.get(probe_name) or {}
            parent_name = probe_channel.get('parent')
            parent_world0 = ((self.rest_world or {}).get(parent_name)
                             if parent_name else probe_channel.get('fixed_rotation'))
            world0 = (self.rest_world or {}).get(probe_name)
            try:
                original = data.get_data_value(probe_channel['rotation'],
                                               rest_frame).to_rotation_matrix()
                angle_t = np.radians(30.0)
                r_test = [[1.0, 0.0, 0.0],
                          [0.0, float(np.cos(angle_t)), -float(np.sin(angle_t))],
                          [0.0, float(np.sin(angle_t)), float(np.cos(angle_t))]]

                def put_test(model, update, updater):
                    editor = model.data_editor()
                    editor.set_data_value(probe_channel['rotation'], rest_frame,
                                          csc.math.Rotation.from_rotation_matrix(r_test))
                    updater.run_update({probe_channel['rotation']}, rest_frame)

                if py_scene.modify_update("RGB MoCap live: probe", put_test) is not False:
                    gr = behaviours.get_behaviour_data(
                        behaviours.get_behaviour_by_name(probe_channel['object_id'],
                                                         'Transform'),
                        'global_rotation')
                    w = data.get_data_value(gr, rest_frame).to_rotation_matrix()
                    w_abs = _multiply(parent_world0, r_test)
                    r = np.asarray(w) @ np.asarray(w_abs).T
                    trace = max(-1.0, min(3.0, float(np.trace(r))))
                    angle = float(np.degrees(np.arccos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))))

                    def put_back(model, update, updater):
                        editor = model.data_editor()
                        editor.set_data_value(probe_channel['rotation'], rest_frame,
                                              csc.math.Rotation.from_rotation_matrix(original))
                        updater.run_update({probe_channel['rotation']}, rest_frame)

                    py_scene.modify_update("RGB MoCap live: probe restore", put_back)
                    if angle < 2.0:
                        self.delta_mode = False
                    else:
                        self.delta_mode = True
                        bone_rest = _multiply(_multiply(_transpose(parent_world0),
                                                        np.asarray(w, dtype=np.float64).tolist()),
                                              _invert(r_test))
                        bone_rest_inv = _invert(bone_rest)
                        if bone_rest_inv is not None:
                            self.bone_rest_inv[probe_name] = bone_rest_inv
            except Exception:
                self._note_trace()
        if self.delta_mode:
            # Every joint of the same document shares the semantics; each
            # bone's own rest comes from its bind state.
            for name in self._order():
                channel = channels.get(name) or {}
                parent_name = channel.get('parent')
                parent_world0 = ((self.rest_world or {}).get(parent_name)
                                 if parent_name else channel.get('fixed_rotation'))
                world0 = (self.rest_world or {}).get(name)
                try:
                    channel0 = data.get_data_value(channel['rotation'],
                                                   rest_frame).to_rotation_matrix()
                except Exception:
                    continue
                if parent_world0 is None or world0 is None:
                    continue
                bone_rest = _multiply(_multiply(_transpose(parent_world0),
                                                world0), _invert(channel0))
                bone_rest_inv = _invert(bone_rest)
                if bone_rest_inv is not None:
                    self.bone_rest_inv[name] = bone_rest_inv
        self._note("channel semantics: %s (probed on %s)"
                   % ('delta-from-bind' if self.delta_mode else 'absolute',
                      probe_name or 'nothing'))
        # Per-joint correction between the sender's skeleton rest and this
        # character's: C = char_rest @ app_rest^T. The sender's world
        # rotations then map straight onto this character - the constant
        # roll difference of every finger bone cancels exactly.
        self.finger_corr = {}
        dirs = self.app_rest_dirs or []
        if len(dirs) == len(self.names):
            children_of = {}
            for name in self._order():
                parent = (channels.get(name) or {}).get('parent')
                if parent:
                    children_of.setdefault(parent, []).append(name)
            for row, name in enumerate(self.names):
                if not ('Hand' in name and any(tag in name for tag in
                        ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'))):
                    continue
                channel = channels.get(name) or {}
                parent_name = channel.get('parent')
                mine = (self.rest_world_positions or {}).get(name)
                parent_pos = (self.rest_world_positions or {}).get(parent_name)                     if parent_name else None
                if mine is None or parent_pos is None:
                    continue
                sent = dirs[row] if row < len(dirs) else None
                if not sent or len(sent) < 2:
                    continue
                app_a = np.asarray(sent[0], dtype=np.float64)
                app_b = np.asarray(sent[1], dtype=np.float64)
                # the character's own rest directions: the continuing child
                # first, the incoming direction as the roll reference
                child_name = children_of.get(name, [None])[0]
                char_a = None
                if child_name:
                    child_pos = (self.rest_world_positions or {}).get(child_name)
                    if child_pos:
                        char_a = np.asarray(child_pos, dtype=np.float64) - np.asarray(mine, dtype=np.float64)
                char_b = np.asarray(mine, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64)
                la = float(np.linalg.norm(char_a)) if char_a is not None else 0.0
                lb = float(np.linalg.norm(char_b))
                if la < 1e-6 or lb < 1e-6:
                    continue
                char_a = char_a / la
                char_b = char_b / lb
                first = _basis(char_a, char_b)
                second = _basis(app_a, app_b)
                if first is None or second is None:
                    continue
                self.finger_corr[name] = _multiply(first, _transpose(second))
        # Each joint's rest rotation RELATIVE TO ITS PARENT. A local rotation
        # only means anything against the frame it is written into, and this is
        # that frame for every joint on this character.
        # Each joint's rest rotation RELATIVE TO ITS PARENT, read from the
        # character itself: `local_rotation` at the frame it was loaded on IS
        # the rest, because a Cascadeur joint stores its rest in its bone and
        # its rotation channel starts there. Deriving it from the skin's bind
        # matrices instead is one step removed - those describe the bind pose
        # the mesh was skinned in, converted by the importer - and a character
        # whose skeleton rests elsewhere than its skin is bound was twisted
        # from the first frame by exactly that difference.
        self.rest_local = {}
        for name, info in channels.items():
            mine = None
            try:
                value = data.get_data_value(info['rotation'], rest_frame)
                rows = value.to_rotation_matrix()
                mine = [[float(rows[r][c]) for c in range(3)] for r in range(3)]
            except Exception:
                mine = None
            if mine is None:
                parent = info.get('parent')
                above = (self.rests or {}).get(parent) if parent else None
                rest = (self.rests or {}).get(name)
                if rest is None:
                    continue
                mine = (np.asarray(rest, dtype=np.float64) if above is None
                        else np.asarray(above, dtype=np.float64).T
                        @ np.asarray(rest, dtype=np.float64)).tolist()
            self.rest_local[name] = mine
        # One child per joint is enough to say where that joint points.
        self.rest_children = {}
        for child, info in channels.items():
            parent = info.get("parent")
            if parent and parent not in self.rest_children:
                self.rest_children[parent] = child
        self._calibrate_root(scene, data)
        return channels

    def _find_rig(self, scene, objects, channels):
        """The rig points that drive these joints, or None for a plain skeleton.

        Quick Rigging gives each joint `<joint>_MainPoint` and, as the joint
        needs, `_AdditionalPoint`, `_DirectionPoint`, `_Self0Point` and a
        `_LimbDir`. Joints without any (the fingers, Spine2) are not rig
        driven and keep taking their rotation directly.
        """
        viewer = scene.model_viewer()
        behaviours = viewer.behaviour_viewer()
        layers = scene.layers_viewer()
        points = {}
        mains = 0
        for pname, object_id in objects.items():
            if '_' not in pname:
                continue
            joint, suffix = pname.rsplit('_', 1)
            if joint not in channels:
                continue
            try:
                if viewer.get_object_type_name(object_id) not in ('Point', 'Limb Direction'):
                    continue
                transform = behaviours.get_behaviour_by_name(object_id, "Transform")
                if transform.is_null():
                    continue
                position = behaviours.get_behaviour_data(transform, "global_position")
                if position.is_null():
                    continue
                layer = layers.layer_id_by_obj_id(object_id)
            except Exception:
                continue
            points.setdefault(joint, []).append(
                dict(name=pname, position=position, layer=layer, rest=None))
            if suffix == "MainPoint":
                mains += 1
        if mains < 3:
            return None
        return {"points": points}

    def _rig_rest(self, data, rest_frame):
        """The rig's rest: joint frames and every point, banked on first sight.

        Writing a take moves the rig itself, so a rest read later is a pose.
        The first untouched read is kept - in memory, and in a temp file named
        by the rig's bone geometry for a later session - and used from then on.
        """
        marks = self._bone_marks(data)
        cache = self.rig_rest
        if cache is not None and cache.get("marks") != marks:
            cache = None
        if cache is None:
            cache = self._load_rig_cache(marks)
        fresh = cache is None
        if fresh:
            cache = {"marks": marks, "rot": {}, "pos": {}, "points": {}}
            for name in self.channels:
                cache["rot"][name] = (self.rest_world or {}).get(name)
                cache["pos"][name] = (self.rest_world_positions or {}).get(name)
            for items in self.rig["points"].values():
                for item in items:
                    try:
                        where = data.get_data_value(item["position"], rest_frame)
                        cache["points"][item["name"]] = [float(where[r]) for r in range(3)]
                    except Exception:
                        pass
        self.rig_rest = cache
        self.rest_world = dict(cache["rot"])
        self.rest_world_positions = dict(cache["pos"])
        for joint, items in list(self.rig["points"].items()):
            for item in items:
                found = cache["points"].get(item["name"])
                item["rest"] = (np.asarray(found, dtype=np.float64)
                                if found is not None else None)
        if fresh:
            try:
                self._rig_cache_path(marks).write_text(json.dumps(cache), encoding="utf-8")
            except Exception:
                self._note_trace()

    def _rig_cache_path(self, marks):
        stamp = hashlib.md5(json.dumps(marks, sort_keys=True)
                            .encode("utf-8")).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / ("rgbmocap-cascadeur-rigrest-%s.json" % stamp)

    def _load_rig_cache(self, marks):
        try:
            cache = json.loads(self._rig_cache_path(marks).read_text(encoding="utf-8"))
        except Exception:
            return None
        if cache.get("marks") != marks:
            return None
        if not all(name in (cache.get("rot") or {}) for name in self.channels):
            return None
        return cache

    def _rig_entries(self, desired, world_position, root_name):
        """Writes that put every rig point where this frame's pose needs it.

        Each point rides rigidly with its joint, so its place is the joint's
        place plus its rest offset turned by the joint's orientation; the
        joints' places come down the hierarchy from the root, with the
        character's own bone lengths. The rig then derives its boxes and
        joints from points that already describe one rigid pose.
        """
        rest_world = self.rest_world or {}
        rest_pos = self.rest_world_positions or {}
        frames = {}
        for name in self._order():
            source = desired.get(name)
            if source is None:
                source = rest_world.get(name)
            if source is not None:
                frames[name] = np.asarray(source, dtype=np.float64)
        spots = {}
        for name in self._order():
            parent = self.channels[name].get("parent")
            if name == root_name and name in world_position:
                spots[name] = np.asarray(world_position[name], dtype=np.float64)
            elif (parent in spots and parent in frames
                    and rest_world.get(parent) is not None
                    and rest_pos.get(parent) is not None
                    and rest_pos.get(name) is not None):
                offset = (np.asarray(rest_pos[name], dtype=np.float64)
                          - np.asarray(rest_pos[parent], dtype=np.float64))
                spots[name] = spots[parent] + frames[parent] @ np.asarray(
                    rest_world[parent], dtype=np.float64).T @ offset
        out = []
        for name, items in self.rig["points"].items():
            if (name not in spots or name not in frames
                    or rest_world.get(name) is None or rest_pos.get(name) is None):
                continue
            carry = frames[name] @ np.asarray(rest_world[name], dtype=np.float64).T
            origin = np.asarray(rest_pos[name], dtype=np.float64)
            for item in items:
                if item.get("rest") is None:
                    continue
                target = spots[name] + carry @ (item["rest"] - origin)
                out.append((dict(rotation=None, position=item["position"],
                                 layer=item["layer"]), None, target))
        return out

    def _animation_keyed(self, data):
        """Whether this scene carries animation past an imported frame.

        Keys outlive their character: a scene keeps its animation length
        after the character it was written on is deleted, which is why a
        fresh import must land in a fresh scene - the transfer tool starts
        one - for this to read one frame on a pristine character and more
        than one once a take has been written.
        """
        try:
            return int(data.get_animation_size()) > 1
        except Exception:
            return False

    def _bone_marks(self, data):
        """The rig's bone geometry, which no amount of keying changes.

        A joint's local position is its bone offset and does not follow the
        pose - only the root's does, and the root is left out - so these
        numbers separate this character from every other without depending
        on which frame anyone is looking at.
        """
        marks = {}
        for name, channel in (self.channels or {}).items():
            if channel.get('parent') is None:
                continue
            try:
                value = data.get_data_value(channel['position'], 0)
                marks[name] = [round(float(value[r]), 4) for r in range(3)]
            except Exception:
                marks[name] = None
        return marks

    def _rest_cache_path(self, marks):
        stamp = hashlib.md5(json.dumps(marks, sort_keys=True)
                            .encode('utf-8')).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / ('rgbmocap-cascadeur-rest-%s.json'
                                              % stamp)

    def _capture_rest_cache(self, data):
        """Bank the pristine character's frame-0 locals for later rebinds.

        Called only on the fresh import, while frame 0 still is the rest.
        The bank is kept in memory for this Cascadeur session and in a temp
        file named by the rig's bone geometry, so a later session with the
        same character finds it again; a different character never can.
        """
        cache = {}
        for name, channel in (self.channels or {}).items():
            try:
                value = data.get_data_value(channel['rotation'], 0)
                rows = value.to_rotation_matrix()
                where = data.get_data_value(channel['position'], 0)
                cache[name] = {
                    'rotation': [[float(rows[r][c]) for c in range(3)]
                                 for r in range(3)],
                    'position': [float(where[0]), float(where[1]),
                                 float(where[2])],
                }
            except Exception:
                continue
        if len(cache) != len(self.channels or {}):
            self._note("the rest could not be banked: %d of %d joints read"
                       % (len(cache), len(self.channels or {})))
            return
        self.rest_cache = cache
        try:
            payload = json.dumps({'names': list(self.names or []),
                                  'rest': cache})
            self._rest_cache_path(self._bone_marks(data)).write_text(
                payload, encoding='utf-8')
        except Exception:
            self._note_trace()

    def _load_rest_cache(self, marks):
        """A banked rest from an earlier session, if this is the same rig."""
        try:
            payload = json.loads(self._rest_cache_path(marks)
                                 .read_text(encoding='utf-8'))
        except Exception:
            return None
        cache = payload.get('rest') or {}
        if set(payload.get('names') or []) != set(self.names or []):
            return None
        if all(name in cache for name in (self.names or [])):
            return cache
        return None

    def _apply_rest_cache(self, scene, cache):
        """Write a banked rest onto a scratch frame; return that frame.

        The rest has to sit ON the character to be read back, but frame 0
        belongs to the take: a previous run keyed it, and this run's first
        frame would land on a rest. The bank goes to the frame just past the
        animation - nothing keyed lives there - and every rest read happens
        there. (The old form wrote over frame 0 and ate the take's first
        pose; a later rebind ate it again.)
        """
        py_scene = pycsc.wrap(scene)
        data = scene.data_viewer()
        try:
            size = int(data.get_animation_size())
        except Exception:
            return None
        # The layers' sections bound what can be written, and they can be
        # shorter than the data viewer's animation size - a write past them
        # is refused, which used to fail the whole bind. Make room first.
        if not self._grow(scene, size + 2):
            return None
        scratch = size + 1

        def put_back(model, update, updater):
            editor = model.data_editor()
            touched = set()
            for name, channel in (self.channels or {}).items():
                entry = cache.get(name)
                if entry is None:
                    continue
                editor.set_data_value(
                    channel['rotation'], scratch,
                    csc.math.Rotation.from_rotation_matrix(entry['rotation']))
                editor.set_data_value(channel['position'], scratch,
                                      [float(v) for v in entry['position']])
                touched.add(channel['rotation'])
                touched.add(channel['position'])
                layer_id = channel.get('layer')
                if layer_id is not None and not layer_id.is_null():
                    model.layers_editor().set_fixed_interpolation_or_key_if_need(
                        layer_id, scratch, True)
            model.set_fixed_interpolation_if_need(touched, scratch)
            updater.run_update(touched, scratch)

        try:
            if py_scene.modify_update("RGB MoCap live: rest", put_back) is False:
                return None
        except Exception:
            self._note_trace()
            return None
        channels = self.channels or {}
        try:
            data.get_data_value(next(iter(channels.values()))['rotation'],
                                scratch)
        except Exception:
            # One past the last frame is refused on some documents; the last
            # real frame costs one take frame at the very end, no more.
            scratch = max(0, scratch - 1)
            try:
                if py_scene.modify_update("RGB MoCap live: rest",
                                          put_back) is False:
                    return None
            except Exception:
                self._note_trace()
                return None
        return scratch

    def _calibrate_root(self, scene, data):
        """Measure how this scene turns a root local position into a world one.

        An imported hierarchy can carry unit scale and axis rotations on the
        nodes between the root joint and the world - a Blender skeleton
        imported here lands under a node that scales by a hundred and turns
        - so the same local value does not mean the same displacement that it
        would in a plain scene. Three small probe writes at frame 0 read the
        world answer back and fit that affine map exactly; the probes are
        undone before anything is keyed, and every later root placement goes
        through the measured map instead of an assumption.
        """
        self.root_map = None
        if self.rig:
            return      # the root is placed through its rig point, in the world
        channels = self.channels or {}
        root_name = None
        for name, channel in channels.items():
            parent = channel.get('parent')
            if parent is None or parent not in channels:
                root_name = name
                break
        if root_name is None:
            return
        channel = channels[root_name]
        viewer = scene.model_viewer()
        behaviours = viewer.behaviour_viewer()
        try:
            transform = behaviours.get_behaviour_by_name(channel['object_id'],
                                                         "Transform")
            if transform.is_null():
                return
            local_id = channel['position']
            global_id = behaviours.get_behaviour_data(transform, "global_position")
            if local_id.is_null() or global_id.is_null():
                return
            at = 0
            mine = data.get_data_value(local_id, at)
            l0 = np.asarray([float(mine[r]) for r in range(3)], dtype=np.float64)
            world = data.get_data_value(global_id, at)
            g0 = np.asarray([float(world[r]) for r in range(3)], dtype=np.float64)
            py_scene = pycsc.wrap(scene)
            columns = []
            step = 1.0
            for axis in range(3):
                probe = l0.copy()
                probe[axis] += step

                def nudge(model, update, updater, value=probe):
                    model.data_editor().set_data_value(
                        local_id, at, [float(v) for v in value])
                    updater.run_update({local_id}, at)

                if py_scene.modify_update("RGB MoCap live: probe", nudge) is False:
                    return
                moved = data.get_data_value(global_id, at)
                columns.append(
                    (np.asarray([float(moved[r]) for r in range(3)],
                                dtype=np.float64) - g0) / step)

            def put_back(model, update, updater):
                model.data_editor().set_data_value(
                    local_id, at, [float(v) for v in l0])
                updater.run_update({local_id}, at)

            py_scene.modify_update("RGB MoCap live: probe restore", put_back)
            matrix = np.column_stack(columns)
            if not np.isfinite(matrix).all():
                return
            if abs(float(np.linalg.det(matrix))) < 1e-9:
                return
            self.root_map = (np.linalg.inv(matrix), g0 - matrix @ l0)
        except Exception:
            self.root_map = None
            self._note_trace()

    def _relative_turn(self, rest_vec, shown_vec, parent_name, desired):
        """The same turn, measured inside the parent's frame.

        A limb that swings far from its rest passes through the antipode of
        its rest direction, where a world swing has no defined axis and the
        roll flips (measured: 169 degrees in one frame at the left elbow while
        the direction moved 27). Inside the parent's frame the same turn is
        only the joint's own bend, which stays small and continuous.
        """
        if not parent_name or not desired:
            return None
        above = desired.get(parent_name)
        above_rest = (self.rest_world or {}).get(parent_name)
        if above is None or above_rest is None:
            return None
        above = np.asarray(above, dtype=np.float64)
        above_rest = np.asarray(above_rest, dtype=np.float64)
        local_rest = above_rest.T @ np.asarray(rest_vec, dtype=np.float64)
        local_shown = above.T @ np.asarray(shown_vec, dtype=np.float64)
        swing = _swing(local_rest, local_shown)
        if swing is None:
            return None
        return _multiply(_multiply(above, swing), above_rest.T)

    def _turn_from_positions(self, name, index, world_position, along_of,
                             across_of, desired=None):
        """The world turn that points this joint the streamed body does.

        Built from directions alone: where the streamed body carries on from
        this joint (its continuing child) and a second direction that fixes
        the roll, aligned onto the same two directions of this character at
        rest. No bone-frame convention on either side can twist it, because
        none takes part - the streamed joint positions and the character's
        own rest positions decide everything. None when the streamed body
        does not say, and the sender's own turn is used instead.
        """
        channel = (self.channels or {}).get(name) or {}
        rests = self.rest_world_positions or {}
        if name not in world_position or name not in rests \
                or rests[name] is None:
            return None
        here = np.asarray(world_position[name], dtype=np.float64)
        mine = np.asarray(rests[name], dtype=np.float64)

        def pair(other):
            if other is None or other not in world_position \
                    or other not in rests or rests[other] is None:
                return None, None
            shown = np.asarray(world_position[other], dtype=np.float64) - here
            rest = np.asarray(rests[other], dtype=np.float64) - mine
            if float(np.linalg.norm(shown)) < 1e-6 \
                    or float(np.linalg.norm(rest)) < 1e-6:
                return None, None
            return shown, rest

        shown_a, rest_a = pair(self.names[along_of[index]]
                               if along_of.get(index, -1) >= 0
                               and along_of[index] < len(self.names) else None)
        turn = None
        if shown_a is not None:
            shown_b, rest_b = pair(
                self.names[across_of[index]]
                if across_of.get(index, -1) >= 0
                and across_of[index] < len(self.names) else None)
            if shown_b is None:
                parent_name = channel.get('parent')
                if parent_name in world_position and parent_name in rests \
                        and rests[parent_name] is not None:
                    shown_b = here - np.asarray(world_position[parent_name],
                                                dtype=np.float64)
                    rest_b = mine - np.asarray(rests[parent_name],
                                               dtype=np.float64)
                    if float(np.linalg.norm(shown_b)) < 1e-6 \
                            or float(np.linalg.norm(rest_b)) < 1e-6:
                        shown_b = rest_b = None
            # Only a well separated pair may decide the roll. A straight limb
            # (the forearm with the hand, the thigh with the shin) has no
            # plane at rest, and a pair that close turns measurement noise
            # into a roll flip.
            if (shown_b is not None and _separated(rest_a, rest_b)
                    and _separated(shown_a, shown_b)):
                turn = _align(rest_a, rest_b, shown_a, shown_b)
                if turn is None:
                    turn = _plane(rest_a, rest_b, shown_a, shown_b)
            if turn is None:
                turn = self._relative_turn(rest_a, shown_a,
                                           channel.get('parent'), desired)
            if turn is None:
                turn = _swing(rest_a, shown_a)
            if name in ("LeftUpLeg", "LeftLeg"):
                self.debug[name] = {
                    "shown_a": [round(float(v), 3) for v in shown_a],
                    "rest_a": [round(float(v), 3) for v in rest_a],
                    "turn": [[round(float(v), 3) for v in row] for row in turn],
                }
            return turn
        # A leaf: no child to point at, so it swings the way it came in -
        # the direction it hangs from the body by, on both sides.
        parent_name = channel.get('parent')
        if parent_name in world_position and parent_name in rests \
                and rests[parent_name] is not None:
            shown_in = here - np.asarray(world_position[parent_name],
                                         dtype=np.float64)
            rest_in = mine - np.asarray(rests[parent_name], dtype=np.float64)
            if float(np.linalg.norm(shown_in)) > 1e-6 \
                    and float(np.linalg.norm(rest_in)) > 1e-6:
                turn = self._relative_turn(rest_in, shown_in, parent_name, desired)
                if turn is None:
                    turn = _swing(rest_in, shown_in)
        return turn

    def _root_local(self, point, channel):
        """The root's local position that lands it on this world point."""
        mapping = self.root_map
        if mapping is not None:
            matrix, shift = mapping
            local = matrix @ (np.asarray(point, dtype=np.float64) - shift)
            return [float(v) for v in local]
        fixed = channel.get('fixed_position')
        if fixed is not None:
            point = np.asarray(point, dtype=np.float64) - np.asarray(
                fixed, dtype=np.float64)
        return point

    def _order(self):
        """Joint names, parents before their children."""
        depth = {}

        def of(name):
            if name in depth:
                return depth[name]
            parent = (self.channels.get(name) or {}).get("parent")
            depth[name] = 0 if parent is None else of(parent) + 1
            return depth[name]

        return sorted(self.channels or {}, key=lambda name: (of(name), name))

    def _tick(self):
        # A Qt timer swallows whatever its callback raises, so anything that
        # goes wrong in here has to be kept or it is invisible.
        try:
            self._tick_inner()
        except Exception:
            self._note_trace()

    def _tick_inner(self):
        # Asked to stop from the HTTP thread. A QTimer belongs to the thread
        # that made it and only that thread may touch it, so the request is
        # left as a flag and honoured here.
        if self.stopping:
            self._shutdown_timer()
            return
        scene = _current_scene()
        if scene is None:
            return

        if self.wanted_frame is not None:
            frame, self.wanted_frame = self.wanted_frame, None
            try:
                scene.set_current_frame(int(frame))
                self.last_at = int(frame)
            except Exception:
                pass

        # What the scene is made of, refreshed here because here is the only
        # thread allowed to ask.
        if time.time() - self.names_at > 1.0:
            try:
                self._refresh_names(scene)
            except Exception:
                pass

        # Bind before the first frame arrives, not when it does: the sender
        # asks whether the character was recognised as soon as a take starts.
        if not self.channels and self.names and time.time() - self.bind_at > 1.0:
            self.bind_at = time.time()
            try:
                self._bind(scene)
            except Exception:
                self._note_trace()

        with self.lock:
            if not self.queue:
                return
            batch = list(self.queue)
            self.queue.clear()

        if not self.channels:
            self.names = list(batch[0].get("names") or [])
            if not self._bind(scene):
                # The bind refused (no usable rest, or no joint matched).
                # Keep the frames - they are written the moment a bind
                # succeeds - and try again on a later tick.
                self._note("bind refused; %d frame(s) kept in the queue"
                           % len(batch))
                with self.lock:
                    batch.extend(self.queue)
                    self.queue = batch[:QUEUE_LIMIT]
                return

        channels = self.channels
        order = self._order()
        prepared = []
        wanted_at = []
        for frame in batch:
            rotations = frame.get("rotations") or []
            positions = frame.get("positions") or []
            aims = frame.get("aims") or []
            aims2 = frame.get("aims2") or []
            refs = frame.get("refs") or []
            turns = frame.get("deltas") or []
            locals_in = frame.get("nats") or []
            # Only the root is placed. Every other joint is where its parent
            # and its rotation put it, which is right for a character of any
            # shape; writing this skeleton's joint positions instead drags a
            # character with different bone lengths onto this one's.
            root = frame.get("root")
            wanted_at.append(frame.get("at"))
            index_of = {name: index for index, name in enumerate(self.names)}
            world_rotation = {}
            world_position = {}
            for name, index in index_of.items():
                if index < len(rotations):
                    world_rotation[name] = rotations[index]
                if index < len(positions):
                    world_position[name] = [float(v) * SCALE for v in positions[index]]

            aim_of = {}
            second_of = {}
            refs_of = {}
            for row, name in enumerate(self.names):
                if row < len(aims):
                    aim_of[name] = aims[row]
                if row < len(aims2):
                    second_of[name] = aims2[row]
                if row < len(refs):
                    refs_of[name] = refs[row]
            # Diagnostic switch: a frame can ask for the sender's own rotation
            # deltas instead of the position-aim path, so the two paths can be
            # measured on identical input.
            aims_enabled = bool(frame.get('use_aims', True))
            along_of = {row: value for row, value in enumerate(frame.get('along') or [])
                        if isinstance(value, int)}
            across_of = {row: value for row, value in enumerate(frame.get('across') or [])
                         if isinstance(value, int)}
            entries = []
            desired = {}
            for name in order:
                channel = channels[name]
                index = index_of.get(name)
                if index is None or name not in world_rotation:
                    continue
                rests = self.rests or {}
                rest = rests.get(name)
                # The whole transfer: how far the joint has TURNED from its
                # rest, measured in that joint's own rest frame, composed onto
                # this character's own rest. The identity turn leaves the
                # joint exactly where this character rests, so a different
                # rest pose is never twisted - it is the pose the character
                # was built with, turned by the same amount.
                turn = None
                finger_desired = None
                # Every joint - fingers included - aims at the streamed body:
                # the position path measured the fingers at 1.5-2.5 degrees
                # where the rotation-delta path flipped them over, because
                # the export skeleton's finger rest frames do not match the
                # geometric bone directions. The last phalanx still rides
                # its middle segment rigidly: the app's fingertip positions
                # wobble more than the bone is long.
                finger_segment = ('Hand' in name and any(
                    tag in name for tag in
                    ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')))
                if finger_segment and name.endswith('3'):
                    above = desired.get(channel.get('parent'))
                    if above is not None:
                        finger_desired = above
                turn = (self._turn_from_positions(name, index, world_position,
                                                  along_of, across_of, desired)
                        if aims_enabled else None)
                from_positions = turn is not None
                if turn is not None:
                    self.path_counts["positions"] += 1
                if turn is None and index < len(turns):
                    turn = turns[index]
                    self.path_counts["deltas"] += 1
                self.debug_paths[name] = "positions" if from_positions else "deltas"
                rest_here = (self.rest_world or {}).get(name)
                if finger_desired is not None:
                    # The corrected sender rotation IS the world orientation:
                    # the per-joint correction already carries this
                    # character's rest.
                    desired[name] = finger_desired
                elif turn is not None and rest_here is not None:
                    if from_positions:
                        # _align / _swing / _plane all return a WORLD rotation:
                        # T carries the character's rest axes onto the streamed
                        # ones. A world rotation multiplies on the LEFT, so the
                        # bone's world orientation becomes T @ rest. Composing
                        # it on the right (rest @ T) is what folded the body.
                        desired[name] = _multiply(turn, rest_here)
                    else:
                        # The sender's delta is expressed in the joint's own
                        # REST frame (rest^T @ world), so it composes on the
                        # right: rest @ delta is the posed world.
                        desired[name] = _multiply(rest_here, turn)
                if desired.get(name) is not None:
                    if name in ("Hips", "LeftUpLeg", "LeftLeg"):
                        self.debug[name + ":desired"] = [
                            [round(float(v), 3) for v in row]
                            for row in desired[name]]
                    parent_name = channel.get('parent')
                    above = desired.get(parent_name)
                    if above is None:
                        above = channel.get('fixed_rotation')
                    if above is not None:
                        local_rotation = _multiply(_transpose(above),
                                                   desired[name])
                    else:
                        local_rotation = desired[name]
                    if self.rig and name in self.rig['points']:
                        continue    # written through its rig points below
                    if self.delta_mode:
                        # The channel is a delta from the bind: the bone's own
                        # rest rotation sits between the parent world and the
                        # channel, so the absolute local goes through
                        # bone_rest^-1 - the factor the first live run missed,
                        # which twisted the head and the spine.
                        inv = self.bone_rest_inv.get(name)
                        if inv is not None:
                            local_rotation = _multiply(inv, local_rotation)
                    place = None
                    if index == root and index < len(positions):
                        point = np.asarray(world_position[name], dtype=np.float64)
                        place = np.asarray(self._root_local(point, channel),
                                           dtype=np.float64)
                    entries.append((channel, local_rotation, place))
                    continue
                # The app's own Mixamo retargeting, written straight in: the
                # turn from rest in this joint's own local frame, corrected by
                # the constant difference between the two rests. Nothing is
                # guessed from positions, and bone lengths are irrelevant.
                given = locals_in[index] if index < len(locals_in) else None
                rest_local = (self.rest_local or {}).get(name)
                if given is not None and rest_local is not None:
                    # Same Mixamo hierarchy on both sides, so the sender's node
                    # rotation is the local rotation to write. Nothing about
                    # bone lengths or poses is inferred here.
                    rotation_local = given
                    parent_name = channel.get('parent')
                    above = desired.get(parent_name)
                    if above is None:
                        above = channel.get('fixed_rotation')
                    desired[name] = (_multiply(above, rotation_local)
                                     if above is not None else rotation_local)
                    if self.rig and name in self.rig['points']:
                        continue
                    place = None
                    if index == root and index < len(positions):
                        point = np.asarray(world_position[name], dtype=np.float64)
                        point = np.asarray(self._root_local(point, channel),
                                           dtype=np.float64)
                        if above is not None:
                            point = np.asarray(_transpose(above)) @ point
                        place = np.asarray(point, dtype=np.float64)
                    entries.append((channel, rotation_local, place))
                    continue
                if rest is None:
                    continue
                # Where this joint must point, taken from the take's own joint
                # positions: the swing that carries the rest bone direction
                # onto the streamed one, applied to the rest orientation. A
                # turn read out of the export skeleton's rotation table lives
                # in that table's basis, which is what knotted the character.
                swing = None
                places = self.rest_positions or {}
                shown = aim_of.get(name)
                second = second_of.get(name)
                pair = refs_of.get(name) or []
                target = None
                # Two references that both hang directly off this joint decide
                # its whole orientation. The offsets are the character's OWN,
                # read from its bones, so this needs no bind matrix and no
                # guess about how it rests: a Mixamo mesh binds in a different
                # pose from the skeleton's rest, which is what left the legs and
                # fingers swinging around their parents.
                direct = (len(pair) == 3 and pair[1] == index
                          and 0 <= pair[0] < len(self.names)
                          and 0 <= pair[2] < len(self.names))
                if (direct and shown is not None and second is not None
                        and any(abs(float(v)) > 1e-6 for v in shown)
                        and any(abs(float(v)) > 1e-6 for v in second)):
                    one = (channels.get(self.names[pair[0]]) or {}).get('offset')
                    two = (channels.get(self.names[pair[2]]) or {}).get('offset')
                    if one is not None and two is not None:
                        target = _plane(one, two, shown, second)
                        if target is not None:
                            self.direct_names.add(name)
                if target is None and shown is not None:
                    places = self.rest_positions or {}
                    child_row = pair[0] if len(pair) > 0 else -1
                    child_name = (self.names[child_row]
                                  if 0 <= child_row < len(self.names) else None)
                    if child_name in places and name in places:
                        here = np.asarray(places[name], dtype=np.float64)
                        rest_along = np.asarray(places[child_name], dtype=np.float64) - here
                        # The plane the limb bends in: the bone and the bone it
                        # came from. A foot or a head has only one child to
                        # follow, and the roll around that one direction is
                        # exactly what decides whether the sole faces down or
                        # the head twists. Both directions come from joints.
                        parent_name = channel.get('parent')
                        rest_across = None
                        across = None
                        if parent_name in places and parent_name in world_position:
                            rest_across = (np.asarray(places[parent_name], dtype=np.float64) - here)
                            across = (np.asarray(world_position[parent_name], dtype=np.float64)
                                      - np.asarray(world_position[name], dtype=np.float64))
                        delta = (_plane(rest_along, rest_across, shown, across)
                                 if rest_across is not None else None)
                        if delta is None:
                            delta = _swing(rest_along, shown)
                        if delta is not None:
                            target = _multiply(delta, rest)
                if target is None:
                    # A joint with nothing below it - the head, the toes - has no
                    # child direction to follow. What it does have is the
                    # direction it came in on, so that is what gets aimed: the
                    # head then sits on its neck the way the take has it. The
                    # turn path that used to run here is in the sender's export
                    # basis and flipped these joints right over.
                    parent_name = channel.get('parent')
                    rests_here = self.rest_positions or {}
                    if (parent_name in rests_here and name in rests_here
                            and parent_name in world_position and name in world_position):
                        here = np.asarray(world_position[name], dtype=np.float64)
                        incoming = here - np.asarray(world_position[parent_name], dtype=np.float64)
                        rest_in = (np.asarray(rests_here[name], dtype=np.float64)
                                   - np.asarray(rests_here[parent_name], dtype=np.float64))
                        swing = _swing(rest_in, incoming)
                        if swing is not None:
                            target = _multiply(swing, rest)
                    if target is None:
                        target = (desired.get(parent_name) or channel.get('fixed_rotation')
                                  or rest)
                desired[name] = target
                parent = channel.get('parent')
                parent_rotation = desired.get(parent)
                if parent_rotation is not None:
                    parent_position = world_position.get(parent)
                else:
                    parent_rotation = channel.get('fixed_rotation')
                    parent_position = channel.get('fixed_position')
                if parent_rotation is None:
                    local_rotation = target
                else:
                    local_rotation = _multiply(_transpose(parent_rotation), target)
                place = None
                if index == root and index < len(positions):
                    point = np.asarray(world_position[name], dtype=np.float64)
                    if parent_position is not None:
                        point = point - np.asarray(parent_position, dtype=np.float64)
                    if parent_rotation is not None:
                        point = np.asarray(_transpose(parent_rotation)) @ point
                    place = point.astype(np.float64)
                if not (self.rig and name in self.rig['points']):
                    entries.append((channel, local_rotation, place))
            if self.rig:
                root_name = (self.names[root] if isinstance(root, int)
                             and 0 <= root < len(self.names) else None)
                entries.extend(self._rig_entries(desired, world_position, root_name))
            prepared.append(entries)

        # Nothing may be written past the end of the animation, and a freshly
        # loaded character is one frame long: writing there fails with
        # "cannot set value on frame 1, last frame in animation is 0".
        named = [a for a in wanted_at if a is not None]
        highest = max(named) + 1 if named else 0
        needed = max(self.frame + len(prepared), highest)
        if needed > self.length:
            block = max(needed, self.length + GROW_BLOCK)
            if self._grow(scene, block):
                self.length = block

        started = time.perf_counter()
        try:
            self._write(scene, prepared, wanted_at)
        except Exception:
            self._note(traceback.format_exc().strip().splitlines()[-1])
            return
        self.last_write_ms = (time.perf_counter() - started) * 1000.0
        if self.follow:
            try:
                scene.set_current_frame(int(self.last_at))
            except Exception:
                self.follow = False

    def _grow(self, scene, frames):
        """Make room for `frames` frames on every layer."""
        layers = scene.layers_viewer()
        py_scene = pycsc.wrap(scene)

        def grow(model, update, updater):
            editor = model.layers_editor()
            for layer_id in layers.all_layer_ids():
                if not layer_id.is_null():
                    editor.set_section(_fixed_section(), frames, layer_id)
            model.fit_animation_size_by_layers()
            updater.generate_update()

        ok = py_scene.modify_update("RGB MoCap live: make room", grow)
        if ok is False:
            self._note("could not make room for frame %d" % frames)
        return ok is not False

    def _write(self, scene, prepared, wanted_at=None):
        py_scene = pycsc.wrap(scene)
        start = self.frame
        wanted_at = wanted_at or [None] * len(prepared)

        def write(model, update, updater):
            editor = model.data_editor()
            layer_editor = model.layers_editor()
            for offset, entries in enumerate(prepared):
                named = wanted_at[offset] if offset < len(wanted_at) else None
                at = start + offset if named is None else int(named)
                touched = set()
                for channel, rotation, place in entries:
                    if rotation is not None:
                        editor.set_data_value(channel['rotation'], at,
                                              csc.math.Rotation.from_rotation_matrix(rotation))
                        touched.add(channel['rotation'])
                    if place is not None and not channel['position'].is_null():
                        editor.set_data_value(channel['position'], at, place)
                        touched.add(channel['position'])
                    layer_id = channel.get('layer')
                    if layer_id is not None and not layer_id.is_null():
                        layer_editor.set_fixed_interpolation_or_key_if_need(
                            layer_id, at, True)
                model.set_fixed_interpolation_if_need(touched, at)
                # This is what commits the values, and it has to happen per
                # frame. A scene in Rig Mode can raise "Cycle in data
                # dependency graph" from inside it - its spline IK helpers
                # read a joint that was just written - and that used to escape
                # and have the whole batch refused. The values written before
                # the cycle are still in place, so the frame keeps them and
                # says which joint complained.
                try:
                    updater.run_update(touched, at)
                except Exception as error:
                    self._note('frame %d: %s' % (at, str(error).splitlines()[0] if str(error) else error))

        ok = py_scene.modify_update("RGB MoCap live", write)
        if ok is False:
            self._note("the scene refused the write")
            return
        if not [a for a in (wanted_at or []) if a is not None]:
            self.frame += len(prepared)
        self.written += len(prepared)
        self.last_at = (int(wanted_at[-1]) if wanted_at and
                        wanted_at[-1] is not None else self.frame - 1)

    def _note_trace(self):
        """The last line is not enough to find anything. Keep the frame too."""
        lines = traceback.format_exc().strip().splitlines()
        where = [l.strip() for l in lines if l.strip().startswith("File ")]
        self._note("%s | %s" % (lines[-1] if lines else "?",
                                where[-1] if where else "?"))

    def _note(self, text):
        # Informational only: which channel semantics were probed, why a bind
        # was skipped. These used to land in `errors`, where every caller read
        # them as a failed transfer (the full-take report said "receiver missed
        # frames" after 706 of 706 frames had been written).
        if text and (not self.notes or self.notes[-1] != text):
            self.notes.append(text)
            del self.notes[:-5]

    # -- lifecycle ------------------------------------------------------
    def start(self):
        self.server = ThreadingHTTPServer((HOST, PORT), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="RGB MoCap Cascadeur link")
        self.thread.daemon = True
        self.thread.start()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(INTERVAL_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _shutdown_timer(self):
        """Main thread only. Called from the tick itself."""
        if self.timer is not None:
            self.timer.stop()
            self.timer.timeout.disconnect()
            self.timer = None

    def stop(self):
        """Ask to stop. Safe from any thread.

        The timer is not touched here - it takes itself down on its next tick.
        The HTTP server is a plain socket server and has no such affinity, but
        `shutdown()` blocks until the serve loop notices, so it may not be
        called from one of its own request threads.
        """
        self.stopping = True
        server, self.server = self.server, None
        if server is not None:
            threading.Thread(target=_close_server, args=(server,),
                             daemon=True).start()

    def rest_rows(self):
        return {name: [[float(value) for value in row] for row in rows]
                for name, rows in (self.rests or {}).items()}

    def local_rest_rows(self):
        """The rest rotations the writes actually compose onto: each joint's
        rest relative to its parent, read from the character itself."""
        return {name: [[float(value) for value in row] for row in rows]
                for name, rows in (self.rest_local or {}).items()}

    def joint_names(self):
        """The scene's object names, as the timer last saw them.

        NEVER read from the scene here: this answers the HTTP thread, and
        touching the scene off the main thread is what takes Cascadeur down.
        """
        return list(self.names_seen)

    def _refresh_names(self, scene):
        """Main thread only, from the tick."""
        viewer = scene.model_viewer()
        self.names_seen = sorted(viewer.get_object_name(o)
                                 for o in viewer.get_objects())
        self.names_at = time.time()

    def report(self):
        with self.lock:
            depth = len(self.queue)
        return {"ok": True, "version": VERSION, "queued": depth,
                "notes": list(self.notes),
                "paths_by_joint": dict(self.debug_paths),
                "received": self.received, "written": self.written,
                "dropped": self.dropped, "frame": self.frame,
                "bound": len(self.channels or {}),
                "rests": len(self.rests or {}),
                "direct": len(self.direct_names),
                "direct_names": sorted(self.direct_names)[:8],
                "rig": bool(self.rig),
                "rig_points": sum(len(v) for v in (self.rig or {}).get("points", {}).values()),
                "roots": sum(1 for c in (self.channels or {}).values()
                             if c.get('parent') is None),
                "write_ms": round(self.last_write_ms, 2),
                "paths": dict(self.path_counts),
                "debug": self.debug,
                "follow": self.follow, "at": self.last_at,
                "errors": list(self.errors),
                "uptime": round(time.time() - self.started, 1)}


class _Handler(BaseHTTPRequestHandler):

    # One request per connection measured out at about 190 ms inside Cascadeur
    # - the connection and its thread cost far more than the values in it, and
    # a live link at five frames a second is not a live link. Keep-alive plus
    # Nagle off brings the same request down to a few milliseconds.
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, *args):
        pass

    def _reply(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/status", "/health"):
            self._reply(_state.report() if _state else {"ok": False})
        elif path == "/joints":
            # What the character over there is actually made of, asked now.
            if _state is None:
                self._reply({"ok": False, "error": "not running"}, 404)
            else:
                self._reply({"ok": True, "names": _state.joint_names()})
        elif path == "/rests":
            # The orientations every turn is composed onto. Read off `rests`,
            # which the timer fills once and never touches again: no scene call
            # happens here, and that is the point. `locals` is the parent-
            # relative form the writes actually use.
            if _state is None:
                self._reply({"ok": False, "error": "not running"}, 404)
            else:
                self._reply({"ok": True, "rests": _state.rest_rows(),
                             "locals": _state.local_rest_rows()})
        else:
            self._reply({"ok": False, "error": "unknown path"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as error:
            self._reply({"ok": False, "error": str(error)}, 400)
            return
        path = self.path.rstrip("/")
        # /stop sets _state to None, and a request that raced it used to take
        # the handler down with AttributeError and answer nothing.
        if path == "/frames":
            if _state is None:
                self._reply({"ok": False, "error": "not running"}, 409)
            else:
                self._reply(_state.offer(payload))
        elif path == "/at":
            # Just move the playhead. During playback this is all that is
            # wanted: the take is already over there, and rewriting every
            # joint of it thirty times a second is what took Cascadeur down.
            state = _state
            if state is None:
                self._reply({"ok": False, "error": "not running"}, 409)
            else:
                state.wanted_frame = int(payload.get("frame") or 0)
                self._reply({"ok": True, "frame": state.wanted_frame})
        elif path == "/reset":
            if _state is None:
                self._reply({"ok": False, "error": "not running"}, 409)
            else:
                _reset(payload)
                self._reply(_state.report())
        elif path == "/stop":
            self._reply({"ok": True})
            stop()
        else:
            self._reply({"ok": False, "error": "unknown path"}, 404)


def _reset(payload):
    """Start a new take: rebind to the character and rewind the frame."""
    with _state.lock:
        _state.queue.clear()
    _state.channels = None
    _state.rests = None
    _state.names = list(payload.get("names") or []) or _state.names
    _state.frame = int(payload.get("frame_start") or 0)
    _state.length = 0
    _state.follow = bool(payload.get("follow"))
    _state.written = 0
    _state.received = 0
    _state.dropped = 0
    _state.bind_at = 0.0
    _state.root_map = None
    _state.app_rest_dirs = list(payload.get("rest_dirs") or [])
    _state.finger_corr = {}
    del _state.errors[:]


def start():
    global _state
    if _state is not None:
        return _state.report()
    _state = _Link()
    _state.start()
    return _state.report()


def stop():
    global _state
    if _state is None:
        return {"ok": True, "running": False}
    _state.stop()
    _state = None
    return {"ok": True, "running": False}


def status():
    return _state.report() if _state is not None else {"ok": False,
                                                       "running": False}


if __name__ == "__main__":
    print(start())
