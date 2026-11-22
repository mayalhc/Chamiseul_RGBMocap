# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: MIT (see LICENSE in the repository root)
"""Live link: send the base body first, then stream a take into 3ds Max.

3ds Max has no script server to post a receiver into, so - unlike Cascadeur -
the receiver is a plug-in that runs inside Max. Install it once
(the RGB MoCap 3ds Max plug-in, distributed on its own) and Max starts it every time it opens; the
package in ``maxplugin/`` also adds an RGB MoCap menu with status, start and
stop. Running ``mocap/max_receiver.py`` from Scripting > Run Script... still
works for a single session. Either way it listens on 127.0.0.1:9570 until this
side stops it.

The order is the same as the Blender and Cascadeur links, and it is the order
the geometry needs:

1. base   - the prepared T-pose base body is imported into an empty Max scene
            and checked to be standing in a T-pose (``POST /base``)
2. bind   - the take's bones are matched to the scene's nodes by name, and the
            rest world rotation of every matched node is read (``POST /bind``)
3. frames - per-frame world turns away from that rest, plus the root position
            (``POST /frames``)

A pose is sent as how far each joint has TURNED from the app's rest, as a WORLD
turn - never as an absolute orientation and never as a turn about the bone's own
axes, which 3ds Max picks for itself when it imports a skeleton. The receiver
lays that turn on the rest the character in Max actually has (after the
corrections its own rig needs: which way it faces, and how its arms and hands
rest), so a character with different bone lengths, roll or stance - a Biped
among them - moves the way the take moves instead of being forced onto this
skeleton's axes. Only the root is placed, and the feet are stood where the
take's feet are; sending every joint's position would stretch a character that
is not this body's shape.

All coordinates arrive in Max's own space: Z up, centimetres.
"""
from __future__ import annotations

import json
import re
import os
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
import threading
import time
import uuid

import numpy as np

HOST = '127.0.0.1'
PORT = 9570
RECEIVER = 'max_receiver.py'

# Frames waiting to be sent, and how many go in one request. The receiver
# applies the newest frame of a batch, so batching is what keeps a live take
# complete instead of dropping every second pose.
PUBLISH_LIMIT = 240
PUBLISH_BATCH = 6

DEFAULT_FACING = 0.0

# The prepared T-pose body the Cascadeur and Unreal links already ship: the
# same rest, imported the same way, so all three links drive one body.
CHARACTER_FILE = Path('assets') / 'cascadeur' / 'base_body.fbx'

# One id for this running app: every person's stream carries it, so the receiver
# can tell this app from a second one sending at the same time.
SENDER = uuid.uuid4().hex

# Both ends are on this machine, so a proxy must never be consulted.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# App world is Y-up; Max is Z-up. A @ (x, y, z) = (x, -z, y).
A = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])


def base_character_path(root):
    """The prepared T-pose base body that ships with an installation."""
    return Path(root) / CHARACTER_FILE


def receiver_source():
    path = Path(__file__).with_name(RECEIVER)
    if not path.is_file():
        raise RuntimeError('This build carries no %s.' % RECEIVER)
    return path.read_text(encoding='utf-8')


def receiver_version():
    """The VERSION of the receiver this build ships, or '' when it ships none.

    The 3ds Max plug-in (which holds the receiver) is distributed on its own, so
    a build without the file simply accepts whatever version answers.
    """
    try:
        source = receiver_source()
    except RuntimeError:
        return ''
    match = re.search(r'^VERSION\s*=\s*[\'"]([^\'"]+)', source, re.M)
    return match.group(1) if match else ''


def receiver_path():
    """Where the receiver lives, for the message that tells Max what to run."""
    return Path(__file__).with_name(RECEIVER).resolve()


def facing_angle():
    try:
        return float(os.environ.get('RGB_MOCAP_MAX_FACING') or DEFAULT_FACING)
    except ValueError:
        return DEFAULT_FACING


def _url(port, path):
    return 'http://%s:%d%s' % (HOST, port, path)


def _post(url, payload, timeout=8.0):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read() or b'{}')


def _get(url, timeout=1.5):
    with _OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read() or b'{}')


def bake(payloads, fps, clear=True, port=PORT, batch=20, progress=None):
    """Key a whole take in 3ds Max: open, send the frames in batches, close.

    ``payloads`` are bake_payload() results (any number of people, any order).
    Unlike the live stream nothing is dropped: each batch is answered before
    the next goes. Returns (ok, message).
    """
    channel = _Channel(port=port)
    try:
        reply = channel.post('/bake', {'op': 'begin', 'fps': float(fps), 'clear': bool(clear)},
                             timeout=60.0)
        if not reply.get('ok'):
            return False, str(reply.get('error') or '3ds Max refused the bake')
        for start in range(0, len(payloads), batch):
            reply = channel.post('/bake', {'op': 'frames', 'frames': payloads[start:start + batch]},
                                 timeout=120.0)
            if not reply.get('ok'):
                channel.post('/bake', {'op': 'end'}, timeout=60.0)
                return False, str(reply.get('error') or '3ds Max stopped the bake')
            if progress is not None:
                progress(min(len(payloads), start + batch), len(payloads))
        reply = channel.post('/bake', {'op': 'end'}, timeout=60.0)
    except (urllib.error.URLError, OSError, ValueError) as error:
        return False, '3ds Max did not answer the bake (%s)' % error
    finally:
        channel.close()
    if not reply.get('ok'):
        return False, str(reply.get('error') or '3ds Max could not finish the bake')
    return True, 'Baked %s frames (%s-%s) as keyframes.' % (
        reply.get('frames'), reply.get('first'), reply.get('last'))


def _yaw_z(degrees):
    """Rotation about the vertical, for a Z-up world."""
    angle = np.radians(float(degrees))
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class _Channel:
    """One kept-alive connection for the frame stream.

    A fresh connection per request costs milliseconds that a live frame does
    not have, so the stream holds one open and reconnects only when it breaks.
    Status and the handshakes use plain one-shot requests: they are asked once.
    """

    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = int(port)
        self._connection = None

    def close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def post(self, path, payload, timeout=4.0):
        import http.client
        import socket
        body = json.dumps(payload).encode('utf-8')
        if self._connection is None:
            self._connection = http.client.HTTPConnection(self.host, self.port,
                                                          timeout=timeout)
            self._connection.connect()
            try:
                self._connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except (OSError, AttributeError):
                pass
        try:
            self._connection.request('POST', path, body=body,
                                    headers={'Content-Type': 'application/json',
                                             'Content-Length': str(len(body))})
            response = self._connection.getresponse()
            data = response.read()
        except Exception:
            self.close()
            raise
        return json.loads(data or b'{}')


class MaxLink:
    """One live stream into the open 3ds Max scene.

    Publishing never blocks the caller: the newest frames are queued and a
    worker thread hands them over, so a scrub cannot stall playback.
    """

    def __init__(self, port=PORT):
        self.port = int(port)
        self.body = None
        self.nodes = []
        self.body_index = []
        self.root = 0
        self.facing = facing_angle()
        self.messages = []
        self.sent = 0
        self.failures = 0
        self.error = ''
        self.channel = _Channel(port=self.port)
        self.sender = SENDER
        self.person = 1
        self.persons = 1
        # How this person sits in the shared scene (a rigid 4x4 in the app's world),
        # or None: the viewer stands every person on one floor with it, so the
        # feet are only on the ground once it has been applied.
        self.transform = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._pending = deque()
        self._pending_set = set()
        # The 3ds Max frame a body frame is keyed at while Max records or bakes:
        # the app's timeline frame, which is not the body's own index when a
        # person's take starts later than the scene does.
        self._keys = {}
        self.skipped = 0
        self._lock = threading.Lock()

    # -- connecting ------------------------------------------------------
    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self, timeout=1.5):
        """What the receiver inside 3ds Max says about itself, or None."""
        try:
            return _get(_url(self.port, '/status'), timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def connect(self):
        """Check that the receiver is running inside 3ds Max.

        There is nothing to inject here: Max has no listening script server.
        The receiver is started in Max once and answers until it is stopped,
        so connecting is a question, not a launch.
        """
        found = self.status()
        if found and found.get('ok'):
            running = str(found.get('version') or '?')
            mine = receiver_version()
            if mine and running != mine:
                return False, ('3ds Max answers on port %d with live link v%s from '
                               'another build. Stop it there and run this one (v%s): %s'
                               % (self.port, running, mine, receiver_path()))
            return True, '3ds Max link already running (v%s).' % running
        return False, ('3ds Max is not answering on %s:%d. Open 3ds Max - the RGB MoCap '
                       'plug-in starts the receiver when Max opens - or install it '
                       'once (the RGB MoCap 3ds Max plug-in is distributed separately). To start it by hand in '
                       'Max: Scripting > Run Script... > %s, then connect again.'
                       % (HOST, self.port, receiver_path()))

    def disconnect(self, wait=6.0):
        """Stop the receiver and wait until the port is really free."""
        self.stop()
        try:
            _post(_url(self.port, '/stop'), {}, timeout=4.0)
        except (urllib.error.URLError, OSError, ValueError):
            return False
        began = time.time()
        while time.time() - began < wait:
            if self.status() is None:
                return True
            time.sleep(0.3)
        return False

    # -- the base body ---------------------------------------------------
    def send_base(self, fbx, new_scene=False, names=(), timeout=180.0, count=1):
        """Import the prepared T-pose base body into 3ds Max, then check it.

        The file is on this machine already, so only its path travels. The
        import goes into the scene that is open: `new_scene` would wipe it, so
        it is off unless the caller asks for it. It has
        to be a T-pose or an A-pose: every pose this link sends is composed
        onto the character's rest, and a character that rests in a take pose is
        bent from its first frame.

        The checks are unit-free on purpose. Max's system units are whatever
        the scene says they are, so an import in inches or in centimetres is
        judged by shape (arms level, feet under the hips), not by an absolute
        height that would only restate the unit setting.
        """
        fbx = str(Path(fbx))
        try:
            found = _post(_url(self.port, '/base'),
                          {'file': fbx, 'new_scene': bool(new_scene),
                           'names': list(names), 'count': int(count)},
                          timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, ('3ds Max did not answer the base character import (%s). '
                           'Start the receiver in Max first.' % error), {}
        if not found.get('ok'):
            return False, str(found.get('error') or 'Max refused the import'), found
        joints = {name: np.asarray(value, dtype=np.float64)
                  for name, value in (found.get('joints') or {}).items()}
        checks = {}

        def elevation(top, base):
            if top is None or base is None:
                return None
            vector = top - base
            length = float(np.linalg.norm(vector))
            if length < 1e-6:
                return None
            return round(float(np.degrees(np.arcsin(max(-1.0, min(1.0, vector[2] / length))))), 1)

        hips, head = joints.get('Hips'), joints.get('Head')
        if hips is not None and head is not None:
            checks['head_above_hips'] = round(float(head[2] - hips[2]), 1)
            checks['hips_height'] = round(float(hips[2]), 1)
        checks['left_arm_elevation_deg'] = elevation(joints.get('LeftArm'),
                                                    joints.get('LeftShoulder'))
        checks['right_arm_elevation_deg'] = elevation(joints.get('RightArm'),
                                                     joints.get('RightShoulder'))
        feet = []
        for side in ('Left', 'Right'):
            foot = joints.get(side + 'Foot')
            if foot is not None:
                feet.append(foot)
                checks[side.lower() + '_foot'] = round(float(foot[2]), 1)
        height = checks.get('head_above_hips')
        arms = (checks.get('left_arm_elevation_deg'), checks.get('right_arm_elevation_deg'))
        ok = bool(int(found.get('count') or 0) >= 20
                  and height is not None and height > 0
                  and all(a is not None and abs(a) <= 35.0 for a in arms))
        if hips is not None and head is not None and len(feet) == 2:
            # The legs are most of the body: the hips sit above the feet by
            # roughly the torso's height. A unit mix-up, a collapsed hierarchy
            # or a posed import all break this ratio.
            torso = float(head[2] - hips[2])
            legs = float(hips[2] - max(foot[2] for foot in feet))
            ratio = legs / torso if abs(torso) > 1e-9 else 0.0
            checks['legs_over_torso'] = round(ratio, 2)
            ok = ok and 0.6 <= ratio <= 2.2
            checks['foot_level_difference'] = round(
                float(abs(feet[0][2] - feet[1][2]) / torso), 3) if abs(torso) > 1e-9 else None
            ok = ok and (checks['foot_level_difference'] is None
                         or checks['foot_level_difference'] <= 0.12)
        if not ok:
            return False, ('the imported body is not standing in a T-pose or A-pose '
                           '(head %s over hips, arms %s deg, feet %s)'
                           % (height, arms, checks.get('left_foot'))), checks
        return True, ('Base character in 3ds Max: %s objects, head +%s over hips, '
                      'arms level.' % (found.get('count'), height)), checks

    # -- binding ---------------------------------------------------------
    @staticmethod
    def _match(body_names, nodes):
        """Pairs of (body index, Max node) that name the same joint.

        Exact names come first, then the name after the last colon (a Mixamo
        character keeps its `mixamorig:` prefix in Max), then the same
        comparison with spaces and case ignored, which is how a Biped names
        its limbs (`Bip01 L Thigh`).
        """
        available = list(nodes)
        exact = {name: name for name in available}
        by_suffix = {}
        by_flat = {}
        for name in available:
            by_suffix.setdefault(name.rsplit(':', 1)[-1].lower(), name)
            by_flat.setdefault(''.join(char for char in name.split(':')[-1].lower()
                                       if char.isalnum()), name)
        pairs = []
        for index, name in enumerate(body_names):
            target = exact.get(name)
            if target is None:
                target = by_suffix.get(name.rsplit(':', 1)[-1].lower())
            if target is None:
                flat = ''.join(char for char in name.rsplit(':', 1)[-1].lower()
                               if char.isalnum())
                target = by_flat.get(flat)
            if target is not None:
                pairs.append((index, target))
        return pairs

    def _rest_max_cm(self, rows, body):
        """The take's rest joints in Max's axes and centimetres, turned like the frames."""
        rest = np.asarray(body.rest, dtype=np.float64)[rows] * 100.0 @ A.T
        if self.facing:
            rest = rest @ _yaw_z(self.facing).T
        return rest

    def start(self, body, frame=0, publish_first=True):
        """Bind the take to the scene's nodes and start streaming it."""
        found = self.status()
        if not found:
            return False, '3ds Max is not linked. Connect the link first.'
        joints = list(found.get('nodes') or [])
        if not joints:
            return False, ('3ds Max has no base character in the scene. Run '
                           'File > Send Base Character to 3ds Max first.')
        pairs = self._match(list(body.names), joints)
        if not pairs:
            # Several bodies in one scene carry a number in their names ("Hips P2"),
            # which no name here matches. The receiver finds whole characters by
            # itself, so the take's own names go over and it does the matching.
            pairs = [(index, str(name)) for index, name in enumerate(body.names)]
        names = [name for _index, name in pairs]
        try:
            reply = _post(_url(self.port, '/bind'),
                          {'protocol': 2, 'sender': self.sender,
                           'person': self.person, 'persons': self.persons,
                           'rows': [index for index, _name in pairs],
                           'names': names,
                           'take_names': [str(body.names[index]) for index, _name in pairs],
                           'parents': [int(body.parents[index]) for index, _name in pairs],
                           'rest': [[round(float(value), 6) for value in row]
                                    for row in np.asarray(body.rest, dtype=np.float64)[
                                        [index for index, _name in pairs]]],
                           'rest_max_cm': [[round(float(value), 4) for value in row]
                                           for row in self._rest_max_cm(
                                               [index for index, _name in pairs], body)]},
                          timeout=8.0)
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, '3ds Max did not answer the bind (%s)' % error
        if not reply.get('ok'):
            return False, str(reply.get('error') or 'Max refused the bind')
        bound = [(int(row), str(node)) for row, node in (reply.get('pairs') or [])]
        if not bound:
            return False, ('3ds Max matched none of this take\'s bones. It has: %s'
                           % ', '.join(joints[:8]))
        self.body = body
        self.body_index = [row for row, _node in bound]
        self.nodes = [node for _row, node in bound]
        hips = [row for row, name in enumerate(body.names) if name == 'Hips']
        self.root = (self.body_index.index(hips[0]) if hips and hips[0] in self.body_index
                     else 0)
        try:
            _post(_url(self.port, '/reset'),
                  {'nodes': self.nodes, 'frame_start': int(frame)}, timeout=6.0)
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, '3ds Max did not answer the reset (%s)' % error
        self.sent = 0
        self.failures = 0
        self.error = ''
        self._stop.clear()
        self._wake.clear()
        if not self.active:
            self._thread = threading.Thread(target=self._pump, name='rgbmocap-max-live',
                                            daemon=True)
            self._thread.start()
        if publish_first:
            self.publish(frame)
        renamed = sum(1 for _index, node in pairs if node not in body.names)
        return True, ('Streaming %d of %d bones%s.'
                      % (len(self.nodes), len(body.names),
                         ', matched after the prefix (e.g. %s)' % self.nodes[0]
                         if renamed and self.nodes else ''))

    # -- streaming -------------------------------------------------------
    def publish(self, index, key=None):
        """Offer one frame. Every offered frame is sent, in order.

        ``key`` is the timeline frame to key it at if 3ds Max is recording.
        """
        if self.body is None:
            return
        with self._lock:
            index = int(index)
            if key is not None:
                self._keys[index] = int(key)
            if index in self._pending_set:
                return
            if len(self._pending) >= PUBLISH_LIMIT:
                stale = self._pending.popleft()
                self._pending_set.discard(stale)
                self.skipped += 1
            self._pending.append(index)
            self._pending_set.add(index)
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(1.5)
        self.channel.close()

    def frame_payload(self, index):
        """One frame: how far every matched joint turned, and where the root is."""
        body = self.body
        index = max(0, min(int(body.count) - 1, int(index)))
        full = np.asarray(body.frames[index], dtype=np.float64)
        display = np.asarray(body.display, dtype=np.float64)
        rest_frames = np.asarray(body.rest_frames, dtype=np.float64)
        # How far each bone has turned from the take's rest, as a turn in WORLD
        # axes (frames @ display @ rest^T - the same turn the Unreal link sends).
        # A turn about the bone's own axes would only land right on a rig whose
        # bones have the app's axes, and 3ds Max gives an imported bone its own.
        posed = np.einsum('jik,jkl->jil', full[:, :3, :3], display)
        deltas = np.einsum('jik,jlk->jil', posed, rest_frames)
        rows = np.asarray(self.body_index)
        scene = None if self.transform is None else np.asarray(self.transform, dtype=np.float64)
        if scene is not None:
            # The scene's turn of this person: every bone turns about the same
            # world axes, so the turn from rest is seen through that rotation.
            deltas = np.einsum('ik,jkl,ml->jim', scene[:3, :3], deltas, scene[:3, :3])
        deltas = deltas[rows]
        positions = np.asarray(full[rows][:, :3, 3], dtype=np.float64) * 100.0
        heights = {int(row): float(full[row][1, 3]) * 100.0 for row in rows}
        if scene is not None:
            rest_root = np.asarray(body.rest, dtype=np.float64)[rows[self.root]] * 100.0
            # Travel is seen through the scene's turn, from where this person
            # started; the floor is where the scene puts it.
            positions = rest_root + (positions - rest_root) @ scene[:3, :3].T
            at = np.asarray(full[rows][:, :3, 3], dtype=np.float64) @ scene[:3, :3].T + scene[:3, 3]
            heights = {int(row): float(at[slot][1]) * 100.0 for slot, row in enumerate(rows)}
        # App Y-up to Max Z-up, as a turn and a point at once.
        deltas = np.einsum('ik,jkl,ml->jim', A, deltas, A)
        positions = positions @ A.T
        if self.facing:
            yaw = _yaw_z(self.facing)
            deltas = np.einsum('ik,jkl,ml->jim', yaw, deltas, yaw)
            positions = positions @ yaw.T
        # Heights of the take's feet and toes above its floor: the receiver stands
        # the rig's own feet at the same heights.
        # Keyed by the take's own bone name: every character in the scene finds its
        # own foot nodes from it.
        ground = {}
        for row in self.body_index:
            if str(body.names[row]) in ('LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase'):
                ground[str(body.names[row])] = round(heights[int(row)], 3)
        return {
            'at': int(index),
            'nodes': list(self.nodes),
            'deltas': np.round(deltas, 5).tolist(),
            'root': int(self.root),
            'root_position': [round(float(value), 4) for value in positions[self.root]],
            'ground': ground,
            'contacts': self.contacts(index),
            'protocol': 2,
            'sender': self.sender,
            'person': self.person,
            'check': bool(os.environ.get('RGB_MOCAP_MAX_CHECK')),
            **({'key': self._keys[index]} if index in self._keys else {}),
        }

    # -- keyframes -------------------------------------------------------
    def record(self, on, fps=30.0, clear=True):
        """Start or stop keying every frame 3ds Max receives (Record Live)."""
        reply = self.channel.post('/record', {'on': bool(on), 'fps': float(fps),
                                              'clear': bool(clear)}, timeout=60.0)
        return reply

    def bake_payload(self, index, key):
        """One frame for a bake, keyed at the timeline frame ``key``."""
        payload = self.frame_payload(index)
        payload['key'] = int(key)
        return payload

    def contacts(self, index):
        """Which feet the take has planted at this frame (see BodyStream.contacts)."""
        found = getattr(self.body, 'contacts', None)
        return found(index) if found is not None else {}

    def rest_payload(self):
        """Every turn the identity: the character left in its own rest pose.

        Used to prove the link before any animation is trusted. If a T-pose is
        not a T-pose on the other side, no amount of tuning will look right.
        """
        body = self.body
        rest = np.asarray(body.rest, dtype=np.float64)
        rows = np.asarray(self.body_index)
        positions = rest[rows] * 100.0
        positions = positions @ A.T
        if self.facing:
            positions = positions @ _yaw_z(self.facing).T
        return {
            'at': 0,
            'nodes': list(self.nodes),
            'deltas': np.tile(np.eye(3), (len(self.nodes), 1, 1)).tolist(),
            'root': int(self.root),
            'root_position': [round(float(value), 4) for value in positions[self.root]],
        }

    def _pump(self):
        while not self._stop.is_set():
            with self._lock:
                busy = bool(self._pending)
            if not busy:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            with self._lock:
                batch = []
                while self._pending and len(batch) < PUBLISH_BATCH:
                    batch.append(self._pending.popleft())
                for index in batch:
                    self._pending_set.discard(index)
            payloads = []
            for index in batch:
                try:
                    payloads.append(self.frame_payload(index))
                except Exception as error:      # noqa: BLE001 - reported, not raised
                    self.failures += 1
                    self.error = str(error)
            if not payloads:
                continue
            try:
                reply = self.channel.post('/frames', {'frames': payloads}, timeout=4.0)
            except Exception as error:          # noqa: BLE001 - reported, not raised
                self.failures += 1
                self.error = str(error)
                with self._lock:
                    for index in reversed(batch):
                        if index not in self._pending_set:
                            self._pending.appendleft(index)
                            self._pending_set.add(index)
                if self.failures > 600:
                    self.error = '3ds Max stopped answering on port %d (%s)' % (
                        self.port, error)
                    self.body = None
                    return
                time.sleep(0.05)
                continue
            if not reply.get('ok'):
                self.failures += 1
                self.error = str(reply.get('error') or 'refused')
                continue
            self.failures = 0
            self.sent += len(payloads)

    def report(self):
        state = self.status() or {}
        return dict(active=self.active, port=self.port, sent=self.sent,
                    skipped=self.skipped,
                    bound=state.get('bound', 0), applied=state.get('applied', 0),
                    errors=list(state.get('errors') or []) + ([self.error] if self.error else []))
