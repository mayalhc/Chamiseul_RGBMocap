"""Stream a take straight into Unreal, without Blender in between.

The receiver is the MotionForge Live Link plugin in the UE project: it connects
to a TCP server and reads one JSON object per line - a ``hello`` describing the
skeleton, then a ``frame`` per update. Positions are centimetres with +X
forward, which is what the plugin converts to bone offsets.

Two things make a body built here readable by a Quinn or a MetaHuman:

* the joints are named the way Unreal names them, so the plugin's retargeter
  finds them by name (`upperarm_l`, not `LeftArm`);
* the rotation of each bone is sent as a turn away from its own rest, which the
  plugin composes onto the character's reference pose. A pose measured on one
  body therefore drives another body - the offset between the two rest poses is
  accounted for on the Unreal side, where the reference pose is known exactly.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_PORT = 9560
PROTOCOL = 1
CM = 100.0
# The capture space is Y-up with the take facing -Z; Unreal is Z-up, +X forward.
#
# Swapping the two up axes, rather than cycling all three, is what carries a
# right-handed space into a left-handed one. A cycle keeps the handedness and
# quietly mirrors the body, which folds the limbs onto each other instead of
# letting them point anywhere. The plugin on the other end swaps the same two
# axes, so the pair agrees on both the positions and the rotations.
# The capture space is Y up with the take facing -Z, and Unreal's is Z up with
# +X forward and +Y to the right. This one mapping does both: it lifts Y onto
# Unreal's Z, sends the take's forward -Z to +X, and reverses the handedness the
# two spaces disagree about. Anything that only swaps two axes leaves the body
# pitched over on its face, which is what it looked like before this.
# Measured, not assumed: the rest pose the app streams was laid over the reference
# pose Unreal reports for the imported base_body, bone direction by bone
# direction, over every signed axis permutation. This one lines the legs, spine,
# arms and head up to within a few degrees; the mapping assumed before was a
# quarter turn about the vertical away, so every limb bent about the wrong axis.
#   Unreal X = -app X,  Unreal Y = -app Z,  Unreal Z = app Y
# (a reflection, which is what carries a right-handed space into Unreal's
# left-handed one). The imported body therefore stands facing -Y at actor
# yaw 0; turn the actor, not the stream, to face it another way.
TO_UNREAL = np.array([[-1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])


# Kept for callers that still import it.
BASIS = TO_UNREAL

# The app's 52-bone body uses Mixamo names, Unreal uses its own.
UNREAL_NAMES = {
    'Hips': 'pelvis', 'Spine': 'spine_01', 'Spine1': 'spine_02', 'Spine2': 'spine_03',
    'Neck': 'neck_01', 'Head': 'head',
    'LeftShoulder': 'clavicle_l', 'LeftArm': 'upperarm_l', 'LeftForeArm': 'lowerarm_l',
    'LeftHand': 'hand_l',
    'RightShoulder': 'clavicle_r', 'RightArm': 'upperarm_r', 'RightForeArm': 'lowerarm_r',
    'RightHand': 'hand_r',
    'LeftUpLeg': 'thigh_l', 'LeftLeg': 'calf_l', 'LeftFoot': 'foot_l', 'LeftToeBase': 'ball_l',
    'RightUpLeg': 'thigh_r', 'RightLeg': 'calf_r', 'RightFoot': 'foot_r',
    'RightToeBase': 'ball_r',
}
for _side, _suffix in (('Left', 'l'), ('Right', 'r')):
    for _finger, _word in (('Thumb', 'thumb'), ('Index', 'index'), ('Middle', 'middle'),
                           ('Ring', 'ring'), ('Pinky', 'pinky')):
        for _joint in (1, 2, 3):
            UNREAL_NAMES['%sHand%s%d' % (_side, _finger, _joint)] = '%s_0%d_%s' % (
                _word, _joint, _suffix)


def unreal_name(mixamo):
    return UNREAL_NAMES.get(mixamo, mixamo)


class UnrealLink:
    """One TCP server that hands the displayed take to Unreal Live Link."""

    def __init__(self, port=DEFAULT_PORT, subject='RGBMoCap', host='127.0.0.1'):
        self.host = host
        self.port = int(port)
        self.subject = subject
        self._server = None
        self._thread = None
        self._clients = []
        self._lock = threading.Lock()
        self._hello = None
        self._rest = None
        self._names = None
        self._index = 0
        self._started = 0.0
        self.frames = 0
        self.error = ''
        # The take's own rest, taken from the first frame it sends. The stream
        # carries a change from rest per bone and the receiver has to know what
        # the rest was; the take's stored rest is in another set of axes, so the
        # frame it starts on is what everything is measured from.
        self._reference = None

    @property
    def active(self):
        return self._server is not None

    def clients(self):
        with self._lock:
            return len(self._clients)

    def start(self):
        if self.active:
            return True, 'Already listening on %s:%d' % (self.host, self.port)
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(8)
            server.settimeout(0.5)
        except OSError as error:
            self.error = str(error)
            return False, 'Cannot listen on %s:%d (%s)' % (self.host, self.port, error)
        self._server = server
        self._index = 0
        self._reference = None
        self.frames = 0
        self._origin = None
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._accept, args=(server,),
                                        name='rgbmocap-unreal-link', daemon=True)
        self._thread.start()
        return True, 'Unreal Live Link on %s:%d as "%s"' % (self.host, self.port, self.subject)

    def stop(self):
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        self._hello = None

    def _accept(self, server):
        while self._server is server:
            try:
                client, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(client)
            if self._hello:
                self._send(client, self._hello)

    # -- skeleton ---------------------------------------------------------
    def describe(self, names, rest, parents):
        """Build the hello once per skeleton. `rest` is (n,3) in metres, Blender axes."""
        key = tuple(names)
        if self._names == key and self._hello:
            return
        rest = np.asarray(rest, dtype=np.float64)
        order = []
        seen = set()
        by_name = {name: index for index, name in enumerate(names)}

        def visit(index):
            name = names[index]
            if name in seen:
                return
            seen.add(name)
            parent = int(parents[index])
            if parent >= 0:
                visit(parent)
            order.append(index)

        for index in range(len(names)):
            visit(index)
        mapped = [index for index in order if unreal_name(names[index]) in UNREAL_NAMES.values()
                  or names[index] in UNREAL_NAMES]
        if not mapped:
            mapped = order
        joints = [unreal_name(names[index]) for index in mapped]
        position = {index: slot for slot, index in enumerate(mapped)}
        links = []
        for index in mapped:
            parent = int(parents[index])
            links.append(position.get(parent))
        joints_rest = []
        for index in mapped:
            # The app's capture space is Y-up metres; the plugin turns that into
            # Unreal's Z-up centimetres, so the numbers go out as they are.
            x, y, z = (TO_UNREAL @ rest[index]) * CM
            joints_rest.append([round(x, 3), round(y, 3), round(z, 3)])
        self._hello = json.dumps({
            'type': 'hello', 'protocol': PROTOCOL, 'skeleton': self.subject,
            'joints': joints, 'parents': links, 'rest': joints_rest,
            'fps': 30.0, 'space': 'local-delta', 'up': 'z'}) + '\n'
        self._names = key
        self._rest = (mapped, rest)
        self._index = 0
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            self._send(client, self._hello)

    # -- one frame --------------------------------------------------------
    def publish(self, names, rest, parents, matrices, frame=0, fps=30.0,
                rest_rotations=None, display_frames=None):
        """`matrices` are the take's 52 world matrices (rest^-1 @ pose gives the delta)."""
        if not self.active:
            return 0
        self.describe(names, rest, parents)
        mapped, _rest_positions = self._rest
        frames = np.asarray(matrices, dtype=np.float64)
        # The root's placement: relative to the take's first frame, so the
        # character walks from wherever it was spawned instead of jumping to
        # the capture's own world coordinates.
        if getattr(self, '_origin', None) is None:
            self._origin = frames[mapped[0]][:3, 3].copy()
        pose = []
        for index in mapped:
            parent = int(parents[index])
            # The bone's turn away from its parent, in the parent's frame - the
            # local delta the plugin composes onto the character's own rest.
            # Measured as a world turn from the rest instead, a take that
            # faces away from its rest arrives as half a turn about a
            # horizontal axis and the character stands on its hands.
            # The bone's turn away from the take's own rest, measured in the
            # take's world axes. The receiver composes this onto the
            # character's rest, so a turn about the parent's own axes would
            # be read as a world turn and twist every limb.
            # The bone's turn away from the take's own rest, in the take's
            # world axes. The receiver composes it with its own rest world
            # orientation, which is what makes the same skeleton land the
            # same way round.
            turn = np.asarray(frames[index][:3, :3], dtype=np.float64)
            if rest_rotations is not None:
                # The turn that carries the take's rest bone onto its posed bone is
                # frames @ display @ rest^T. The display basis sits in the middle
                # because the posed bone frame is frames @ display while the rest
                # frame the offsets are measured in is rest_frames itself; leaving
                # it out sent every bone a turn about the wrong axes (99 degrees
                # off on average against the take's own joint positions).
                if display_frames is not None:
                    turn = turn @ np.asarray(display_frames[index], dtype=np.float64)
                turn = turn @ np.asarray(rest_rotations[index], dtype=np.float64).T
            quaternion = _to_quaternion(TO_UNREAL @ turn @ TO_UNREAL.T)
            pose.append(_axis_angle(quaternion))
        root = mapped[0] if mapped and names[mapped[0]] == 'Hips' else None
        translation = [0.0, 0.0, 0.0]
        if root is not None:
            now = np.asarray(matrices[root], dtype=np.float64)[:3, 3]
            # Travel since the first frame, not the take's standing height:
            # the character's own rest already holds its hips at hip height,
            # so sending the height again lifted the body off the floor.
            # The base body the plugin drives is imported with a root scale of
            # 100, so its bone translations are metres: sending centimetres
            # multiplied every step by 100 and threw the character clear of
            # the level. Travel goes out in the same unit the skeleton uses.
            shift = TO_UNREAL @ (now - self._origin)
            # Already in Unreal's axes: the stream announces local-delta, so the
            # plugin adds these to the root as they arrive.
            translation = [round(float(shift[0]), 3), round(float(shift[1]), 3),
                           round(float(shift[2]), 3)]
        line = json.dumps({'type': 'frame', 'index': int(frame), 'fps': float(fps),
                           'time': round(time.monotonic() - self._started, 4),
                           'translation': translation,
                           'pose': [[round(v, 6) for v in row] for row in pose]}) + '\n'
        self.frames += 1
        with self._lock:
            clients = list(self._clients)
        dead = [client for client in clients if not self._send(client, line)]
        if dead:
            with self._lock:
                self._clients = [client for client in self._clients if client not in dead]
            for client in dead:
                try:
                    client.close()
                except OSError:
                    pass
        return len(clients) - len(dead)

    def _local_rest(self, index, parent):
        frames = self._rest_frames
        if frames is None:
            return np.eye(3)
        here = np.asarray(frames[index], dtype=np.float64)
        if parent < 0:
            return here
        return np.asarray(frames[parent], dtype=np.float64).T @ here

    @staticmethod
    def _local_pose(frames, index, parent):
        here = np.asarray(frames[index], dtype=np.float64)[:3, :3]
        if parent < 0:
            return here
        return np.asarray(frames[parent], dtype=np.float64)[:3, :3].T @ here

    _rest_frames = None

    def set_rest_frames(self, frames):
        self._rest_frames = np.asarray(frames, dtype=np.float64) if frames is not None else None

    @staticmethod
    def _send(client, text):
        try:
            client.sendall(text.encode('utf-8'))
            return True
        except OSError:
            return False


def _quaternion_inverse(quaternion):
    w, x, y, z = quaternion
    norm = w * w + x * x + y * y + z * z
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, -x / norm, -y / norm, -z / norm)


def _multiply(first, second):
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _to_quaternion(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)


def _axis_angle(quaternion):
    w, x, y, z = quaternion
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-8 or abs(angle) < 1e-8:
        return [0.0, 0.0, 0.0]
    return [x / scale * angle, y / scale * angle, z / scale * angle]


def rest_frames(project, names):
    """The take's own rest rotations, when the project carries them."""
    try:
        info = json.loads((Path(project) / 'project.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return info.get('rest_frames')


class MultiLink:
    """One stream per person, so Unreal receives Person 1, Person 2, ...

    Unreal names a Live Link subject after the skeleton its stream describes, and
    a stream carries one skeleton. A scene with several people therefore needs
    one stream each - which is also what makes them separate subjects in the
    Live Link panel, one per actor in the level.
    """

    def __init__(self, base_port=DEFAULT_PORT):
        self.base_port = int(base_port)
        self.links = {}
        self.error = ''

    @property
    def active(self):
        return bool(self.links)

    def start(self, subjects):
        """`subjects` is the list of names to serve, in order."""
        wanted = list(subjects)
        for subject in list(self.links):
            if subject not in wanted:
                self.links.pop(subject).stop()
        started = []
        for slot, subject in enumerate(wanted):
            link = self.links.get(subject)
            if link is None:
                link = self.links[subject] = UnrealLink(self.base_port + slot, subject)
            ok, message = link.start()
            if not ok:
                self.error = message
                return False, message
            started.append('%s on %d' % (subject, link.port))
        return True, 'Unreal: ' + ', '.join(started)

    def publish(self, subject, names, rest, parents, matrices, frame=0, fps=30.0,
                rest_rotations=None, display_frames=None):
        link = self.links.get(subject)
        if link is None:
            return 0
        return link.publish(names, rest, parents, matrices, frame=frame, fps=fps,
                            rest_rotations=rest_rotations, display_frames=display_frames)

    def clients(self):
        return sum(link.clients() for link in self.links.values())

    def frames(self):
        return sum(link.frames for link in self.links.values())

    def stop(self):
        for link in list(self.links.values()):
            link.stop()
        self.links = {}


# -- base character ---------------------------------------------------------
MCP_URL = 'http://127.0.0.1:8000/mcp'
BASE_CHARACTER = Path('assets') / 'cascadeur' / 'base_body.fbx'


class _Mcp:
    """The editor's own MCP endpoint, so the app can ask it to import a file."""

    def __init__(self, url=MCP_URL):
        self.url = url
        self.session = None
        self._id = 0

    def _post(self, payload, notify=False):
        import urllib.request
        headers = {'Content-Type': 'application/json',
                   'Accept': 'application/json, text/event-stream'}
        if self.session:
            headers['Mcp-Session-Id'] = self.session
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            for key, value in response.headers.items():
                if key.lower() == 'mcp-session-id':
                    self.session = value
            body = response.read().decode('utf-8', 'replace')
        if notify or not body.strip():
            return None
        if body.lstrip().startswith('{'):
            return json.loads(body)
        for line in body.splitlines():
            if line.startswith('data:'):
                chunk = line[5:].strip()
                if chunk:
                    try:
                        return json.loads(chunk)
                    except ValueError:
                        continue
        return None

    def call(self, tool, arguments):
        self._post({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                    'params': {'protocolVersion': '2024-11-05', 'capabilities': {},
                               'clientInfo': {'name': 'rgbmocap', 'version': '0.1'}}})
        self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, notify=True)
        self._id += 1
        reply = self._post({'jsonrpc': '2.0', 'id': self._id, 'method': 'tools/call',
                            'params': {'name': tool, 'arguments': arguments}})
        result = (reply or {}).get('result') or {}
        texts = [part.get('text') for part in (result.get('content') or [])
                 if part.get('type') == 'text']
        return bool(texts), (texts[0] if texts else ''), bool(result.get('isError'))


def import_base_character(root, destination='/Game/RGBMoCap', fbx=None):
    """Put the T-pose base body into the open Unreal project, once."""
    source = Path(fbx) if fbx else Path(root) / BASE_CHARACTER
    if not source.is_file():
        return False, 'the base character is missing at %s' % source
    try:
        found, text, failed = _Mcp().call(
            'ImportAsset', {'filePath': str(source), 'destinationPath': destination})
    except Exception as error:                  # noqa: BLE001 - reported to the user
        return False, 'Unreal did not answer (%s)' % error
    if not found:
        return False, 'Unreal answered without a result'
    if failed:
        return False, text[:300]
    try:
        detail = json.loads(text)
    except ValueError:
        return True, text[:300]
    if not detail.get('success'):
        return False, str(detail)[:300]
    assets = str(detail.get('assets') or '')

    # Importing the body is only half of a take: without an actor whose
    # animation follows the stream there is nothing in the level to watch, so
    # the same click also builds and places the character.
    placed = _build_character()
    return True, ('Base character in Unreal: %d asset(s)%s - %s'
                  % (int(detail.get('imported') or 0), placed, assets[:180]))


def _build_character():
    """Ask Unreal to make the character Blueprint and put one down per person."""
    script = Path(__file__).with_name('ue_character_setup.py')
    if not script.is_file():
        return ''
    command = 'py "%s"' % str(script).replace('\\', '/')
    try:
        found, text, failed = _Mcp().call('RunConsoleCommand', {'command': command})
    except Exception as error:                  # noqa: BLE001 - reported to the user
        return ' (character not made: %s)' % str(error)[:80]
    if failed:
        return ' (character not made)'
    if 'RGBMOCAP_SETUP' in (text or ''):
        return ', character placed'
    return ''
