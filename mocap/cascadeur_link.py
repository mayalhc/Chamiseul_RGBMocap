# Copyright (c) 2026 Chamiseul. All rights reserved.
# Proprietary: redistribution prohibited. See LICENSE-CHAMISEUL.md in the repository root.
"""Live link: stream what the app shows into an open Cascadeur scene.

Cascadeur has no listening socket of its own, so the link is created from this
side. If nothing answers on 9566, the source of ``mocap/cascadeur_receiver.py``
is posted to Cascadeur's MCP script server (127.0.0.1:8765/run), which execs it
inside the application; that receiver then listens on 9566 and drives the open
character from a QTimer on Cascadeur's own Qt loop. Frames go to 9566 - never
through the script server, which runs work on a fixed three second idle timer.

What is sent per joint is how far it has turned from that joint's rest, in
world axes, and the receiver composes that onto the rest the character in
Cascadeur actually has. That is the difference between a pose and a knot: a
character rests in its own orientation, and only a turn is a shared language.

Nothing about the app's rig is otherwise assumed. Joints are matched by name,
then by the name after a prefix colon so a Mixamo character binds as it stands,
and only the root's position is sent, so a skeleton with different bone lengths
is never stretched.

The stream is not turned about the vertical by default: the exported base body
and the app's takes share one facing, and a half turn applied to the animation
but not to the rest pose made the character snap round on the first frame. Set
``RGB_MOCAP_CASCADEUR_FACING`` to yaw the stream (degrees) if a character was
imported facing the other way.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import socket
from collections import deque

import numpy as np

HOST = '127.0.0.1'
PORT = 9566
MCP_PORT = 8765
RECEIVER = 'cascadeur_receiver.py'
# Frames waiting to be sent, and how many go in one request. The writer
# inside Cascadeur keys a whole batch in one modify session, so batching
# here is what keeps a 30 fps take complete instead of full of holes.
PUBLISH_LIMIT = 240
PUBLISH_BATCH = 6
DEFAULT_FACING = 0.0

# Both ends are on this machine, so a proxy must never be consulted: the
# default opener reads the system's proxy settings, and a machine configured
# to send everything through one turns "is Cascadeur listening" into a
# question asked of a server on the internet.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


CHARACTER_FILE = Path('assets') / 'cascadeur' / 'base_body.fbx'


def base_character_path(root):
    """The prepared T-pose base body that ships with an installation."""
    return Path(root) / CHARACTER_FILE


def receiver_source():
    path = Path(__file__).with_name(RECEIVER)
    if not path.is_file():
        raise RuntimeError('This build carries no %s.' % RECEIVER)
    return path.read_text(encoding='utf-8')


def receiver_version():
    """The VERSION the receiver source in this build declares."""
    import re
    match = re.search(r'^VERSION\s*=\s*[\'\"]([^\'\"]+)', receiver_source(), re.M)
    return match.group(1) if match else ''


def facing_angle():
    try:
        return float(os.environ.get('RGB_MOCAP_CASCADEUR_FACING') or DEFAULT_FACING)
    except ValueError:
        return DEFAULT_FACING


class _Channel:
    """One kept-alive connection for the frame stream.

    A fresh connection per frame costs about 190 ms inside Cascadeur - more
    than the frame is worth by an order of magnitude - so the stream holds one
    open and reconnects only when it breaks. Status and binding still use the
    plain one-shot helpers below: they are asked once a second at most.
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


def _yaw(degrees):
    """Rotation about the vertical, for a Y-up world."""
    angle = np.radians(float(degrees))
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


class CascadeurLink:
    """One live stream into the open Cascadeur scene.

    Publishing never blocks the caller: the newest frame wins, and a worker
    thread hands it over, so a scrub or a dropped link cannot stall playback.
    """

    def __init__(self, port=PORT, mcp_port=MCP_PORT):
        self.port = int(port)
        self.mcp_port = int(mcp_port)
        self.body = None
        self.names = []
        self.body_index = []
        self.root = 0
        self.facing = facing_angle()
        self.messages = []
        self.sent = 0
        self.failures = 0
        self.error = ''
        self.channel = _Channel(port=self.port)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._pending = deque()
        self._pending_set = set()
        self.skipped = 0
        self._oneshot = None
        self._lock = threading.Lock()

    # -- connecting ------------------------------------------------------
    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self, timeout=1.5):
        """What the link inside Cascadeur says about itself, or None."""
        try:
            return _get(_url(self.port, '/status'), timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _joints(self, timeout=5.0):
        try:
            found = _get(_url(self.port, '/joints'), timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError):
            return []
        return list(found.get('names') or []) if found.get('ok') else []

    def connect(self, wait=12.0):
        """Put the receiver inside Cascadeur and start it. Safe to call twice."""
        found = self.status()
        if found and found.get('ok'):
            # Port 9566 is also used by the Blender live link that ships in
            # Cascadeur's animation_scripts (v1.8.0). Reusing it would stream
            # this build's frames into another build's receiver, which is how
            # a character ends up bent for no visible reason. Only the same
            # version is reused; anything else is stopped and replaced.
            running = str(found.get('version') or '?')
            mine = receiver_version()
            if mine and running != mine:
                try:
                    _post(_url(self.port, '/stop'), {}, timeout=4.0)
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                for _ in range(30):
                    if not self.status(timeout=0.4):
                        break
                    time.sleep(0.1)
                else:
                    return False, ('Cascadeur answers on port %d with live link v%s '
                                   'from another tool. Stop that link there, then '
                                   'connect this one (v%s).' % (self.port, running, mine))
            else:
                return True, 'Cascadeur link already running (v%s).' % running

        source = receiver_source()
        stamp = '%d:%s' % (len(source), hashlib.md5(source.encode('utf-8')).hexdigest()[:10])
        boot = (
            "import builtins, types, traceback\n"
            "old = getattr(builtins, '_rgbmocap_cascadeur_live', None)\n"
            "if old is None or getattr(old, 'STAMP', None) != STAMP:\n"
            "    mod = types.ModuleType('rgbmocap_cascadeur_live')\n"
            "    mod.STAMP = STAMP\n"
            "    exec(compile(SOURCE, 'rgbmocap_cascadeur_live', 'exec'), mod.__dict__)\n"
            "    builtins._rgbmocap_cascadeur_live = mod\n"
            "else:\n"
            "    mod = old\n"
            "try:\n"
            "    print('rgbmocap-live', mod.start())\n"
            "except Exception:\n"
            "    print('rgbmocap-live-failed', traceback.format_exc())\n")
        code = 'SOURCE = %r\nSTAMP = %r\n%s' % (source, stamp, boot)
        try:
            reply = _post(_url(self.mcp_port, '/run'), {'code': code}, timeout=20.0)
        except (urllib.error.URLError, OSError) as error:
            return False, ('Cascadeur is not answering on %s:%d. Start Cascadeur and '
                           'make sure its MCP script server is listening. (%s)'
                           % (HOST, self.mcp_port, error))
        except ValueError as error:
            return False, ('%s:%d answered with something that is not JSON (%s)'
                           % (HOST, self.mcp_port, error))
        if not reply.get('ok'):
            return False, ('Cascadeur refused the live link: %s'
                           % (reply.get('error') or 'no detail'))
        said = ' '.join(m.get('text', '') for m in reply.get('messages') or [])
        if 'rgbmocap-live-failed' in said:
            return False, said.split('rgbmocap-live-failed', 1)[1].strip()[-400:]

        # The receiver binds its socket as the module starts, so give it a
        # moment rather than declaring failure on the first try.
        began = time.time()
        while time.time() - began < wait:
            found = self.status()
            if found and found.get('ok'):
                return True, 'Cascadeur link started on %s:%d.' % (HOST, self.port)
            time.sleep(0.4)
        return False, ('The receiver started but is not answering on %s:%d. %s'
                       % (HOST, self.port, said.strip()[-300:]))

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

    # -- streaming -------------------------------------------------------
    # The UE5 mannequin's bone names, per canonical Mixamo name: a document
    # built on the UE5 skeleton (UE5_Quinn and friends) carries these, and
    # the spine/neck are split further than the app's skeleton - the extra
    # bones (spine_04, spine_05, neck_02, the twist and metacarpal bones)
    # simply follow their parents.
    UE5_ALIASES = {
        'Hips': 'pelvis',
        'Spine': 'spine_01', 'Spine1': 'spine_02', 'Spine2': 'spine_03',
        'Neck': 'neck_01', 'Head': 'head',
        'LeftShoulder': 'clavicle_l', 'RightShoulder': 'clavicle_r',
        'LeftArm': 'upperarm_l', 'RightArm': 'upperarm_r',
        'LeftForeArm': 'lowerarm_l', 'RightForeArm': 'lowerarm_r',
        'LeftHand': 'hand_l', 'RightHand': 'hand_r',
        'LeftUpLeg': 'thigh_l', 'RightUpLeg': 'thigh_r',
        'LeftLeg': 'calf_l', 'RightLeg': 'calf_r',
        'LeftFoot': 'foot_l', 'RightFoot': 'foot_r',
        'LeftToeBase': 'ball_l', 'RightToeBase': 'ball_r',
        'LeftHandThumb1': 'thumb_01_l', 'LeftHandThumb2': 'thumb_02_l',
        'LeftHandThumb3': 'thumb_03_l',
        'LeftHandIndex1': 'index_01_l', 'LeftHandIndex2': 'index_02_l',
        'LeftHandIndex3': 'index_03_l',
        'LeftHandMiddle1': 'middle_01_l', 'LeftHandMiddle2': 'middle_02_l',
        'LeftHandMiddle3': 'middle_03_l',
        'LeftHandRing1': 'ring_01_l', 'LeftHandRing2': 'ring_02_l',
        'LeftHandRing3': 'ring_03_l',
        'LeftHandPinky1': 'pinky_01_l', 'LeftHandPinky2': 'pinky_02_l',
        'LeftHandPinky3': 'pinky_03_l',
        'RightHandThumb1': 'thumb_01_r', 'RightHandThumb2': 'thumb_02_r',
        'RightHandThumb3': 'thumb_03_r',
        'RightHandIndex1': 'index_01_r', 'RightHandIndex2': 'index_02_r',
        'RightHandIndex3': 'index_03_r',
        'RightHandMiddle1': 'middle_01_r', 'RightHandMiddle2': 'middle_02_r',
        'RightHandMiddle3': 'middle_03_r',
        'RightHandRing1': 'ring_01_r', 'RightHandRing2': 'ring_02_r',
        'RightHandRing3': 'ring_03_r',
        'RightHandPinky1': 'pinky_01_r', 'RightHandPinky2': 'pinky_02_r',
        'RightHandPinky3': 'pinky_03_r',
    }

    @classmethod
    def _match(cls, body_names, joints):
        """Pairs of (body index, Cascadeur joint) that name the same joint.

        Exact names come first, then the name after the last colon, then the
        UE5 mannequin's bone names - a document built on the UE5 skeleton
        carries `pelvis`/`thigh_l`/`thumb_01_l` where this app says
        `Hips`/`LeftUpLeg`/`LeftHandThumb1`. A Mixamo character imported
        into Cascadeur keeps its `mixamorig:` prefix, and this app's own
        skeleton does not carry it, so without the fallbacks every joint of
        a perfectly ordinary character would be refused.
        """
        available = set(joints)
        by_suffix = {}
        for name in joints:
            by_suffix.setdefault(name.rsplit(':', 1)[-1].lower(), name)
        pairs = []
        for index, name in enumerate(body_names):
            target = name if name in available else by_suffix.get(
                name.rsplit(':', 1)[-1].lower())
            if target is None:
                alias = cls.UE5_ALIASES.get(name)
                if alias is not None and alias in available:
                    target = alias
            if target is not None:
                pairs.append((index, target))
        return pairs

    def send_character(self, fbx, new_scene=True, timeout=180.0):
        """Put the prepared T-pose base body into Cascadeur as its own scene.

        Cascadeur does not accept this app's own FBX writer output, so the file
        is the one Blender exports (rest pose, no animation). It has to be a
        T-pose or an A-pose: everything the live link writes is composed onto
        the character's rest, so a character that rests in a take pose is bent
        from its very first frame.
        """
        fbx = str(Path(fbx))
        code = """
import json, os
out = {'file': %r}
out['exists'] = os.path.isfile(out['file'])
app = csc.app.get_application(); sm = app.get_scene_manager()
if not out['exists']:
    out['error'] = 'the base character file is missing'
else:
    if %r:
        made = sm.create_application_scene(); sm.set_current_scene(made)
    view = sm.current_scene(); scene = view.domain_scene()
    tool = app.get_tools_manager().get_tool('FbxSceneLoader')
    loader = tool.get_fbx_loader(view)
    for method in ('import_model', 'add_model'):
        try:
            getattr(loader, method)(out['file']); out['method'] = method; break
        except Exception as error:
            out[method + '_error'] = repr(error)[:200]
    viewer = scene.model_viewer(); behaviours = viewer.behaviour_viewer(); data = scene.data_viewer()
    out['objects'] = len(list(viewer.get_objects()))
    out['scene'] = str(app.current_scene().name())
    rest = {}
    for item in viewer.get_objects():
        name = viewer.get_object_name(item)
        if name not in ('Hips', 'Head', 'LeftShoulder', 'RightShoulder', 'LeftArm',
                        'RightArm', 'LeftFoot', 'RightFoot'):
            continue
        transform = behaviours.get_behaviour_by_name(item, 'Transform')
        if transform.is_null():
            continue
        where = data.get_data_value(behaviours.get_behaviour_data(transform, 'global_position'), 0)
        rest[name] = [float(where[i]) for i in range(3)]
    out['rest'] = rest
print('CHAR ' + json.dumps(out))
""" % (fbx, bool(new_scene))
        try:
            reply = _post(_url(self.mcp_port, '/run'), {'code': code}, timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, ('Cascadeur is not answering on %s:%d (%s)'
                           % (HOST, self.mcp_port, error)), {}
        if not reply.get('ok'):
            return False, ('Cascadeur refused the character import: %s'
                           % (reply.get('error') or 'no detail')), {}
        found = None
        for message in reply.get('messages') or []:
            line = message.get('text') or ''
            if line.startswith('CHAR '):
                found = json.loads(line[5:])
        if found is None:
            return False, 'Cascadeur answered without the import result', {}
        if found.get('error'):
            return False, str(found['error']), found
        rest = found.get('rest') or {}
        checks = {}
        hips = rest.get('Hips')
        head = rest.get('Head')
        if hips and head:
            checks['hips_height_cm'] = round(hips[1], 1)
            checks['head_above_hips_cm'] = round(head[1] - hips[1], 1)

        def elevation(top, base):
            if not top or not base:
                return None
            vector = np.asarray(top, dtype=np.float64) - np.asarray(base, dtype=np.float64)
            length = float(np.linalg.norm(vector))
            if length < 1e-6:
                return None
            return round(float(np.degrees(np.arcsin(max(-1.0, min(1.0, vector[1] / length))))), 1)

        checks['left_arm_elevation_deg'] = elevation(rest.get('LeftArm'), rest.get('LeftShoulder'))
        checks['right_arm_elevation_deg'] = elevation(rest.get('RightArm'), rest.get('RightShoulder'))
        for side in ('Left', 'Right'):
            foot = rest.get(side + 'Foot')
            if foot:
                checks[side.lower() + '_foot_cm'] = round(foot[1], 1)
        height = checks.get('head_above_hips_cm')
        arms = (checks.get('left_arm_elevation_deg'), checks.get('right_arm_elevation_deg'))
        feet = (checks.get('left_foot_cm'), checks.get('right_foot_cm'))
        ok = bool(int(found.get('objects') or 0) >= 50
                  and height is not None and 40.0 <= height <= 85.0
                  and all(a is not None and abs(a) <= 35.0 for a in arms)
                  and all(f is not None and -15.0 <= f <= 30.0 for f in feet))
        if not ok:
            return False, ('the imported body is not standing in a T-pose or A-pose '
                           '(head %s cm over hips, arms %s deg, feet %s cm)'
                           % (height, arms, feet)), checks
        return True, ('Base character in Cascadeur: %d objects, head +%.1f cm over '
                      'hips, arms level.' % (int(found.get('objects') or 0), height)), checks

    def start(self, body, frame=0, follow=True, publish_first=True):
        """Begin a take on `body`, rewinding Cascadeur to `frame`.

        `publish_first=False` binds and streams nothing yet: the frame the app
        is showing is offered by the caller later. That is what makes a rest
        check a rest check - otherwise the take's own frame lands right after
        the T-pose and the character is already posed when it is measured.
        """
        if not self.status():
            return False, 'Cascadeur is not linked. Connect the link first.'
        joints = self._joints()
        if not joints:
            # The receiver fills its object list on its own tick, so the first
            # question can arrive before that list exists.
            began = time.time()
            while not joints and time.time() - began < 5.0:
                time.sleep(0.15)
                joints = self._joints()
        pairs = self._match(list(body.names), joints)
        if not pairs:
            return False, ('Cascadeur has no joint named like this take (it has %d '
                           'objects). Import the character into Cascadeur first, '
                           'either the one exported from here or a Mixamo one.'
                           % len(joints))
        self.body = body
        self.body_index = [index for index, _name in pairs]
        # The stream names are Cascadeur's own, so the receiver binds exactly;
        # rotations stay in the app's bone order through `body_index`.
        self.names = [name for _index, name in pairs]
        hips = [row for row, name in enumerate(body.names) if name == 'Hips']
        self.root = (self.body_index.index(hips[0]) if hips and hips[0] in self.body_index
                     else 0)
        try:
            # The app's rest BONE DIRECTIONS per matched row: where each
            # segment points at rest and the direction it hangs from - the
            # geometric quantities both skeletons share, measured from the
            # rest positions. Yawed to the streamed world like everything
            # else. The receiver builds its per-joint frame correction from
            # these instead of trusting the export skeleton's rest rotation
            # tables, whose finger rows sit in another basis.
            rest_origins = np.asarray(body.rest, dtype=np.float64)
            parents_arr = np.asarray(body.parents)
            children_map = {}
            for child, parent in enumerate(parents_arr):
                if parent >= 0:
                    children_map.setdefault(int(parent), []).append(int(child))
            yaw = _yaw(self.facing)
            rest_dirs = []
            for row, j in enumerate(self.body_index):
                parent = int(parents_arr[j])
                if parent >= 0:
                    incoming = rest_origins[j] - rest_origins[parent]
                    length = float(np.linalg.norm(incoming))
                    incoming = incoming / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])
                else:
                    incoming = np.array([0.0, 1.0, 0.0])
                best = (None, -2.0)
                for child in children_map.get(j) or []:
                    delta = rest_origins[child] - rest_origins[j]
                    length = float(np.linalg.norm(delta))
                    if length <= 0.008:
                        continue
                    score = float(np.dot(delta / length, incoming))
                    if score > best[1]:
                        best = (delta / length, score)
                along = best[0]
                if along is None:
                    # A leaf (a fingertip, the head) has no child to point
                    # at: its own incoming direction is what it carries.
                    along = incoming
                across = None
                if along is not None:
                    for other in children_map.get(j) or []:
                        delta = rest_origins[other] - rest_origins[j]
                        length = float(np.linalg.norm(delta))
                        if length > 0.008 and not np.allclose(delta / length, along):
                            across = delta / length
                            break
                if across is None:
                    across = -incoming
                rest_dirs.append([
                    [round(float(v), 6) for v in (along @ yaw.T)],
                    [round(float(v), 6) for v in (across @ yaw.T)]])
            _post(_url(self.port, '/reset'),
                  {'names': self.names, 'frame_start': int(frame), 'follow': bool(follow),
                   'rest_dirs': rest_dirs},
                  timeout=6.0)
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, 'Cascadeur did not answer the reset (%s)' % error
        # The reset only names the joints; the receiver binds them on its next
        # tick. Waiting for that is what tells the status line how many joints
        # the character really has, rather than how many were offered.
        began = time.time()
        while time.time() - began < 3.0:
            if (self.status() or {}).get('bound'):
                break
            time.sleep(0.1)
        self.sent = 0
        self.failures = 0
        self.error = ''
        self._stop.clear()
        self._wake.clear()
        if not self.active:
            self._thread = threading.Thread(target=self._pump, name='rgbmocap-cascadeur-live',
                                            daemon=True)
            self._thread.start()
        if publish_first:
            self.publish(frame)
        bound = (self.status() or {}).get('bound') or len(pairs)
        renamed = sum(1 for _index, name in pairs if name not in body.names)
        return True, ('Streaming %d of %d joints%s.'
                      % (bound, len(body.names),
                         ', matched by name after the prefix (e.g. %s)'
                         % pairs[0][1] if renamed else ''))

    def publish(self, index):
        """Offer one frame. Every offered frame is sent, in order.

        This used to keep only the newest frame. When the writer inside
        Cascadeur could not keep up, the frames in between were never sent at
        all - so a playback wrote a different subset each time and the only way
        to fill a take was to play it again and again. The queue below keeps
        them; only a backlog past PUBLISH_LIMIT drops, oldest first.
        """
        if self.body is None:
            return
        with self._lock:
            index = int(index)
            if index in self._pending_set:
                return
            if len(self._pending) >= PUBLISH_LIMIT:
                stale = self._pending.popleft()
                self._pending_set.discard(stale)
                self.skipped += 1
            self._pending.append(index)
            self._pending_set.add(index)
        self._wake.set()

    def publish_payload(self, payload):
        """Send one hand-made pose instead of a take frame (a T-pose check)."""
        with self._lock:
            self._oneshot = payload
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(1.5)
        self.channel.close()

    def frame_payload(self, index):
        """One frame: where every joint points in the world, and where it is."""
        body = self.body
        index = max(0, min(int(body.count) - 1, int(index)))
        full = np.asarray(body.frames[index], dtype=np.float64)
        data = full[self.body_index]
        rotation = data[:, :3, :3].copy()
        position = data[:, :3, 3].copy()
        # Where each joint points, taken from the take's own joint positions.
        # The committed 52-node rotations are written in the export skeleton's
        # own basis, so a "turn away from rest" built from them is not a
        # human-readable rotation for another character: aiming at the next
        # joint is, and it is the geometry the app actually draws.
        aims = []
        second_aims = []
        refs = []
        rows_of = {bone: row for row, bone in enumerate(self.body_index)}
        origins = full[:, :3, 3]
        parents = np.asarray(body.parents)
        children = {}
        for child, parent in enumerate(parents):
            if parent >= 0:
                children.setdefault(int(parent), []).append(child)

        def aim_and_first_child(bone):
            """The direction this bone carries on in, from its joint positions.

            Where a bone branches, the child that CONTINUES it is the one that
            should be followed: the hips carry the spine and both legs, and
            SMPL-X moves its hip joints apart with the pose while a Mixamo
            pelvis keeps them fixed. Following a leg there would tilt the whole
            torso to satisfy a body width; following the spine keeps the body
            stacked the way it is in the app. "Continuing" means the child
            whose direction most agrees with the bone's own incoming direction,
            and for the root - which has none - straight up.
            """
            parent = int(parents[int(bone)])
            if parent >= 0:
                incoming = origins[int(bone)] - origins[parent]
                length = float(np.linalg.norm(incoming))
                incoming = incoming / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])
            else:
                incoming = np.array([0.0, 1.0, 0.0])
            best = (np.zeros(3), None, -2.0)
            for child in children.get(int(bone)) or []:
                delta = origins[child] - origins[bone]
                length = float(np.linalg.norm(delta))
                if length <= 0.008:
                    continue
                unit = delta / length
                score = float(np.dot(unit, incoming))
                if score > best[2]:
                    best = (unit, child, score)
            return best[0], best[1]

        for bone in self.body_index:
            aim, child = aim_and_first_child(bone)
            aims.append(aim)
            # A joint with two children needs both to be pinned down: the hips
            # carry the spine AND both legs, and one direction leaves the legs
            # free to swing around the spine. So a second vector is sent, with
            # the two joints that define it: one anchor and one joint measured
            # FROM that anchor. A second child hangs off this joint; a joint
            # further down the limb hangs off the first child.
            second = np.zeros(3)
            # The aimed child is always named, so the receiver aims at the same
            # joint on its own rest instead of guessing which one it was.
            pair = [rows_of.get(child, -1) if child is not None else -1,
                    rows_of.get(bone, -1), -1]
            if child is not None:
                for other in (children.get(int(bone)) or [])[1:]:
                    delta = origins[other] - origins[bone]
                    length = float(np.linalg.norm(delta))
                    if length > 0.008 and other in rows_of:
                        second = delta / length
                        pair = [rows_of[child], rows_of[bone], rows_of[other]]
                        break
            if not second.any() and child is not None:
                below, grand = aim_and_first_child(child)
                if grand is not None and grand in rows_of and child in rows_of:
                    second = (origins[grand] - origins[child])
                    second = second / float(np.linalg.norm(second))
                    pair = [rows_of[child], rows_of[child], rows_of[grand]]
            second_aims.append(second)
            refs.append(pair)
        # The whole transfer is one sentence per joint: how far that joint's
        # WORLD orientation has turned from the app's own rest, with the
        # facing yaw folded in. World turns carry no bone-frame convention -
        # a joint's roll on either skeleton cannot twist them - and the
        # receiver composes each one onto the character's own world rest,
        # parents first. (The earlier local form - rest_local @ local -
        # compounded every skeleton's roll difference down the chain and
        # knotted the character.)
        display_all = np.asarray(body.display, dtype=np.float64)
        all_world = np.einsum('jik,jkl->jil', full[:, :3, :3], display_all)
        rest_world = np.einsum('jik,jkl->jil',
                               np.asarray(body.rest_frames, dtype=np.float64),
                               display_all)
        rest_world_t = rest_world.swapaxes(-1, -2)
        if self.facing:
            yaw = _yaw(self.facing)
            yawed_world = np.einsum('ik,jkc->jic', yaw, all_world)
            deltas = np.einsum('jik,jkl->jil', rest_world_t, yawed_world)
            rotation = np.einsum('ik,jkc->jic', yaw, rotation)
            position = position @ yaw.T
        else:
            deltas = np.einsum('jik,jkl->jil', rest_world_t, all_world)
        deltas = deltas[self.body_index]

        # Where each joint should POINT, as rows of this stream: the child
        # that continues the bone, and a second one that fixes its roll. The
        # receiver rebuilds orientations from these and the streamed joint
        # positions alone - no skeleton frame convention on either side,
        # which is what every rotation-table form of this transfer got wrong.
        along = []
        across = []
        for bone in self.body_index:
            aim, child = aim_and_first_child(bone)
            row_child = rows_of.get(child, -1) if child is not None else -1
            row_second = -1
            if child is not None:
                for other in (children.get(int(bone)) or [])[1:]:
                    delta = origins[other] - origins[bone]
                    if float(np.linalg.norm(delta)) > 0.008 and other in rows_of:
                        row_second = rows_of[other]
                        break
                if row_second < 0:
                    below, grand = aim_and_first_child(child)
                    if grand is not None and grand in rows_of:
                        row_second = rows_of[grand]
            along.append(row_child)
            across.append(row_second)

        # Turns and the root's position only. Sending every joint's position
        # would drag a character with different bone lengths onto this
        # skeleton's - a Mixamo character is not this body's shape, and a
        # stretched mesh is worse than a root that is a few centimetres out.
        return {
            'names': self.names,
            'rotations': np.round(rotation, 5).tolist(),
            'positions': np.round(position, 5).tolist(),
            'deltas': np.round(np.asarray(deltas, dtype=np.float64), 5).tolist(),
            'along': along,
            'across': across,
            'root': self.root,
            'at': int(index),
        }

    def _rest_aims(self, origins):
        """along/across rows for a pose given only by its joint positions.

        The receiver fixes a joint's roll from a second direction; without one
        a pure turn about the vertical (the facing yaw) is invisible to the
        aim path, because the spine points up either way.
        """
        rows_of = {bone: row for row, bone in enumerate(self.body_index)}
        parents = np.asarray(self.body.parents)
        children = {}
        for child, parent in enumerate(parents):
            if parent >= 0:
                children.setdefault(int(parent), []).append(child)

        def aim_and_first_child(bone):
            parent = int(parents[int(bone)])
            if parent >= 0:
                incoming = origins[int(bone)] - origins[parent]
                length = float(np.linalg.norm(incoming))
                incoming = incoming / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])
            else:
                incoming = np.array([0.0, 1.0, 0.0])
            best = (np.zeros(3), None, -2.0)
            for child in children.get(int(bone)) or []:
                delta = origins[child] - origins[bone]
                length = float(np.linalg.norm(delta))
                if length <= 0.008:
                    continue
                direction = delta / length
                score = float(np.dot(direction, incoming))
                if score > best[2]:
                    best = (direction, child, score)
            return best[0], best[1]

        along, across = [], []
        for bone in self.body_index:
            _aim, child = aim_and_first_child(bone)
            row_child = rows_of.get(child, -1) if child is not None else -1
            row_second = -1
            if child is not None:
                for other in (children.get(int(bone)) or [])[1:]:
                    delta = origins[other] - origins[bone]
                    if float(np.linalg.norm(delta)) > 0.008 and other in rows_of:
                        row_second = rows_of[other]
                        break
                if row_second < 0:
                    _below, grand = aim_and_first_child(child)
                    if grand is not None and grand in rows_of:
                        row_second = rows_of[grand]
            along.append(row_child)
            across.append(row_second)
        return along, across

    def rest_payload(self):
        """The character standing in its own rest pose, with nothing animated.

        Used to prove the link itself: if a T-pose is not a T-pose on the other
        side, no amount of animation tuning will look right. Every turn is the
        identity, so the character on the other side is left exactly in its own
        rest pose - the one pose whose answer is not a matter of taste.
        """
        body = self.body
        rest = np.asarray(body.rest, dtype=np.float64)
        # The facing yaw lives in the streamed POSITIONS only. The receiver
        # rebuilds every joint's turn from those positions, and the character's
        # own rest already faces the same way as the yawed app rest, so those
        # turns come out as the identity. Putting the yaw in the root's turn as
        # well turned the root alone by half a turn while every child stayed
        # put - the whole body twisted at the spine (measured 170 degrees).
        deltas = np.tile(np.eye(3), (len(self.names), 1, 1))
        names = list(body.names)
        hips = names.index('Hips') if 'Hips' in names else 0
        root_row = self.body_index.index(hips) if hips in self.body_index else 0
        # Every joint's rest position, not only the root. The receiver rebuilds
        # each joint's turn from the directions between the streamed joint
        # positions; a row left at the origin is a real direction towards the
        # world origin, and the hips' children (both legs and the spine) were
        # swung by it on every rest write.
        positions = np.asarray(rest, dtype=np.float64)[self.body_index].copy()
        if self.facing:
            positions = positions @ _yaw(self.facing).T
        along, across = self._rest_aims(np.asarray(rest, dtype=np.float64))
        return {'names': self.names,
                'rotations': np.round(np.asarray(
                    body.rest_frames, dtype=np.float64)[self.body_index], 5).tolist(),
                'positions': np.round(positions, 5).tolist(),
                'deltas': np.round(deltas, 5).tolist(),
                'along': along,
                'across': across,
                'root': self.root,
                'at': 0}

    def _pump(self):
        while not self._stop.is_set():
            with self._lock:
                busy = bool(self._pending) or self._oneshot is not None
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
                oneshot, self._oneshot = self._oneshot, None
            payloads = [oneshot] if oneshot is not None else []
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
                    self.error = 'Cascadeur stopped answering on port %d (%s)' % (self.port, error)
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
                    bound=state.get('bound', 0), written=state.get('written', 0),
                    errors=list(state.get('errors') or []) + ([self.error] if self.error else []))
