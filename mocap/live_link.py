"""Live link: stream what the app shows to a local Blender client.

Two stream kinds share one localhost link:

* ``character`` — the imported character itself. The app sends where its rest
  file lives, the bone palette behind the displayed poses, and then per-frame
  bone matrices relative to that rest. Blender appends the same rest file, so
  bones, skin, textures and rest pose match the app exactly; only the pose
  changes, with no heuristic retargeting.
* ``joints`` — fallback for a source person with no character on screen: the 22
  canonical body joints, for Blender to aim its own mapped rig at.

All coordinates are converted to Blender's Z-up world before sending.

Wire format (little endian):
  handshake/update  one UTF-8 JSON line, ``{"version": 2, ...}`` sent on connect
                    and again whenever the stream description changes
  frame             b'RGFC' + uint32 frame + float32 fps + uint16 count
                    + count * 16 float32 (4x4 matrices), or
                    b'RGMF' + uint32 frame + float32 fps + 22 * 3 float32
"""
from __future__ import annotations

import json
import os
import select
import socket
import struct
import threading

import numpy as np

NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1',
         'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck',
         'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm',
         'RightForeArm', 'LeftHand', 'RightHand']
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# App world is Y-up; Blender is Z-up. A @ (x, y, z) = (x, -z, y).
A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
A4 = np.eye(4)
A4[:3, :3] = A

MAGIC_CHARACTER = b'RGFC'
MAGIC_JOINTS = b'RGMF'
VERSION = 2
DEFAULT_PORT = 9187


def default_port():
    """Port from RGB_MOCAP_LIVE_PORT, so two setups can run side by side."""
    try:
        return int(os.environ.get('RGB_MOCAP_LIVE_PORT') or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT


def to_blender(points):
    """Y-up app coordinates to Z-up Blender coordinates."""
    return np.asarray(points, dtype=np.float64) @ A.T


def matrices_to_blender(matrices):
    """Convert stored app-space (A-conjugated) 4x4 bone deltas back to Blender."""
    values = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
    return np.einsum('ij,njk,kl->nil', A4, values, A4.T)


def transform_to_blender(matrix):
    if matrix is None:
        return None
    return (A4 @ np.asarray(matrix, dtype=np.float64).reshape(4, 4) @ A4.T).tolist()


class LiveLink:
    """One-shot local server; a single Blender client is the normal case."""

    def __init__(self, host='127.0.0.1', port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self._server = None
        self._clients = []
        self._handshaken = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = None
        self._latest = None
        self._meta = {}
        self._meta_signature = ''
        self._sent = 0
        self._frame = -1
        self._published = 0
        self._delivered = 0
        self._error = ''

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.active:
            return True, 'Already listening on %s:%d' % (self.host, self.port)
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(2)
            server.settimeout(0.5)
        except OSError as error:
            return False, 'Cannot listen on %s:%d: %s' % (self.host, self.port, error)
        self._server = server
        self._stopping.clear()
        self._thread = threading.Thread(target=self._serve, name='rgbmocap-live-link', daemon=True)
        self._thread.start()
        return True, 'Listening on %s:%d' % (self.host, self.port)

    def stop(self):
        self._stopping.set()
        self._wake.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(1.5)
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients = []
            self._handshaken = {}
            self._latest = None
            self._sent = 0
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None

    @staticmethod
    def _signature(meta):
        """Describe the stream; the frame counter changes every packet."""
        return json.dumps({key: value for key, value in meta.items() if key != 'frame'},
                          sort_keys=True, default=str)

    def _store(self, packet, meta):
        with self._lock:
            self._latest = packet
            self._meta = meta
            self._meta_signature = self._signature(meta)
            self._frame = int(meta.get('frame', -1))
            self._published += 1
        self._wake.set()

    def _store_meta(self, meta):
        with self._lock:
            self._meta = meta
            self._meta_signature = self._signature(meta)
        self._wake.set()

    def publish_character(self, blend, palette, armature, matrices, frame, fps=30.0,
                          frames=None, label='', transform=None, height=None,
                          rest_format='blend'):
        """Send the displayed character, and its pose when one is available.

        ``matrices=None`` describes the character alone, so Blender can append
        its rest file and put the rig in place before the first pose arrives.
        """
        if not self.active:
            return False
        meta = dict(version=VERSION, kind='character', blend=str(blend), palette=list(palette),
                    armature=str(armature), transform=transform_to_blender(transform),
                    height=float(height) if height else None,
                    fps=float(fps), frames=int(frames) if frames else None, frame=int(frame),
                    label=str(label), rest_format=str(rest_format),
                    converter='app_to_blender')
        if matrices is None:
            meta['rest_only'] = True
            self._store_meta(meta)
            return True
        values = matrices_to_blender(matrices)
        if values.shape[0] != len(palette) or not np.isfinite(values).all():
            return False
        packet = (MAGIC_CHARACTER + struct.pack('<IfH', int(frame), float(fps), len(palette))
                  + values.astype('<f4').tobytes())
        self._store(packet, meta)
        return True

    def publish_joints(self, joints, frame, fps=30.0, frames=None, label=''):
        """Send the canonical 22-joint pose of a source person."""
        if not self.active:
            return False
        points = to_blender(joints)
        if points.shape != (len(NAMES), 3) or not np.isfinite(points).all():
            return False
        packet = MAGIC_JOINTS + struct.pack('<If', int(frame), float(fps)) + points.astype('<f4').tobytes()
        meta = dict(version=VERSION, kind='joints', names=NAMES, parents=PARENTS,
                    fps=float(fps), frames=int(frames) if frames else None, frame=int(frame),
                    label=str(label), converter='yup_to_zup')
        self._store(packet, meta)
        return True

    # Backwards-compatible name used by the original wire-format check.
    publish = publish_joints

    def status(self):
        return dict(active=self.active, clients=len(self._clients), sent=self._sent,
                    frame=self._frame, address='%s:%d' % (self.host, self.port),
                    kind=self._meta.get('kind', ''), error=self._error)

    def _serve(self):
        """Write each new frame once, and refresh the handshake when it changes.

        A blocking accept() would cap the stream at one update per timeout, so
        the server and its clients share one short select() instead.
        """
        while not self._stopping.is_set():
            with self._lock:
                clients = list(self._clients)
                signature = self._meta_signature
                meta = dict(self._meta)
            watch = [self._server] + clients
            try:
                readable, _, _ = select.select(watch, [], [], 0.005)
            except (OSError, ValueError):
                break
            for sock in readable:
                if sock is self._server:
                    try:
                        client, _address = self._server.accept()
                    except OSError:
                        continue
                    client.settimeout(1.0)
                    try:
                        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        if meta:
                            client.sendall((json.dumps(meta) + '\n').encode('utf-8'))
                    except OSError:
                        client.close()
                        continue
                    with self._lock:
                        self._clients.append(client)
                        self._handshaken[client] = signature
                        self._delivered = self._published
                    continue
                # Any inbound byte means the peer closed or misbehaves; drop it.
                try:
                    sock.recv(4096)
                except OSError:
                    pass
                with self._lock:
                    self._clients = [item for item in self._clients if item is not sock]
                    self._handshaken.pop(sock, None)
                try:
                    sock.close()
                except OSError:
                    pass
            if not meta:
                continue
            with self._lock:
                stale = [client for client in self._clients
                         if self._handshaken.get(client) != signature]
            line = (json.dumps(meta) + '\n').encode('utf-8')
            for client in stale:
                try:
                    client.sendall(line)
                except OSError:
                    with self._lock:
                        self._clients = [item for item in self._clients if item is not client]
                        self._handshaken.pop(client, None)
                    try:
                        client.close()
                    except OSError:
                        pass
                else:
                    with self._lock:
                        self._handshaken[client] = signature
            with self._lock:
                packet = self._latest
                pending = self._published != self._delivered and bool(self._clients)
                if pending:
                    self._delivered = self._published
                    self._sent += 1
                targets = list(self._clients)
            if not pending or packet is None:
                continue
            for client in targets:
                try:
                    client.sendall(packet)
                except OSError:
                    with self._lock:
                        self._clients = [item for item in self._clients if item is not client]
                        self._handshaken.pop(client, None)
                    try:
                        client.close()
                    except OSError:
                        pass
