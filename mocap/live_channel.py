"""Shared-memory channel between the GUI and the live inference worker.

One block carries everything: the newest camera frames (a small ring the GUI
writes, the worker reads), the solved body (joint positions and mannequin
transforms the GUI reads), and the worker's state. Frames and results never
cross a process boundary as Python objects - the worker runs SAM 3D Body, the
GUI stays torch-free, and both touch the same bytes.

Header (little-endian):
    0  magic (uint64)
    8  width, 16 height, 24 limit, 32 slots, 40 frame stride (uint64 each)
    48 state, 56 count, 64 sequence (uint64 each)
    72 worker fps (float64), 80 error length (uint64)
    HEADER-512 .. HEADER  error text (utf-8)
    HEADER ..             frame ring (slots * width * height * 3, uint8)
    then joints (limit * 22 * 3 * float32), then rig (limit * 52 * 4x4 * float32)

States: 0 idle, 1 worker ready, 2 running, 3 stop requested, 4 worker failed.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
from multiprocessing import shared_memory

MAGIC = 0x4C495645424F4459  # "LIVEBODY"
HEADER = 4096
ERROR_BYTES = 512

IDLE, READY, RUNNING, STOP, FAILED, CALIBRATING = 0, 1, 2, 3, 4, 5
SLOTS = 4
DEFAULT_LIMIT = 5400  # 3 minutes at 30 fps; the live clip stops growing there
MESH_RING = 600       # the follower's posed base-character frames
MESH_FLOATS = 55 * 16 + 54 * 9 + 20 + 3


def progress_file(root, key):
    """Where one worker reports its start-up step to the 3D view.

    The wait before the first pose is real (the model load when nothing was
    preloaded, then the first inference), so the view shows the worker's own
    step and the elapsed seconds rather than an unexplained delay.
    """
    return Path(root) / 'cache' / ('live-progress-%s.json' % key)


class Progress:
    """The worker's start-up step, written for the GUI to read.

    A file rather than the shared channel because a preloaded worker has no
    channel until the live source opens: it still has to say how far its load
    got while the source is coming up.
    """

    STEPS = 5  # solver, character, detector, body model, first pose

    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.path = None

    def step(self, step, detail, within=0.0):
        if self.path is None:
            return
        payload = json.dumps({'step': int(step), 'of': self.STEPS, 'detail': str(detail),
                              'within': float(within)})
        temporary = self.path.with_name(self.path.name + '.part')
        try:
            temporary.write_text(payload, encoding='utf-8')
            os.replace(temporary, self.path)
        except OSError:
            pass


class LiveChannel:
    """Created by the GUI; attached by name from the worker process."""

    def __init__(self, name, width=None, height=None, limit=DEFAULT_LIMIT, create=False):
        self._owns = bool(create)
        self._view = None
        if create:
            self.width = int(width)
            self.height = int(height)
            self.limit = int(limit)
            self.frame_stride = self.width * self.height * 3
            self.ring_bytes = SLOTS * self.frame_stride
            self.joints_offset = HEADER + self.ring_bytes
            self.joints_bytes = self.limit * 22 * 3 * 4
            self.rig_offset = self.joints_offset + self.joints_bytes
            self.rig_bytes = self.limit * 52 * 16 * 4
            # One posed base-character frame: the GUI skins it on the GPU.
            self.mesh_offset = self.rig_offset + self.rig_bytes
            self.total = self.mesh_offset + MESH_FLOATS * 4
        self._memory = shared_memory.SharedMemory(name=name, create=create,
                                                  size=self.total if create else 0)
        self._view = memoryview(self._memory.buf)[:self.total if create else None]
        if create:
            struct.pack_into('<QQQQQQQQQdQ', self._view, 0, MAGIC, self.width, self.height,
                             self.limit, SLOTS, self.frame_stride, IDLE, 0, 0, 0.0, 0)
        else:
            magic, self.width, self.height, self.limit, slots, self.frame_stride = \
                struct.unpack_from('<QQQQQQ', self._view, 0)
            if magic != MAGIC:
                self.close()
                raise ValueError('Live channel layout mismatch')
            self.ring_bytes = SLOTS * self.frame_stride
            self.joints_offset = HEADER + self.ring_bytes
            self.joints_bytes = self.limit * 22 * 3 * 4
            self.rig_offset = self.joints_offset + self.joints_bytes
            self.rig_bytes = self.limit * 52 * 16 * 4
            self.mesh_offset = self.rig_offset + self.rig_bytes
            self.total = self.mesh_offset + MESH_FLOATS * 4

    @property
    def name(self):
        return self._memory.name

    # -- state -------------------------------------------------------------
    def _field(self, offset):
        return struct.unpack_from('<Q', self._view, offset)[0]

    def state(self):
        return self._field(48)

    def count(self):
        return self._field(56)

    def sequence(self):
        return self._field(64)

    def worker_fps(self):
        return struct.unpack_from('<d', self._view, 72)[0]

    def set_state(self, state):
        struct.pack_into('<Q', self._view, 48, state)

    # -- error -------------------------------------------------------------
    def error(self):
        length = self._field(80)
        if not length:
            return ''
        return bytes(self._view[HEADER - ERROR_BYTES:HEADER - ERROR_BYTES + length]).decode('utf-8', 'replace')

    def fail(self, message):
        raw = str(message)[:ERROR_BYTES - 1].encode('utf-8')
        self._view[HEADER - ERROR_BYTES:HEADER - ERROR_BYTES + len(raw)] = raw
        struct.pack_into('<Q', self._view, 80, len(raw))
        struct.pack_into('<Q', self._view, 48, FAILED)

    # -- frames (GUI writes, worker reads) ----------------------------------
    def write_frame(self, bgr):
        """Publish one BGR frame; the reader always jumps to the newest."""
        if bgr.shape[0] != self.height or bgr.shape[1] != self.width:
            return False
        seq = self._field(64) + 1
        slot = HEADER + (seq % SLOTS) * self.frame_stride
        self._view[slot:slot + self.frame_stride] = np.ascontiguousarray(bgr, np.uint8).tobytes()
        struct.pack_into('<Q', self._view, 64, seq)
        return True

    def read_frame(self, last_seq):
        """The newest frame past ``last_seq``; returns (bgr, seq) or (None, last_seq)."""
        seq = self._field(64)
        if seq == last_seq or seq == 0:
            return None, last_seq
        slot = HEADER + (seq % SLOTS) * self.frame_stride
        data = np.frombuffer(self._view, np.uint8, self.frame_stride, slot)
        return data.reshape(self.height, self.width, 3).copy(), seq

    # -- results (worker writes, GUI reads) ---------------------------------
    def publish(self, index, joints, rig, fps=0.0):
        """One solved frame: 22 joint positions and the (52, 4, 4) rig row.

        The ring is reused rather than grown: a rehearsal can run for as long
        as the performer likes, and only the newest frame is displayed. It used
        to stop solving at ``limit`` frames (three minutes).
        """
        slot = index % self.limit
        joints = np.ascontiguousarray(joints, np.float32).reshape(22, 3)
        rig = np.ascontiguousarray(rig, np.float32).reshape(52, 4, 4)
        base = self.joints_offset + slot * 22 * 3 * 4
        self._view[base:base + 22 * 3 * 4] = joints.tobytes()
        base = self.rig_offset + slot * 52 * 16 * 4
        self._view[base:base + 52 * 16 * 4] = rig.tobytes()
        struct.pack_into('<Q', self._view, 56, index + 1)
        struct.pack_into('<d', self._view, 72, float(fps))
        return True

    def publish_box(self, box, seq):
        """The detected person's box in preview pixels: x0, y0, x1, y1."""
        struct.pack_into('<Q', self._view, 96, seq)
        struct.pack_into('<ffff', self._view, 104, *[float(v) for v in box[:4]])

    def box(self):
        """The latest box and its sequence: (x0, y0, x1, y1, seq) or None."""
        seq = self._field(96)
        if seq == 0:
            return None
        return (*struct.unpack_from('<ffff', self._view, 104), seq)

    def joints_view(self, count):
        """A (count, 22, 3) float32 view of the published joints.

        Only frames still in the ring are meaningful; the live view shows the
        newest one, so this is for short runs and verification.
        """
        count = min(int(count), self.limit)
        if count <= 0:
            return None
        data = np.frombuffer(self._view, np.float32, count * 22 * 3, self.joints_offset)
        return data.reshape(count, 22, 3)

    def joints_latest(self, count):
        """The newest solved frame as (1, 22, 3), wherever the ring is."""
        if count <= 0:
            return None
        slot = (int(count) - 1) % self.limit
        data = np.frombuffer(self._view, np.float32, 22 * 3,
                             self.joints_offset + slot * 22 * 3 * 4)
        return data.reshape(1, 22, 3)

    def rig_latest(self):
        """The newest solved frame's (52, 4, 4) rig row."""
        count = self.count()
        if count <= 0:
            return np.tile(np.eye(4), (52, 1, 1)).astype(np.float32)
        slot = (count - 1) % self.limit
        data = np.frombuffer(self._view, np.float32, 52 * 16,
                             self.rig_offset + slot * 52 * 16 * 4)
        return data.reshape(52, 4, 4)

    @property
    def rig_frames(self):
        """Index-agnostic newest rig row, for a renderer keyed by view index."""
        channel = self

        class Latest:
            def __getitem__(self, index):
                return channel.rig_latest()

            def __len__(self):
                return 1

        return Latest()

    def rig_view(self, count):
        """A (count, 52, 4, 4) float32 view of the published rig transforms."""
        if count <= 0:
            return None
        data = np.frombuffer(self._view, np.float32, count * 52 * 16, self.rig_offset)
        return data.reshape(count, 52, 4, 4)

    def publish_mesh(self, frame):
        """One posed base-character frame (the packed SMPL-X driver)."""
        base = self.mesh_offset
        self._view[base:base + MESH_FLOATS * 4] =             np.ascontiguousarray(frame, np.float32).tobytes()
        struct.pack_into('<Q', self._view, 88, self._field(88) + 1)
        return True

    def mesh_count(self):
        return self._field(88)

    def mesh_view(self):
        """The latest posed frame as a (1, MESH_FLOATS) float32 array."""
        if self.mesh_count() == 0:
            return None
        data = np.frombuffer(self._view, np.float32, MESH_FLOATS, self.mesh_offset)
        return data.reshape(1, MESH_FLOATS)

    @property
    def mesh_latest(self):
        """Index-agnostic latest frame for a renderer keyed by view index."""
        channel = self

        class Latest:
            def __getitem__(self, index):
                frame = channel.mesh_view()
                if frame is None:
                    return np.zeros(MESH_FLOATS, np.float32)
                return frame[0]

            def __len__(self):
                return max(1, channel.mesh_count())

        return Latest()

    def close(self):
        view, self._view = self._view, None
        memory, self._memory = self._memory, None
        if view is not None:
            try:
                view.release()
            except (BufferError, ValueError):
                pass
        if memory is not None:
            try:
                memory.close()
            except (BufferError, ValueError):
                pass
            if self._owns:
                try:
                    memory.unlink()
                except (BufferError, OSError):
                    pass
