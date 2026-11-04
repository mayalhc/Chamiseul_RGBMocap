"""Pose an imported character straight from the live rig, with no keys and no bake.

The live worker publishes, per frame, the world frames of the app's body rig
(``rig_row``: a (52, 4, 4) array whose rotations are deltas from a T-pose rest).
An imported character only needs those 22 body deltas plus its own rest bone
positions (``live-rig.npz``, written once at import): every palette bone then
gets the skin matrix ``[D | p - D h]`` where ``D`` is its joint's delta, ``h`` its
rest head and ``p`` the head after forward kinematics. That is the same matrix
the baked path produces (``pose @ rest^-1``), computed here in well under a
millisecond so the character moves with the video the moment a pose arrives.

Pure numpy; the GUI uses it without Blender or torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .mixamo_rig import BODY_PARENTS, align, fit_frame

FILE = 'live-rig.npz'
VERSION = 2
FLOOR_JOINTS = (7, 8, 10, 11)
FOOT_JOINTS = (7, 8, 10, 11)


def rig_file(asset):
    return Path(asset) / FILE


def rig_current(asset):
    """True when the asset has a live rig of the current version."""
    path = rig_file(asset)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(data['version']) >= VERSION
    except (OSError, ValueError, KeyError):
        return False


def _children():
    below = {}
    for joint, parent in enumerate(BODY_PARENTS):
        if parent >= 0:
            below.setdefault(parent, []).append(joint)
    return below


def rest_alignment(source_rest, target_rest):
    """Per body joint: the world rotation taking the target's rest bones to the
    source's rest bones.

    A character whose rest pose differs from the T-pose of the live rig (an
    A-pose, say) would otherwise copy the performer's motion on top of its own
    rest and stand with the wrong arm angle. Joints without children (head,
    hands, toes) keep their parent's alignment, as the live rig does.
    """
    source_rest = np.asarray(source_rest, dtype=np.float64)[:22]
    target_rest = np.asarray(target_rest, dtype=np.float64)[:22]
    below = _children()
    # The pelvis and chest are aimed along the trunk only: their other children
    # (hip sockets, collars) sit differently on every body and would tilt them.
    below[0] = [3]
    below[9] = [12]
    result = np.tile(np.eye(3), (22, 1, 1))
    for joint in range(22):
        if joint in FOOT_JOINTS:
            continue
        kids = below.get(joint)
        if kids:
            reference = np.stack([target_rest[k] - target_rest[joint] for k in kids])
            wanted = np.stack([source_rest[k] - source_rest[joint] for k in kids])
            if len(kids) == 1:
                result[joint] = align(reference[0], wanted[0])
            else:
                result[joint] = fit_frame(reference, wanted[None])[0]
        elif BODY_PARENTS[joint] >= 0:
            result[joint] = result[BODY_PARENTS[joint]]
    # A foot is judged by its sole, not by the ankle-to-toe line: both rests
    # stand flat, so only the facing (yaw) can differ. Aligning the lines
    # instead tilts the sole by however much the two ankle geometries differ.
    for foot, toe in ((7, 10), (8, 11)):
        a = target_rest[toe] - target_rest[foot]
        b = source_rest[toe] - source_rest[foot]
        yaw = np.arctan2(b[0], b[2]) - np.arctan2(a[0], a[2])
        c, d = np.cos(yaw), np.sin(yaw)
        turn = np.array([[c, 0.0, d], [0.0, 1.0, 0.0], [-d, 0.0, c]])
        result[foot] = result[toe] = turn
    return result


class LiveCharacter:
    """One imported character, ready to be posed from live rig rows."""

    def __init__(self, asset, source_rest):
        with np.load(rig_file(asset), allow_pickle=False) as data:
            self.names = [str(n) for n in data['names']]
            self.parents = data['parents'].astype(int)
            self.head = data['head'].astype(np.float64)
            self.joints = data['joints'].astype(int)
            self.target_rest = data['canonical_rest'].astype(np.float64)
        count = len(self.names)
        if self.head.shape != (count, 3) or self.parents.shape != (count,):
            raise ValueError('Live rig file does not match its bones')
        mapped = {int(j) for j in self.joints if j >= 0}
        if not set(range(22)) <= mapped:
            raise ValueError('Live rig is missing body joints')
        # Parents first, whatever order the palette is sorted in.
        order, seen = [], set()
        def visit(i):
            if i in seen:
                return
            seen.add(i)
            if self.parents[i] >= 0:
                visit(int(self.parents[i]))
            order.append(i)
        for i in range(count):
            visit(i)
        self.order = order
        self.hips = int(np.flatnonzero(self.joints == 0)[0])
        self.alignment = rest_alignment(source_rest, self.target_rest)
        source_rest = np.asarray(source_rest, dtype=np.float64)[:22]
        # Toes (not ankles) are what the live floor is measured from.
        source_height = source_rest[0, 1] - source_rest[list(FLOOR_JOINTS), 1].min()
        self.floor = float(self.target_rest[list(FLOOR_JOINTS), 1].min())
        target_height = self.target_rest[0, 1] - self.floor
        self.scale = float(target_height / source_height) if source_height > 1e-6 else 1.0
        self.palette = count
        self.positions = self.head.copy()

    def pose(self, row):
        """The (palette + 1, 4, 4) skin matrices for one live (52, 4, 4) row."""
        row = np.asarray(row, dtype=np.float64)
        delta = row[:22, :3, :3] @ self.alignment
        root = row[0, :3, 3] * self.scale
        root[1] += self.floor
        return self.pose_solved(delta, root)

    def pose_solved(self, delta, root, extra=None):
        """Skin matrices from 22 world deltas and the hips position.

        ``delta`` is already in this character's own terms (a retarget solve's
        output), so no rest alignment or scaling is applied here.
        """
        delta = np.asarray(delta, dtype=np.float64)
        root = np.asarray(root, dtype=np.float64)
        count = self.palette
        rotation = np.tile(np.eye(3), (count, 1, 1))
        position = self.head.copy()
        moved = np.zeros(count, bool)
        for i in self.order:
            joint = self.joints[i]
            parent = self.parents[i]
            if joint >= 0:
                rotation[i] = delta[joint]
                moved[i] = True
            elif extra is not None and i in extra:
                rotation[i] = extra[i]
                moved[i] = True
            elif parent >= 0 and moved[parent]:
                rotation[i] = rotation[parent]
                moved[i] = True
            if i == self.hips:
                position[i] = root
            elif parent >= 0 and moved[parent]:
                position[i] = position[parent] + rotation[parent] @ (self.head[i] - self.head[parent])
        self.positions = position
        out = np.tile(np.eye(4, dtype=np.float32), (count + 1, 1, 1))
        out[:count, :3, :3] = rotation
        out[:count, :3, 3] = position - np.einsum('nij,nj->ni', rotation, self.head)
        return out
