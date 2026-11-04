"""Stream the base body the app displays as a character for the Blender bridge.

The app already exports a take's displayed body surface skinned to a 52-bone
skeleton. For the live link the same numbers are enough: the rest pose becomes a
one-frame FBX Blender imports, and every live frame sends ``pose @ rest^-1`` per
bone, exactly like an imported character take. No retargeting and no proxy mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SKIN_KEYS = ('skin_vertices', 'skin_normals', 'skin_faces', 'skin_bones', 'skin_weights',
             'skin_offsets', 'skin_bind')

# The app's front view looks from -Z, and a reconstructed take faces that way.
# Blender's front view looks from -Y, which the Y-up to Z-up conversion sends
# -Z to +Y: half a turn apart. Turning the whole body and its transforms by 180
# degrees about the vertical axis makes both views show the same side, while a
# rotation (not a mirror) keeps the character's left and right intact.
YAW = np.diag([-1.0, 1.0, -1.0])


def yawed_points(points):
    return np.asarray(points, dtype=np.float64) @ YAW.T


def yawed_frames(frames):
    """Same half turn applied to 4x4 transforms, so poses still match."""
    frames = np.array(frames, dtype=np.float64, copy=True)
    frames[..., :3, :3] = YAW @ frames[..., :3, :3] @ YAW
    frames[..., :3, 3] = frames[..., :3, 3] @ YAW.T
    return frames


# Where a take's export sits, relative to a project workspace root. A take folder
# holds it directly, which is the first and normal case.
TAKE_LAYOUTS = ('exmr/people/*/person_*/takes/*/*_mixamo.json',
                'people/*/person_*/takes/*/*_mixamo.json',
                'takes/*/*_mixamo.json')


def _manifests(project):
    """Export manifests in this folder, else in the newest take below it."""
    metas = list(project.glob('*_mixamo.json'))
    if metas:
        return metas
    found = []
    for pattern in TAKE_LAYOUTS:
        found += list(project.glob(pattern))
    return found


def discovery(project):
    """Newest committed export in a take folder, with a usable body skin."""
    project = Path(project)
    if not project.is_dir():
        return None
    found = []
    for meta in _manifests(project):
        try:
            info = json.loads(meta.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        names = info.get('names')
        if not isinstance(names, list) or len(names) != 52:
            continue
        # A manifest's files sit beside it, whichever folder we started from.
        if not (meta.parent / str(info.get('frames_file', ''))).is_file():
            continue
        if not (meta.parent / str(info.get('asset_file', ''))).is_file():
            continue
        found.append((meta.stat().st_mtime, meta, info))
    if not found:
        return None
    _stamp, meta, info = max(found, key=lambda row: row[0])
    return meta, info


class BodyStream:
    """One take's body surface, skeleton and per-frame bone deltas."""

    def __init__(self, project, meta, info):
        self.project = Path(project)
        self.info = info
        self.names = list(info['names'])
        self.fps = float(info.get('fps') or 30.0)
        with np.load(self.project / info['asset_file'], allow_pickle=False) as data:
            self.parents = data['parents']
            self.rest = data['fbx_rest']
            self.rest_frames = data['fbx_rest_frames']
            self.display = data['fbx_display_frames']
            self.tip = data['fbx_tip_local']
            self.skin = ({key.replace('skin_', ''): data[key] for key in SKIN_KEYS}
                         if all(key in data for key in SKIN_KEYS) else None)
        self.frames = np.load(self.project / info['frames_file'], mmap_mode='r',
                              allow_pickle=False)
        self.rest_world = np.tile(np.eye(4), (len(self.names), 1, 1))
        self.rest_world[:, :3, :3] = self.rest_frames @ self.display
        self.rest_world[:, :3, 3] = self.rest
        self.rest_inverse = np.linalg.inv(self.rest_world)
        points = (self.skin['vertices'] if self.skin else self.rest).astype(float)
        # The app's world is Y-up metres.
        self.height = float(points[:, 1].max() - points[:, 1].min())
        # Blender's Python cannot open paths past Windows' 260-character limit, and a
        # take folder inside a long-titled project easily exceeds it. A short cache
        # path keyed by the take keeps the file reachable.
        import hashlib
        from .paths import ROOT
        key = hashlib.sha1((str(self.project.resolve()) + '|' + str(info['frames_file'])).encode('utf-8')).hexdigest()[:20]
        self.rest_path = ROOT / 'cache' / 'live-body' / (key + '.npz')

    @property
    def count(self):
        return int(self.frames.shape[0])

    def matrices(self, frame):
        """The transform that actually moves the body's surface, per bone.

        This is the cluster the app's own viewport and its FBX reader use:
        ``node_world @ bind^-1``, with the bind pivot at the joint. Sending it
        as-is means Blender's ``pose @ rest^-1`` reproduces the app surface
        exactly, rather than a rotation about a different origin.
        """
        index = max(0, min(self.count - 1, int(frame)))
        if self.skin is None:
            return np.tile(np.eye(4), (len(self.names), 1, 1))
        data = np.asarray(self.frames[index], dtype=np.float64)
        bind = np.asarray(self.skin['bind'], dtype=np.float64)
        rotation = data[:, :3, :3] @ self.display
        position = data[:, :3, 3]
        turn = rotation @ np.linalg.inv(bind[:, :3, :3])
        offset = position - np.einsum('jik,jk->ji', turn, bind[:, :3, 3])
        matrices = np.tile(np.eye(4), (len(self.names), 1, 1))
        matrices[:, :3, :3] = turn
        matrices[:, :3, 3] = offset
        return yawed_frames(matrices)

    def rest_file(self):
        """Write (once) the bind pose the bridge builds its rig from.

        A format Blender can construct directly, so no importer convention
        (bone axes, unit scale, a stray action) can shift the rest pose away
        from the matrices the stream is relative to.
        """
        target = self.rest_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_mtime >= (self.project / self.info['asset_file']).stat().st_mtime:
            # A body saved before the joint pivots were written would build a
            # surface sunk below the skeleton: rebuild it instead.
            with np.load(target, allow_pickle=False) as cached:
                if 'pivots' in cached.files and 'yaw' in cached.files:
                    return target
        if self.skin is None:
            raise ValueError('The take has no body surface to stream')
        temporary = target.with_name(target.name + '.partial.npz')
        try:
            np.savez_compressed(
                temporary, names=np.asarray(self.names), parents=self.parents,
                rest=yawed_points(self.rest),
                rest_frames=self.rest_frames, tip=yawed_points(self.tip),
                # The joint each cluster's transform turns the surface around.
                pivots=yawed_points(self.skin['bind'][:, :3, 3]),
                vertices=yawed_points(self.skin['vertices']), faces=self.skin['faces'],
                weights=self.skin['weights'], offsets=self.skin['offsets'],
                bones=self.skin['bones'], bind=yawed_frames(self.skin['bind']),
                yaw=np.int8(1), height=np.float64(self.height))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


_CACHE = {}


def for_project(project):
    """Cached body stream for a take folder, or None when it has no export.

    A workspace root works too: the newest take below it is streamed, so the
    single-person view sends the same base body as the people view.
    """
    found = discovery(project)
    if found is None:
        _CACHE.pop(str(project), None)
        return None
    meta, info = found
    key = str(meta.parent)
    stamp = (meta.stat().st_mtime_ns, str(info.get('frames_file')))
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    body = BodyStream(meta.parent, meta, info)
    _CACHE[key] = (stamp, body)
    return body
