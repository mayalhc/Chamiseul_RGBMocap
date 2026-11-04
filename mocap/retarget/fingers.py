"""Finger rotations for a solved take, from the source's finger frames.

The same arithmetic Blender's bake applies (blender_bake.bake, ``extra``), done
in numpy so the solved motion can be posed without Blender: each source finger
joint's rotation relative to its hand, carried onto the character's matching
bone through the one-time rest alignment of the two bones' directions.
"""
import json
from pathlib import Path
import numpy as np
from mocap.mixamo_rig import align


def finger_deltas(asset, folder, kind):
    """{palette index: (frames, 3, 3) world delta} for the character's fingers."""
    if kind not in {'mixamo', 'glb'}:
        return {}
    asset, folder = Path(asset), Path(folder)
    rig = np.load(asset / 'live-rig.npz', allow_pickle=False)
    names = [str(n) for n in rig['names']]
    head, tail = rig['head'].astype(float), rig['tail'].astype(float)
    entry = json.loads((folder / 'sources.json').read_text(encoding='utf8'))[0]
    info = entry['info']
    frames = np.load(Path(entry['folder']) / info['frames_file'], mmap_mode='r')
    source = np.load(Path(entry['folder']) / info['asset_file'])
    source_rest, ends = source['rest'], source['ends']
    base = np.load(folder / f'motion-{kind}-rotation.npz')['world']
    solved = np.load(folder / f'motion-{kind}-stance.npz')['world']
    bones = {n.split(':')[-1]: i for i, n in enumerate(names)}
    out = {}
    for j, name in enumerate(info['names'][22:], 22):
        target = bones.get(name)
        if target is None:
            continue
        hand = 20 if name.startswith('Left') else 21
        phalanx = int(name.split('Hand')[1][-1])
        vsource = source_rest[j + 1] - source_rest[j] if phalanx < 3 else ends[j]
        C = align(tail[target] - head[target], vsource)
        Ch = np.asarray(frames[0, hand, :3, :3]).T @ base[0, hand]
        hand_world = np.asarray(frames[:, hand, :3, :3], float)
        relative = hand_world.swapaxes(-1, -2) @ np.asarray(frames[:, j, :3, :3], float)
        out[target] = solved[:, hand] @ Ch.T @ relative @ C
    return out
