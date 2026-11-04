"""Measure how far the current constant-height floor is from the real stance plane.

Reads one people-scene root, samples sole geometry for every actor take, keeps the
frames the contact model marks as stance, and fits a plane to those contact points.
The fitted plane is the ground the actor actually stood on, so its distance from
the current world-Y floor is the grounding error the app currently ships.

Also reports the plane tilt against world Y: a large tilt means no single scalar
floor can ever plant both feet, which is what the reference board is meant to fix.
"""
import json
import sys
from pathlib import Path

import numpy as np


def world_up_plane(points):
    """Least-squares plane through stance points; normal is oriented toward world +Y."""
    centre = points.mean(axis=0) if len(points) < 256 else np.median(points, axis=0)
    centred = points - centre
    _, singular, basis = np.linalg.svd(centred, full_matrices=False)
    normal = basis[-1]
    if normal[1] < 0:
        normal = -normal
    offset = float(normal @ centre)
    residual = centred @ normal
    return dict(normal=normal.tolist(), offset=offset,
                rms_mm=float(np.sqrt((residual ** 2).mean()) * 1000),
                span_mm=float(np.percentile(np.abs(residual), 95) * 1000),
                singular=singular.tolist())


def main(root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mocap.correction import SoleGeometry, prepare_contacts
    from mocap.people_store import read_people_json, safe_child
    from mocap.scene_floor import inputs, read_config, current_resolution
    import smplx
    import torch
    from mocap.paths import MODELS
    from scipy.spatial.transform import Rotation

    root = Path(root).resolve()
    config = read_config(root) or {}
    resolved = current_resolution(root)
    context = inputs(root)
    model = MODELS / 'body_models/smplx/SMPLX_NEUTRAL.npz'
    body = smplx.SMPLX(str(model), gender='neutral', num_pca_comps=12,
                       flat_hand_mean=False, batch_size=1).eval()
    inverse = np.linalg.inv(np.asarray(context['reference']))
    report = dict(root=str(root), config=config,
                  resolved_floor=(resolved or {}).get('floor'),
                  actors=[])
    every = []
    for row in context['actors']:
        project = safe_child(root, row['project'])
        record = torch.load(project / 'hmr4d_results.pt', map_location='cpu',
                            weights_only=False, mmap=True)
        params = record['smpl_params_global']
        geometry = SoleGeometry(body, params['betas'][0].numpy())
        transform = inverse @ np.asarray(row['world_to_camera'])
        rotation = transform[:3, :3]
        probabilities = record['net_outputs']['static_conf_logits'][0, :, :4]
        fps = float(read_people_json(project / 'project.json')['fps'])
        length = len(params['body_pose'])
        zones = []
        for first in range(0, length, 256):
            end = min(length, first + 256)
            rv = torch.cat((params['global_orient'][first:end, None],
                            params['body_pose'][first:end].reshape(-1, 21, 3)), 1).numpy()
            matrices = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
            matrices[:, 0] = rotation @ matrices[:, 0]
            quat = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1, 22, 4)[..., [3, 0, 1, 2]]
            roots = (params['transl'][first:end].numpy() + geometry.rest[0]) @ rotation.T + transform[:3, 3]
            probes = geometry.sample(roots, quat)[0]
            active = prepare_contacts(probes, probabilities[first:end].sigmoid().numpy(), fps,
                                      preset='normal')[0]
            lowest = probes.reshape(-1, 2, 2, 3).min(2)
            zones.append(lowest[active])
        points = np.concatenate([z for z in zones if len(z)], 0) if any(len(z) for z in zones) else np.zeros((0, 3))
        if not len(points):
            report['actors'].append(dict(person=row['person'], project=row['project'], contacts=0))
            continue
        plane = world_up_plane(points)
        up = np.asarray(plane['normal'])
        tilt = float(np.degrees(np.arccos(np.clip(up[1], -1, 1))))
        floor = (resolved or {}).get('floor', 0.0)
        error_mm = (points[:, 1] - floor) * 1000
        report['actors'].append(dict(person=row['person'], project=row['project'],
                                     contacts=int(len(points)), plane=plane, tilt_deg=tilt,
                                     floor_used=floor,
                                     current_error_mean_mm=float(error_mm.mean()),
                                     current_error_abs_mm=float(np.abs(error_mm).mean()),
                                     current_error_span_mm=float(np.percentile(error_mm, 95) - np.percentile(error_mm, 5))))
        every.append(points)
    if every:
        merged = np.concatenate(every, 0)
        plane = world_up_plane(merged)
        up = np.asarray(plane['normal'])
        report['scene'] = dict(contacts=int(len(merged)), plane=plane,
                               tilt_deg=float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))),
                               floor_from_plane_at_origin=float(plane['offset']))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
