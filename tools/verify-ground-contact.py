"""A/B the released grounding against Ground Tracking on one real take.

Run from the app root with any corrected project folder:

    python tools/verify-ground-contact.py "projects/input_5495f0054d2d5d_project/gvhmr"

Both runs use the same input motion, the same contact probabilities and the
same floor plane, so only the grounding rule differs. Nothing is written into
the project. Metrics are anchored to that one plane:

* hover    - both soles 2..8 cm above the plane (the artifact a viewer sees);
* sink     - the lowest sole more than 1.5 cm below the plane;
* grounded - the lowest sole within 1.5 cm of the plane;
* clearance distortion - deviation from the source's own height above its local
  ground. Grounding should remove drift, not change how high the feet are.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mocap.ground_track as ground_track  # noqa: E402
from mocap.correction import SoleGeometry, correct_arrays  # noqa: E402
from mocap.paths import MODELS  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def motion(path, body):
    record = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    params = record['smpl_params_global']
    geometry = SoleGeometry(body, params['betas'][0].numpy())
    root = params['transl'].numpy() + geometry.rest[0]
    rotvec = torch.cat((params['global_orient'][:, None],
                        params['body_pose'].reshape(-1, 21, 3)), 1).numpy()
    quat = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(-1, 22, 4)[..., [3, 0, 1, 2]]
    return geometry, root, quat, geometry.sample(root, quat)[0]


def run(geometry, root, quat, probes, probabilities, fps, floor, preset, legacy):
    """Correct once; ``legacy`` hides the ground curve to reproduce old grounding."""
    original = ground_track.reference
    if legacy:
        ground_track.reference = lambda *a, **k: None
    try:
        out_root = np.empty_like(root)
        out_quat = np.empty_like(quat)
        _, _, report = correct_arrays(geometry.rest[:22], root, quat, probes, probabilities, fps,
                                      floor, output=(out_root, out_quat), preset=preset, mode='flat')
    finally:
        ground_track.reference = original
    return geometry.sample(out_root, out_quat)[0], report


def scores(probes, plane, source_clearance, fps):
    value = ground_track.audit(probes, fps, plane)
    lowest = np.asarray(probes)[..., 1].reshape(-1, 2, 2).min(2).min(1) - plane
    value['clearance_distortion_p95_m'] = float(np.quantile(np.abs(lowest - source_clearance), .95))
    return value


def show(label, values):
    print(label)
    for key in ('hover_pct', 'sink_pct', 'grounded_pct', 'airborne_over_8cm'):
        if key in values:
            print('  %-30s %6.1f %%' % (key, values[key]))
    for key in ('hover_p95_m', 'clearance_distortion_p95_m', 'per_second_float_range_m',
                'ground_drift_m'):
        if key in values:
            print('  %-30s %6.1f cm' % (key, 100. * values[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--source', default='footmr_initial.pt')
    parser.add_argument('--contacts', default='hmr4d_results.pt')
    parser.add_argument('--preset', default='normal', choices=('weak', 'normal', 'strong'))
    args = parser.parse_args()

    project = args.project if args.project.is_absolute() else ROOT / args.project
    fps = float(json.loads((project / 'project.json').read_text(encoding='utf-8')).get('fps', 30))
    import smplx
    body = smplx.SMPLX(str(MODELS / 'body_models/smplx/SMPLX_NEUTRAL.npz'), gender='neutral',
                       num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
    geometry, root, quat, probes = motion(project / args.source, body)
    n = len(root)
    probabilities = torch.load(project / args.contacts, map_location='cpu', weights_only=False,
                               mmap=True)['net_outputs']['static_conf_logits'][0, :, :4].sigmoid().numpy()
    if len(probabilities) != n:
        raise SystemExit('contact and motion frame counts differ')

    reference = ground_track.reference(probes, fps)
    clearance, ref = ground_track.clearance(probes, fps)
    plane = float(ref['anchor'])
    print(f'{project.name}  frames={n}  fps={fps:g}  preset={args.preset}  source={args.source}')
    print('source local ground: drift %.1f cm, anchor plane %.1f cm, '
          'lowest sole above local ground p50 %.1f p95 %.1f cm'
          % (100 * np.ptp(ref['level']), 100 * plane, 100 * np.median(clearance),
             100 * np.quantile(clearance, .95)))

    for label, legacy in (('released grounding', True), ('Ground Tracking', False)):
        probes_out, report = run(geometry, root, quat, probes, probabilities, fps, plane,
                                 args.preset, legacy)
        show(label, scores(probes_out, plane, clearance, fps))
        print('  contacts %d  lock error %.1f mm  sole penetration (non-contact) %.1f mm  '
              'locked slide %.2f mm/s  root Y offset %.1f cm'
              % (report['contacts'], 1000 * report['max_contact_goal_error_m'],
                 1000 * (report.get('noncontact_sole_penetration_max_m') or 0.),
                 1000 * (report.get('support_material_speed_after_max_mps') or 0.),
                 100 * report['max_root_y_offset_m']))
        if report.get('warning'):
            print('  warning: %s' % report['warning'])
        json.dumps(report)  # a receipt must stay JSON-serializable


if __name__ == '__main__':
    main()
