"""Run the floor reference board through the app on a real take.

The take's own footage has no sheet, so one is composited in at a known plane:
the stance plane fitted through that take's foot contacts, which is the best
independent estimate of the floor the actor actually stood on. The app then has
to recover that plane from the video and ground the corrected motion on it.

This measures three separate things:
  1. how far the current contact-inferred floor is from the fitted stance plane,
  2. whether the measured board plane matches what was injected (detector and
     world mapping end to end, not just the synthetic check),
  3. whether correction then grounds the soles on the measured plane.

The injected plane is itself a contact fit, so the grounding column is a
consistency check, not independent evidence of real-world accuracy.

Run from the app directory:
  runtime/python.exe -B tools/verify-floor-board-e2e.py [take_project] [result_kind]
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
BASE = APP.parent
sys.path.insert(0, str(APP))

BOARD = APP / 'build/floor-board-page.png'
TARGET = APP / 'projects/input_5495f0054d2d5d_project/gvhmr'
KIND = 'corrected_footmr_initial'


def fit_stance_plane(project, fps, limit=4096):
    """Plane through the frames the contact model calls stance, in motion world."""
    import torch
    import smplx
    from scipy.spatial.transform import Rotation
    from mocap.correction import SoleGeometry, prepare_contacts
    from mocap.paths import MODELS
    project = Path(project)
    record = torch.load(project / 'hmr4d_results.pt', map_location='cpu', weights_only=False, mmap=True)
    params = record['smpl_params_global']
    body = smplx.SMPLX(str(MODELS / 'body_models/smplx/SMPLX_NEUTRAL.npz'), gender='neutral',
                       num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
    geometry = SoleGeometry(body, params['betas'][0].numpy())
    probabilities = record['net_outputs']['static_conf_logits'][0, :, :4]
    from mocap.scene_floor import sample_windows
    length = len(params['body_pose'])
    points = []
    # Contacts need a contiguous window: a single frame cannot survive the
    # minimum-stance filter.
    for start, end in sample_windows(length, limit):
        rv = torch.cat((params['global_orient'][start:end, None],
                        params['body_pose'][start:end].reshape(-1, 21, 3)), 1).numpy()
        matrices = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
        quat = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1, 22, 4)[..., [3, 0, 1, 2]]
        roots = params['transl'][start:end].numpy() + geometry.rest[0]
        probes = geometry.sample(roots, quat)[0]
        active = prepare_contacts(probes, probabilities[start:end].sigmoid().numpy(), fps, preset='normal')[0]
        lowest = probes.reshape(-1, 2, 2, 3).min(2)
        if active.any():
            points.append(lowest[active])
    if not points:
        raise SystemExit('No stance frames found in this take')
    points = np.concatenate(points, 0)
    centre = np.median(points, 0)
    _, _, basis = np.linalg.svd(points - centre, full_matrices=False)
    normal = basis[-1]
    if normal[1] < 0:
        normal = -normal
    return dict(normal=normal, offset=float(normal @ centre), centre=centre,
                contacts=int(len(points)),
                rms_mm=float(np.sqrt(((points - centre) @ normal).__pow__(2).mean()) * 1000))


def board_placement(project, plane, size):
    """Where the sheet sits on the floor, and how this take's camera sees it.

    The sheet goes on the nearest floor point the camera can actually see, which
    is what a user can do in a fixed wide shot, and it is composed from the same
    camera transform the app uses for correction.
    """
    from mocap.floor_board import camera_for
    project = Path(project)
    width, height = int(size['width']), int(size['height'])
    intrinsic, transform, _ = camera_for(project)
    transform = np.asarray(transform)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    normal = plane['normal']
    ray = np.linalg.inv(intrinsic) @ np.array([width / 2, height - 140.0, 1.0])
    origin = -rotation.T @ translation
    direction = rotation.T @ ray
    reach = (plane['offset'] - normal @ origin) / (normal @ direction)
    centre = origin + reach * direction
    along = np.array([1.0, 0.0, 0.0])
    along = along - normal * (along @ normal)
    if np.linalg.norm(along) < 1e-6:
        along = np.array([0.0, 0.0, 1.0])
        along = along - normal * (along @ normal)
    along /= np.linalg.norm(along)
    across = np.cross(normal, along)
    return dict(intrinsic=intrinsic, rotation=rotation, translation=translation,
                centre=centre, along=along, across=across, reach=float(reach),
                width=width, height=height)


def projected_quad(placement, paper):
    """Projected sheet footprint in video pixels, with its corners in millimetres."""
    from mocap import floor_board
    span_x, span_y = floor_board.span_mm(paper)
    half_x = span_x / 2 + floor_board.marker_mm(paper) / 2 + 10
    half_y = span_y / 2 + floor_board.marker_mm(paper) / 2 + 10
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    pixels, depths = [], []
    for x, y in corners:
        point = placement['centre'] + placement['along'] * (x / 1000.0) + placement['across'] * (y / 1000.0)
        in_camera = placement['rotation'] @ point + placement['translation']
        pixel = placement['intrinsic'] @ in_camera
        pixels.append(pixel[:2] / pixel[2])
        depths.append(float(in_camera[2]))
    return np.array(pixels, np.float32), corners, np.array(depths)


def projected_marker(placement, paper):
    """Shortest projected edge of one marker: the code has to decode at this size."""
    from mocap import floor_board
    corners = floor_board.layout(paper)[0]['corners']
    pixels = []
    for x, y in corners:
        point = placement['centre'] + placement['along'] * (x / 1000.0) + placement['across'] * (y / 1000.0)
        in_camera = placement['rotation'] @ point + placement['translation']
        pixel = placement['intrinsic'] @ in_camera
        pixels.append(pixel[:2] / pixel[2])
    pixels = np.array(pixels)
    return float(np.linalg.norm(np.roll(pixels, -1, 0) - pixels, axis=1).min())


def recommend_paper(placement):
    """Smallest supported print that would be readable from this camera."""
    from mocap import floor_board
    best = dict(span_px=0.0, marker_px=0.0)
    for paper in floor_board.PAPERS:
        quad, _, depths = projected_quad(placement, paper)
        span = float(np.linalg.norm(quad[1] - quad[0]))
        marker = projected_marker(placement, paper)
        best = dict(span_px=max(best['span_px'], span), marker_px=max(best['marker_px'], marker))
        if (depths.min() > 0.05 and span >= floor_board.MIN_SPAN_PX
                and marker >= floor_board.MIN_MARKER_PX):
            return paper, dict(span_px=span, marker_px=marker)
    return None, best


def compose_board(placement, paper='a1'):
    """Draw the sheet flat on the given plane into the take's own footage."""
    import cv2
    from mocap import floor_board
    board_path = BOARD.with_name(f'floor-board-page-{paper}.png')
    if not board_path.is_file():
        floor_board.save_page(board_path, paper=paper)
    page = cv2.imread(str(board_path), cv2.IMREAD_GRAYSCALE)
    page_height, page_width = page.shape
    origin_x, origin_y = page_width / 2, page_height / 2
    destination_quad, corners, depths = projected_quad(placement, paper)
    if depths.min() <= 0.05:
        raise SystemExit('The sheet would be behind the camera; pick another take')
    centre = placement['centre']
    reach = placement['reach']
    # Shrink the page to roughly twice the footprint it will occupy before the
    # warp. Warping a 7000 px sheet straight onto ~200 px aliases the marker
    # modules away and nothing decodes.
    biggest = max(float(np.linalg.norm(destination_quad[1] - destination_quad[0])),
                  float(np.linalg.norm(destination_quad[2] - destination_quad[1])), 64.0)
    factor = min(1.0, (2.2 * biggest) / max(page_width, page_height))
    if factor < 1.0:
        page = cv2.resize(page, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        page_height, page_width = page.shape
        origin_x, origin_y = page_width / 2, page_height / 2
    per_mm = floor_board_px() * factor
    source_quad = np.array([[origin_x + x * per_mm, origin_y - y * per_mm]
                            for x, y in corners], np.float32)
    matrix = cv2.getPerspectiveTransform(source_quad.astype(np.float32), destination_quad.astype(np.float32))
    sheet = cv2.warpPerspective(page, matrix, (int(placement['width']), int(placement['height'])),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return sheet, dict(centre=centre.round(4).tolist(), reach_m=round(float(reach), 3),
                       paper=paper,
                       pixel_span=round(float(np.linalg.norm(destination_quad[1] - destination_quad[0])), 1))


def floor_board_px(dpi=300):
    return dpi / 25.4


def main():
    project = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TARGET
    kind = sys.argv[2] if len(sys.argv) > 2 else KIND
    paper = sys.argv[3] if len(sys.argv) > 3 else None
    report = dict(project=str(project), kind=kind)
    from mocap.motion_tools import RESULTS, base_result, result_sources
    from mocap import floor_board
    import cv2
    import torch

    manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    fps = float(manifest.get('fps', 30))
    baseline_path = project / f'{kind}_correction.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else {}
    plane = fit_stance_plane(project, fps)
    report['stance_plane'] = dict(normal=plane['normal'].round(4).tolist(),
                                  offset=round(plane['offset'], 5),
                                  contacts=plane['contacts'], rms_mm=round(plane['rms_mm'], 2))
    # Grounding has to be judged where the actor stands, not at the world origin:
    # the fitted stance plane is slightly tilted, so its height moves with position.
    stats = torch.load(project / 'hmr4d_results.pt', map_location='cpu', weights_only=False, mmap=True)
    anchor = np.median(stats['smpl_params_global']['transl'].numpy(), 0)
    truth_at_anchor = floor_board.height_at(plane, anchor)
    report['actor_anchor'] = [round(float(value), 4) for value in anchor]
    report['truth_at_actor_mm'] = round(truth_at_anchor * 1000, 2)
    report['baseline'] = dict(floor=baseline.get('floor'),
                              error_at_actor_mm=None if baseline.get('floor') is None else
                              round(abs(float(baseline['floor']) - truth_at_anchor) * 1000, 2),
                              contact_sole_height_after_m=baseline.get('contact_sole_height_after_m'),
                              sole_below_floor_after_m=baseline.get('sole_below_floor_after_m'))

    work = APP / 'build/floor-board-e2e'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    keep = ['project.json', 'tracking.npz', 'hmr4d_results.pt', 'opt_results.pt', 'scene_camera.json']
    for name in list(keep):
        if not (project / name).exists():
            keep.remove(name)
    for source in result_sources(kind):
        filename = RESULTS[base_result(source)][0]
        if (project / filename).exists():
            keep.append(filename)
    for name in keep:
        shutil.copy2(project / name, work / name)
    state = json.loads((work / 'project.json').read_text(encoding='utf-8'))
    state['correction'] = {**state.get('correction', {}), 'auto_floor': True, 'floor_board': True}
    (work / 'project.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

    # Rotation must be exact beyond the sheet: the composited frames keep the
    # take's own resolution and frame rate.
    capture = cv2.VideoCapture(str(project / 'input.mp4'))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    placement = board_placement(project, plane, dict(width=width, height=height))
    recommended, sizes = recommend_paper(placement)
    report['recommended'] = dict(paper=recommended,
                                 span_px=round(sizes['span_px'], 1),
                                 marker_px=round(sizes['marker_px'], 1),
                                 gate=dict(min_span_px=floor_board.MIN_SPAN_PX,
                                           min_marker_px=floor_board.MIN_MARKER_PX))
    if paper is None:
        paper = recommended
    if paper is None:
        raise SystemExit('No supported print size reads at this camera distance')
    report['paper'] = paper
    sheet, placement_detail = compose_board(placement, paper)
    report['sheet_placement'] = placement_detail
    writer = cv2.VideoWriter(str(work / 'input.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit('OpenCV could not write the composited video')
    drawn = 0
    show_until = max(1, int(total * 0.25))
    try:
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                break
            if index < show_until:
                frame = np.minimum(frame, cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR))
                drawn += 1
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    report['composited_frames'] = drawn

    from mocap import floor_board
    measured = floor_board.detect_for(work)
    report['detected'] = measured is not None
    if measured is not None:
        from mocap.scene_camera import take_camera
        transform, _ = take_camera(project)
        world = floor_board.scene_plane(measured, transform)
        normal = np.array(world['normal'])
        board_height = floor_board.height_at(world, anchor)
        report['detected_plane'] = dict(
            height_at_actor_mm=round(board_height * 1000, 2),
            truth_height_mm=round(truth_at_anchor * 1000, 2),
            height_error_mm=round(abs(board_height - truth_at_anchor) * 1000, 2),
            tilt_error_deg=round(float(np.degrees(np.arccos(np.clip(abs(normal @ plane['normal']), -1, 1)))), 3),
            marker_px=round(measured['marker_px'], 1), span_px=round(measured['span_px'], 1),
            paper=measured['paper'], agreement_mm=round(measured['agreement_mm'], 2),
            frames=len(measured['frames']))

    from mocap.correction import run
    events = []
    run(work, lambda event, **values: events.append(event), kind)
    after_path = work / f'{kind}_correction.json'
    report['correction_ran'] = after_path.exists()
    if after_path.exists():
        after = json.loads(after_path.read_text(encoding='utf-8'))
        report['after'] = dict(floor=after.get('floor'), board_present='floor_board' in after,
                               error_at_actor_mm=None if after.get('floor') is None else
                               round(abs(float(after['floor']) - truth_at_anchor) * 1000, 2),
                               measured_anchor=(after.get('floor_board') or {}).get('anchor'),
                               measured_tilt_mm=(after.get('floor_board') or {}).get('tilt_mm'),
                               warning=after.get('warning'),
                               contact_sole_height_after_m=after.get('contact_sole_height_after_m'),
                               sole_below_floor_after_m=after.get('sole_below_floor_after_m'),
                               contact_sole_penetration_over_1cm=after.get('contact_sole_penetration_over_1cm'))
        report['floor_matches_measurement'] = bool(
            after.get('floor') is not None and abs(after['floor'] - truth_at_anchor) < 0.02)
        report['comparison'] = dict(
            auto_error_mm=report['baseline'].get('error_at_actor_mm'),
            board_error_mm=report['after'].get('error_at_actor_mm'),
            auto_sole_height_mm=None if baseline.get('contact_sole_height_after_m') is None else
            round(float(baseline['contact_sole_height_after_m']) * 1000, 2),
            board_sole_height_mm=None if after.get('contact_sole_height_after_m') is None else
            round(float(after['contact_sole_height_after_m']) * 1000, 2))
    report['ok'] = bool(report['detected'] and report.get('correction_ran')
                        and report.get('floor_matches_measurement')
                        and report.get('detected_plane', {}).get('height_error_mm', 1e9) < 20)
    text = json.dumps(report, indent=2)
    (BASE / 'screenshots/floor-board-e2e.json').write_text(text, encoding='utf-8')
    print(text)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
