"""Printed floor reference board: a measured ground plane instead of an inferred one.

The correction floor is normally inferred from foot contacts, so it drifts with
whatever the contact model happens to believe. A sheet printed with four ArUco
markers at known millimetre positions gives the floor directly: its corners are
solved in the camera frame, the sheet plane is the ground, and the camera height
above it follows from the same solve.

A4 is only usable when the sheet is close to the lens. A full-body shot from a
camera a few metres back sees the floor at a grazing angle, which squashes the
markers to a few pixels, so A3 (about 1.41x linear) is the recommended paper for
those shots.

Nothing here imports Torch at module scope; detection runs in the GUI process on
a handful of sampled frames.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VERSION = 3
DICTIONARY = 'DICT_4X4_50'
# Ids are unique per paper size: a printed pattern carries no scale, so the
# marker ids are what tell the app which sheet it is looking at.
IDS = {'a4': (0, 1, 2, 3), 'a3': (4, 5, 6, 7), 'a2': (8, 9, 10, 11), 'a1': (12, 13, 14, 15)}
PAGE_TITLE = 'RGB MoCap floor reference board'
# A4 geometry. Marker centres sit near the sheet corners: a wide baseline is what
# makes the recovered height accurate, and large markers are what make them
# readable from a camera that stands back to film the whole body.
MARKER_MM = 70.0
SPAN_X_MM = 130.0
SPAN_Y_MM = 170.0
PAPERS = {
    'a4': dict(width=210.0, height=297.0, scale=1.0),
    'a3': dict(width=297.0, height=420.0, scale=math.sqrt(2.0)),
    'a2': dict(width=420.0, height=594.0, scale=2.0),
    'a1': dict(width=594.0, height=841.0, scale=math.sqrt(8.0)),
}
DEFAULT_PAPER = 'a4'
SAMPLE_FRAMES = 48
# Three markers give a cross-checked baseline. Accurate depth comes from the
# whole sheet's angular size, not from one marker: measured at 17.8 px markers,
# a wide sheet still recovered the floor exactly. The gate is therefore on the
# sheet's own footprint in pixels, which is what the user can control by moving
# closer or printing larger.
MIN_MARKERS = 3
MIN_SPAN_PX = 120.0
# Below roughly 16 px per marker edge the 4x4 code itself starts decoding to the
# wrong id, which produces a confident-looking but nonsense plane.
MIN_MARKER_PX = 16.0
MAX_REPROJECTION_PX = 2.0
MAX_PLANE_RMS_MM = 4.0


def paper_spec(paper=DEFAULT_PAPER):
    if paper not in PAPERS:
        raise ValueError('Unknown floor board paper: ' + str(paper))
    return dict(PAPERS[paper], key=paper)


def marker_mm(paper=DEFAULT_PAPER):
    return MARKER_MM * paper_spec(paper)['scale']


def span_mm(paper=DEFAULT_PAPER):
    spec = paper_spec(paper)
    return SPAN_X_MM * spec['scale'], SPAN_Y_MM * spec['scale']


def footprint_mm(paper=DEFAULT_PAPER):
    return max(span_mm(paper)) + marker_mm(paper)


def dictionary():
    import cv2
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))


def layout(paper=DEFAULT_PAPER):
    """Marker corners in board millimetres: +X right, +Y away from the camera.

    Corner order is OpenCV's: top-left, top-right, bottom-right, bottom-left as
    printed, so a marker held with its printed face up and read from above keeps
    a right-handed frame pointing out of the page.
    """
    half = marker_mm(paper) / 2
    span_x, span_y = span_mm(paper)
    cells = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    rows = []
    for marker, (cx, cy) in zip(IDS[paper], cells):
        x, y = cx * span_x / 2, cy * span_y / 2
        corners = [[-half, half], [half, half], [half, -half], [-half, -half]]
        rows.append(dict(id=marker, corners=[[x + dx, y + dy] for dx, dy in corners]))
    return rows


def object_points(paper=DEFAULT_PAPER):
    return np.array([[c[0], c[1], 0.0] for row in layout(paper) for c in row['corners']],
                    np.float64) / 1000.0


def save_page(path, dpi=300, paper=DEFAULT_PAPER):
    """Write a 1:1 printable PNG. The sheet must be printed without scaling."""
    import cv2
    spec = paper_spec(paper)
    px = dpi / 25.4
    width, height = round(spec['width'] * px), round(spec['height'] * px)
    page = np.full((height, width), 255, np.uint8)
    origin_x, origin_y = width / 2, height / 2
    board = dictionary()
    side = round(marker_mm(paper) * px)
    for row in layout(paper):
        marker = cv2.aruco.generateImageMarker(board, row['id'], side, borderBits=1)
        x_mm = sum(corner[0] for corner in row['corners']) / 4
        y_mm = sum(corner[1] for corner in row['corners']) / 4
        x0 = round(origin_x + x_mm * px - side / 2)
        y0 = round(origin_y - y_mm * px - side / 2)
        page[y0:y0 + side, x0:x0 + side] = marker
    font = cv2.FONT_HERSHEY_SIMPLEX
    span_x, span_y = span_mm(paper)
    text = [
        (PAGE_TITLE + f' ({paper.upper()})', 9.0, 6.0),
        ('Print at 100% / actual size. Do not fit to page.', 5.5, 18.0),
        ('Lay it flat on the floor, face up, and film it once.', 5.5, 26.0),
        (f'Marker {marker_mm(paper):.0f} mm square; span {span_x:.0f} x {span_y:.0f} mm.', 5.5, 34.0),
    ]
    # Cap heights are given in millimetres so the sheet stays readable when it is
    # printed at A2 or A1, where a fixed pixel size would vanish.
    for body, height_mm, from_top in text:
        cap = max(1, cv2.getTextSize(body, font, 1.0, 2)[0][1])
        scale = height_mm * px / cap
        cv2.putText(page, body, (round(14 * px), round((from_top + height_mm) * px)), font, scale, 0, 2, cv2.LINE_AA)
    # 100 mm ruler so a shrunken print is obvious before it is used.
    ruler_y = round(height - 16 * px)
    x0 = round(origin_x - 50 * px)
    cv2.line(page, (x0, ruler_y), (x0 + round(100 * px), ruler_y), 0, 3)
    for tick in range(0, 101, 10):
        x = x0 + round(tick * px)
        cv2.line(page, (x, ruler_y - 6), (x, ruler_y + 6), 0, 2)
    caption = 'Ruler must measure exactly 100 mm when printed.'
    cap = max(1, cv2.getTextSize(caption, font, 1.0, 2)[0][1])
    cv2.putText(page, caption, (round(14 * px), round(height - 6 * px)), font,
                5.5 * px / cap, 0, 2, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), page):
        raise OSError('Could not write the floor board page: ' + str(path))
    return path


def _detector():
    import cv2
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary(), parameters)


def _frame_plane(corners, ids, intrinsic, paper=DEFAULT_PAPER):
    """One frame: solve the sheet pose and return its plane in camera coordinates."""
    import cv2
    keep = [i for i, marker in enumerate(ids.ravel()) if int(marker) in IDS[paper]]
    if len(keep) < MIN_MARKERS:
        return None
    per_marker = {}
    for index in keep:
        per_marker[int(ids.ravel()[index])] = np.asarray(corners[index], np.float64).reshape(4, 2)
    layout_rows = layout(paper)
    order = [row['id'] for row in layout_rows if row['id'] in per_marker]
    image_points = np.concatenate([per_marker[marker] for marker in order], 0)
    if len(image_points) < 4:
        return None
    blocks = [np.array([[c[0], c[1], 0.0] for c in next(r for r in layout_rows if r['id'] == marker)['corners']])
              for marker in order]
    object_in_board = np.concatenate(blocks, 0) / 1000.0
    flags = cv2.SOLVEPNP_IPPE if len(image_points) == 4 else cv2.SOLVEPNP_SQPNP
    ok, rotation, translation = cv2.solvePnP(object_in_board, image_points, intrinsic, None, flags=flags)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_in_board, rotation, translation, intrinsic, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    reprojection = float(np.sqrt((residual ** 2).mean()))
    if reprojection > MAX_REPROJECTION_PX:
        return None
    # Small, foreshortened markers are what make a plausible-looking solve wrong.
    marker_px = min(float(np.linalg.norm(np.roll(quad, -1, 0) - quad, axis=1).min())
                    for quad in per_marker.values())
    spread = image_points[:, None, :] - image_points[None, :, :]
    span_px = float(np.sqrt((spread ** 2).sum(-1)).max())
    matrix, _ = cv2.Rodrigues(rotation)
    in_camera = object_in_board @ matrix.T + translation.reshape(1, 3)
    centre = in_camera.mean(0)
    _, _, basis = np.linalg.svd(in_camera - centre, full_matrices=False)
    normal = basis[-1]
    offset = float(normal @ centre)
    if offset < 0:
        # The printed face was read from below: the camera must stay above it.
        normal, offset = -normal, -offset
    if not np.isfinite(normal).all() or not math.isfinite(offset) or offset <= 0:
        return None
    rms = float(np.sqrt(((in_camera - centre) @ normal).__pow__(2).mean()) * 1000)
    if rms > MAX_PLANE_RMS_MM:
        return None
    # Longest distance across the printed corner set, in the solved metric frame.
    # Compared with the same distance in the layout it exposes a print that was
    # scaled by the printer, which silently scales the whole measured floor.
    corners_mm = float(np.linalg.norm(in_camera[:, None] - in_camera[None], axis=-1).max() * 1000)
    truth_mm = float(np.linalg.norm(object_in_board[:, None] - object_in_board[None], axis=-1).max() * 1000)
    return dict(normal=normal.tolist(), offset=offset, centre=centre.tolist(),
                reprojection_px=reprojection, rms_mm=rms, markers=order,
                marker_px=marker_px, span_px=span_px, corners=in_camera.tolist(),
                corners_mm=corners_mm, corners_truth_mm=truth_mm,
                print_scale=corners_mm / truth_mm)


def _observations(video, intrinsic, samples, verbose):
    """One pass over sampled frames: marker corners and ids, nothing interpreted."""
    import cv2
    detector = _detector()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError('Could not open the video for floor board detection')
    rows = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // max(1, samples))
        failures = 0
        for frame in range(0, max(1, total), stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                failures += 1
                if failures > 8:
                    break
                continue
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if ids is None:
                continue
            rows.append((frame, [np.asarray(corner, np.float64).reshape(4, 2) for corner in corners], ids))
    finally:
        capture.release()
    return rows


def detect(video, intrinsic, samples=SAMPLE_FRAMES, papers=None, verbose=None):
    """Find the sheet at any supported print size and return its measured plane.

    The sheet is scanned once; the printed size is identified afterwards by which
    layout the corner geometry actually satisfies. A user prints whichever size
    their shot needs without telling the app which one it is.
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError('Video not found for floor board detection: ' + str(video))
    intrinsic = np.asarray(intrinsic, np.float64)
    if intrinsic.shape != (3, 3) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics for floor board detection')
    observations = _observations(video, intrinsic, samples, verbose)
    if not observations:
        return None
    for paper in (papers or tuple(PAPERS)):
        seen = []
        for frame, corners, ids in observations:
            plane = _frame_plane(corners, ids, intrinsic, paper)
            if (plane is None or len(plane['markers']) < MIN_MARKERS
                    or plane['marker_px'] < MIN_MARKER_PX or plane['span_px'] < MIN_SPAN_PX):
                continue
            plane['frame'] = frame
            seen.append(plane)
            if verbose:
                verbose('Floor board ({0}): {1} marker(s) at frame {2}, {3:.2f} px, {4:.1f} mm plane'
                        .format(paper, len(plane['markers']), frame, plane['reprojection_px'], plane['rms_mm']))
        if len(seen) < 2:
            continue
        normals = np.array([row['normal'] for row in seen])
        offsets = np.array([row['offset'] for row in seen])
        normal = np.median(normals, 0)
        normal /= np.linalg.norm(normal)
        offset = float(np.median(offsets))
        # Where the sheet lies in the frame. The floor is sampled here (or under
        # the actor), never at the world origin: the measured plane is slightly
        # tilted by the camera's own roll, so its height changes with position.
        centre = np.median(np.array([row['centre'] for row in seen]), 0)
        centre = centre - normal * (float(normal @ centre) - offset)
        spread = float(np.percentile(
            np.linalg.norm(normals * offsets[:, None] - normal * offset, axis=1), 95) * 1000)
        span_x, span_y = span_mm(paper)
        return dict(version=VERSION, plane=dict(normal=normal.tolist(), offset=offset),
                    centre=centre.tolist(),
                    frames=[row['frame'] for row in seen],
                    markers=max(len(row['markers']) for row in seen),
                    reprojection_px=float(np.median([row['reprojection_px'] for row in seen])),
                    rms_mm=float(np.median([row['rms_mm'] for row in seen])),
                    marker_px=float(np.median([row['marker_px'] for row in seen])),
                    span_px=float(np.median([row['span_px'] for row in seen])),
                    corners_mm=float(np.median([row['corners_mm'] for row in seen])),
                    corners_truth_mm=float(seen[0]['corners_truth_mm']),
                    print_scale=float(np.median([row['print_scale'] for row in seen])),
                    agreement_mm=spread, camera_height=offset, paper=paper,
                    marker_mm=marker_mm(paper), span_mm=[span_x, span_y], dictionary=DICTIONARY)
    return None


def scene_plane(record, world_to_camera):
    """Express a camera-frame plane in the world that world_to_camera maps from."""
    transform = np.asarray(world_to_camera, np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    normal_camera = np.asarray(record['plane']['normal'], np.float64)
    offset_camera = float(record['plane']['offset'])
    normal = rotation.T @ normal_camera
    offset = offset_camera - float(normal_camera @ translation)
    if normal[1] < 0:
        normal, offset = -normal, -offset
    if abs(float(normal[1])) < 0.2:
        raise ValueError('Floor board plane is not close to horizontal in this world frame')
    return dict(normal=normal.tolist(), offset=offset,
                tilt_deg=float(np.degrees(np.arccos(np.clip(abs(normal[1]) / np.linalg.norm(normal), -1, 1)))),
                camera_height=float(abs(offset_camera)))


def height_at(plane, point=None):
    """Floor height of a plane at a horizontal position (world origin by default)."""
    normal = np.asarray(plane['normal'], np.float64)
    offset = float(plane['offset'])
    point = np.zeros(3) if point is None else np.asarray(point, np.float64)
    return float((offset - normal[0] * point[0] - normal[2] * point[2]) / normal[1])


def board_center(record, world_to_camera):
    """Where the sheet itself sits in the world, used as the floor anchor."""
    transform = np.asarray(world_to_camera, np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    normal = np.asarray(record['plane']['normal'], np.float64)
    offset = float(record['plane']['offset'])
    centre_in_camera = record.get('centre')
    if centre_in_camera is None:
        # Older cache without the sheet centre: fall back to the closest point.
        centre_in_camera = normal * offset
    return rotation.T @ (np.asarray(centre_in_camera, np.float64) - translation)


def _stamp(path):
    state = Path(path).stat()
    return [state.st_size, state.st_mtime_ns]


def camera_for(project):
    """Intrinsics and world transform the reconstruction already uses."""
    project = Path(project)
    camera = project / 'scene_camera.json'
    if camera.exists():
        record = json.loads(camera.read_text(encoding='utf-8'))
        return np.asarray(record['K'], np.float64), record['world_to_camera'], 'scene_camera.json'
    tracking = project / 'tracking.npz'
    if tracking.exists():
        # A lone take has no shared scene camera yet: derive the same pairing the
        # people pipeline would, so the measured plane lands in this motion world.
        from .scene_camera import take_camera
        transform, intrinsic = take_camera(project)
        return np.asarray(intrinsic, np.float64), np.asarray(transform).tolist(), 'take_camera'
    raise FileNotFoundError('Camera intrinsics are required for floor board detection: ' + str(project))


def detect_for(project, verbose=None):
    """Detect once per take and cache the camera-frame plane next to the motion."""
    project = Path(project)
    intrinsic, transform, source = camera_for(project)
    video = project / 'input.mp4'
    cache = project / 'floor_board.json'
    stamp = _stamp(video)
    if cache.exists():
        try:
            record = json.loads(cache.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            record = None
        if (record and record.get('version') == VERSION and record.get('source') == stamp
                and record.get('intrinsic') == np.asarray(intrinsic).tolist()
                and record.get('camera_source') == source):
            return record
    measured = detect(video, intrinsic, verbose=verbose)
    if measured is None:
        return None
    record = dict(measured, source=stamp, intrinsic=np.asarray(intrinsic).tolist(),
                  camera_source=source, video=video.name)
    from .storage import atomic_json
    atomic_json(cache, record)
    return record


def _measured(record, transform, point):
    """Sample the measured plane where grounding actually happens.

    A sheet lying on a real floor is rarely exactly perpendicular to the world
    up axis the app uses (the camera can be rolled), so the plane's height drifts
    by a couple of centimetres per metre away from the sheet. The correction
    floor is a single height, therefore it is read under the actor, not at the
    world origin: `tilt_mm` reports how much that choice moved the value.
    """
    plane = scene_plane(record, transform)
    centre = board_center(record, transform)
    at = centre if point is None else np.asarray(point, np.float64)
    value = height_at(plane, at)
    return value, dict(plane=plane, board_center=centre.tolist(),
                       anchor=[float(at[0]), float(at[1]), float(at[2])],
                       tilt_mm=float((value - height_at(plane, centre)) * 1000),
                       camera_height=plane['camera_height'], frames=record['frames'],
                       agreement_mm=record['agreement_mm'], marker_px=record['marker_px'],
                       span_px=record['span_px'], paper=record['paper'])


def scene_floor(root, context, verbose=None, anchor=None):
    """Shared people-scene floor from the board seen by the group's reference camera."""
    from .people_store import read_people_json
    root = Path(root)
    name = read_people_json(root / 'people_scene.json')['anchor']
    record = detect_for(root / name, verbose=verbose)
    if record is None:
        return None, None
    reference = np.asarray(context['reference'], np.float64)
    value, detail = _measured(record, reference, anchor)
    return value, dict(detail, anchor_take=name)


def floor_for(project, anchor=None, verbose=None):
    """Measured floor height in this take's own world, or None without a board."""
    project = Path(project)
    record = detect_for(project, verbose=verbose)
    if record is None:
        return None, None
    _, transform, _ = camera_for(project)
    return _measured(record, transform, anchor)
