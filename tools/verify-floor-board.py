"""Verify the floor reference board against exact synthetic ground truth.

Renders the printable page into a synthetic camera with a known pose, writes a
real video, and runs the shipped detector on it. Because the pose is chosen by
us, every recovered number has an exact answer: the sheet plane, the camera
height above it and the recovered screen orientation.

Also checks the world mapping used by the app: the plane expressed in an
arbitrary world frame must put every board corner at the same height.

Run from the app directory:
  runtime/python.exe -B tools/verify-floor-board.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

WIDTH, HEIGHT = 720, 1280
FOCAL = 900.0
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], np.float64)
DPI = 300
PX_PER_MM = DPI / 25.4


def camera_pose(ahead_m, pitch_deg, yaw_deg=0.0):
    """Board->camera rotation, the sheet centre in the camera, and the true height.

    The camera sits ahead_m in front of the sheet, pitched down by pitch_deg, so
    the optical axis lands on the sheet centre. yaw_deg turns the sheet on the
    floor. Every quantity here is exact, which is what the detector is judged on.
    """
    pitch, yaw = np.radians(pitch_deg), np.radians(yaw_deg)
    right = np.array([1.0, 0.0, 0.0])
    down = np.array([0.0, -np.sin(pitch), -np.cos(pitch)])
    forward = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    world_to_camera = np.stack([right, down, forward])  # camera axes in world
    height = ahead_m * np.tan(pitch)
    distance = ahead_m / np.cos(pitch)
    centre = np.array([0.0, 0.0, distance])
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                     [np.sin(yaw), np.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])
    return world_to_camera @ turn, centre, height


def render_page(page, rotation, centre, frames=6, paper='a4'):
    """Warp the printed sheet into frames using the exact projection."""
    import cv2
    from mocap import floor_board
    image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit('Could not read the generated page: ' + str(page))
    height, width = image.shape
    origin_x, origin_y = width / 2, height / 2

    def project(x_mm, y_mm):
        point = np.array([x_mm / 1000.0, y_mm / 1000.0, 0.0])
        in_camera = rotation @ point + centre
        pixel = K @ in_camera
        return pixel[:2] / pixel[2]

    margin = 10.0
    span_x, span_y = floor_board.span_mm(paper)
    span_x = span_x / 2 + floor_board.marker_mm(paper) / 2 + 10
    span_y = span_y / 2 + floor_board.marker_mm(paper) / 2 + 10
    corners_mm = [(-span_x, -span_y), (span_x, -span_y), (span_x, span_y), (-span_x, span_y)]
    # Supersample and average down: blocky nearest-neighbour edges bias the
    # subpixel corner refinement far more than a real camera does.
    supersample = 3
    scaled = K.copy()
    scaled[0, 0] *= supersample
    scaled[1, 1] *= supersample
    scaled[0, 2] = WIDTH * supersample / 2
    scaled[1, 2] = HEIGHT * supersample / 2

    def project_big(x_mm, y_mm):
        point = np.array([x_mm / 1000.0, y_mm / 1000.0, 0.0])
        in_camera = rotation @ point + centre
        pixel = scaled @ in_camera
        return pixel[:2] / pixel[2]

    destination = np.array([project_big(x, y) for x, y in corners_mm], np.float32)
    source = np.array([[origin_x + x * PX_PER_MM, origin_y - y * PX_PER_MM] for x, y in corners_mm], np.float32)
    x0 = int(max(0, source[:, 0].min() - margin))
    x1 = int(min(width, source[:, 0].max() + margin))
    y0 = int(max(0, source[:, 1].min() - margin))
    y1 = int(min(height, source[:, 1].max() + margin))
    crop = image[y0:y1, x0:x1]
    source_local = source - np.array([x0, y0], np.float32)
    matrix = cv2.getPerspectiveTransform(source_local.astype(np.float32), destination.astype(np.float32))
    big = cv2.warpPerspective(crop, matrix, (WIDTH * supersample, HEIGHT * supersample),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    sheet = cv2.resize(big, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return [sheet.copy() for _ in range(frames)]


def write_video(path, frames):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit('OpenCV could not start a video writer')
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def world_check(measured, rotation, centre, paper='a4'):
    """The plane mapped into an arbitrary world must hold every corner level."""
    from mocap import floor_board
    world_to_camera = np.eye(4)
    # A deliberately awkward world frame: axes permuted, offset in all three.
    world_to_camera[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    world_to_camera[:3, 3] = [0.3, 0.1, -0.2]
    plane = floor_board.scene_plane(measured, world_to_camera)
    in_camera = floor_board.object_points(paper) @ rotation.T + centre
    inverse = np.linalg.inv(world_to_camera)
    in_world = in_camera @ inverse[:3, :3].T + inverse[:3, 3]
    # Every sheet corner must lie on the mapped plane, whatever the world tilt.
    normal = np.asarray(plane['normal'])
    residuals = np.abs(in_world @ normal - plane['offset']) * 1000
    heights = [floor_board.height_at(plane, point) for point in in_world]
    return dict(spread_mm=float(residuals.max()), tilt_deg=plane['tilt_deg'],
                height_mm=heights[0] * 1000)


def main():
    from mocap import floor_board
    paper = 'a4'
    report = dict(cases=[], ok=True, paper=paper, marker_mm=floor_board.marker_mm(paper),
                  min_span_px=floor_board.MIN_SPAN_PX, min_markers=floor_board.MIN_MARKERS)
    with tempfile.TemporaryDirectory(prefix='floor-board-') as temporary:
        page = floor_board.save_page(Path(temporary) / 'board.png', dpi=DPI, paper=paper)
        report['page'] = dict(path=page.name, bytes=page.stat().st_size)
        # Real use, plus two shots that must be refused rather than answered badly.
        cases = (('close_45deg_0.8m', 0.8, 45.0, 0.0, True),
                 ('close_55deg_0.8m', 0.8, 55.0, 0.0, True),
                 ('steep_70deg_0.6m_yaw-25', 0.6, 70.0, -25.0, True),
                 ('far_1.2m_62deg_too_small', 1.2, 62.0, 18.0, False),
                 ('far_1.6m_66deg_too_small', 1.6, 66.0, 0.0, False))
        for label, ahead, pitch, yaw, expect in cases:
            rotation, centre, true_height = camera_pose(ahead, pitch, yaw)
            video = Path(temporary) / (label + '.mp4')
            write_video(video, render_page(page, rotation, centre, paper=paper))
            measured = floor_board.detect(video, K, papers=(paper,))
            case = dict(label=label, expect_detection=expect,
                        detected=measured is not None,
                        truth_height_m=round(float(true_height), 4))
            if measured is not None:
                normal = np.array(measured['plane']['normal'])
                offset = float(measured['plane']['offset'])
                cosine = float(np.clip(abs(normal @ rotation[:, 2]), -1, 1))
                case.update(height_mm=round(offset * 1000, 2),
                            truth_height_mm=round(true_height * 1000, 2),
                            height_error_mm=round(abs(offset - true_height) * 1000, 2),
                            tilt_error_deg=round(float(np.degrees(np.arccos(cosine))), 3),
                            frames=len(measured['frames']), markers=measured['markers'],
                            marker_px=round(measured['marker_px'], 1),
                            span_px=round(measured['span_px'], 1),
                            reprojection_px=round(measured['reprojection_px'], 3),
                            agreement_mm=round(measured['agreement_mm'], 3))
                case['world'] = {key: round(value, 3) for key, value in
                                 world_check(measured, rotation, centre, paper).items()}
                case['ok'] = bool(expect and case['height_error_mm'] < 12.0
                                  and case['tilt_error_deg'] < 1.0
                                  and case['world']['spread_mm'] < 8.0)
            else:
                case['ok'] = not expect
            report['cases'].append(case)
            report['ok'] = report['ok'] and case['ok']
    # A frame with no sheet at all must stay undetected.
    with tempfile.TemporaryDirectory(prefix='floor-board-empty-') as temporary:
        video = Path(temporary) / 'blank.mp4'
        write_video(video, [np.full((HEIGHT, WIDTH), 255, np.uint8) for _ in range(6)])
        blank = floor_board.detect(video, K, papers=(paper,))
        report['blank_video_detected'] = blank is not None
        report['ok'] = bool(report['ok'] and blank is None)
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
