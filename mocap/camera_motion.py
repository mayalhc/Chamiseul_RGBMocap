"""Advise a camera mode from measured image motion. GUI-safe: numpy and OpenCV only.

A static-camera assumption leaves real camera motion inside the character's world
root, and the contact solver then drags planted feet across a moving floor. This
measures the source clip itself: the per-frame shift a shaky hand produces and
the net travel a pan produces, as a percentage of frame width.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .storage import atomic_json

VERSION = 1
SAMPLES = 48
WIDTH = 192
# Calibrated on synthetic clips (frozen, shaken, panned) and on the recorded
# takes in this installation: a tripod clip stays below 0.1% of frame width per
# frame, while the hand-held takes measured 0.23-0.44%.
SHAKE_PERCENT = 0.2
DRIFT_PERCENT = 6.0


def measure(video, samples=SAMPLES, width=WIDTH):
    """Return motion statistics for one clip, or None when it cannot be read."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 4:
            return None
        ok, first = capture.read()
        if not ok:
            return None
        height, source_width = first.shape[:2]
        if source_width < 2 or height < 2:
            return None
        scale = width / float(source_width)
        size = (width, max(2, int(round(height * scale))))
        window = cv2.createHanningWindow(size, cv2.CV_32F)
        step = max(1, total // samples)
        positions = list(range(0, max(1, total - 1), step))[:samples]
        shifts = []
        responses = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            previous = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
            ok, frame = capture.read()
            if not ok:
                break
            current = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(previous, current, window)
            shifts.append((dx, dy))
            responses.append(response)
    finally:
        capture.release()
    if not shifts:
        return None
    array = np.asarray(shifts, np.float64)
    magnitude = np.hypot(array[:, 0], array[:, 1])
    # Percent of frame width, so the numbers do not depend on resolution.
    per_frame = 100.0 * magnitude / size[0]
    drift = 100.0 * float(np.linalg.norm(array.sum(0))) * (total / len(array)) / size[0]
    return dict(version=VERSION, frames=total, sampled=len(array),
                shake_percent=round(float(per_frame.mean()), 4),
                shake_p95_percent=round(float(np.percentile(per_frame, 95)), 4),
                drift_percent=round(drift, 2),
                direction_changes=int(np.sum(np.abs(np.diff(np.sign(array[:, 0]))) > 0)),
                response=round(float(np.mean(responses)), 3))


def verdict(report):
    """Classify a report as 'moving' (camera motion likely), 'static' or 'unknown'."""
    if not report:
        return 'unknown'
    if report.get('shake_percent', 0) >= SHAKE_PERCENT or report.get('drift_percent', 0) >= DRIFT_PERCENT:
        return 'moving'
    return 'static'


def cached(project):
    """Saved measurement for this input file, without decoding the video."""
    from .result_artifacts import file_stat
    project = Path(project)
    source = project / 'input.mp4'
    if not source.exists():
        return None
    try:
        saved = json.loads((project / 'camera_shake.json').read_text(encoding='utf-8'))
        if saved.get('version') != VERSION or saved.get('input') != file_stat(source):
            return None
        return saved
    except (OSError, ValueError):
        return None


def report(project):
    """Cached measurement for a prepared project; computed once per input file."""
    from .result_artifacts import file_stat
    project = Path(project)
    source = project / 'input.mp4'
    if not source.exists():
        return None
    saved = cached(project)
    if saved is not None:
        return saved
    stamp = file_stat(source)
    cached_path = project / 'camera_shake.json'
    value = measure(source)
    if value is None:
        return None
    value = dict(value, input=stamp, verdict=verdict(value))
    try:
        atomic_json(cached_path, value)
    except OSError:
        pass
    return value


def advice(report_value, tool, static_selected):
    """User-facing warning for the selected mode, or None when nothing is wrong."""
    if verdict(report_value) != 'moving':
        return None
    if tool == 'gvhmr':
        return ('This clip moves. GVHMR always assumes a fixed camera, so camera motion stays in '
                'the character and planted feet can slide. Use ExMR with Moving Camera / SimpleVO '
                'or FlowHMR with Moving / Unknown - Auto, or shoot from a tripod.')
    if tool == 'flowhmr':
        if static_selected:
            return ('This clip moves. Static Camera leaves that motion in the character and planted '
                    'feet can slide. Choose Moving / Unknown - Auto.')
        return None
    if static_selected:
        return ('This clip moves (%.2f%% of the frame per frame). Static Camera leaves that motion in '
                'the character, so planted feet can slide. Choose Moving Camera / SimpleVO.'
                % report_value['shake_percent'])
    return None
