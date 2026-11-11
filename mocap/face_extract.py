"""ARKit face values for a reconstructed take, from its own video.

Runs in the optional face runtime (MediaPipe Face Landmarker). A take folder
holds the clip it was reconstructed from (input.mp4) and the person's box on
every frame (tracking.npz). The face in a full-body clip is far too small for
the landmarker's own detector, so the head is cut from the top of the person
box, enlarged and tracked; that found the face in 97% of a 1080x1920 dance clip
where the whole frame found none.

Writes face.npz next to tracking.npz: ``values`` (frames x 52, float16, ARKit
order), ``valid`` (frames, bool) and ``names``. Frames without a face hold the
nearest valid frame's values so playback does not snap to neutral.
"""
import json
import sys
from pathlib import Path

import numpy as np

ARKIT = (
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft',
    'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight',
    'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward',
    'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
    'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 'browDownRight',
    'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
)
FILE = 'face.npz'
CROP = 512


def emit(kind, **value):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **value)), flush=True)


def model_path(root):
    lock = json.loads((root / 'installer' / 'face-runtime-lock.json').read_text(encoding='utf-8'))
    path = root / lock['model']['path']
    if not path.is_file():
        raise RuntimeError('The face model is missing. Run Setup_Face.cmd.')
    return path


def head_box(box, width, height):
    """Square around the head: the top of the person box, a third of its height."""
    x0, y0, x1, y1 = (float(v) for v in box)
    tall = max(y1 - y0, 1.0)
    side = max(tall * 0.32, 64.0)
    cx, cy = (x0 + x1) / 2, y0 + tall * 0.09
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return a, b, int(min(width, a + side)), int(min(height, b + side))


def extract(take, root, progress=lambda done, total: None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    take = Path(take)
    video = take / 'input.mp4'
    boxes_file = take / 'tracking.npz'
    if not video.is_file() or not boxes_file.is_file():
        raise RuntimeError('This take has no input.mp4 and tracking.npz to read a face from.')
    with np.load(boxes_file, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], float)
        observed = np.asarray(data['observed'], bool) if 'observed' in data else np.ones(len(boxes), bool)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(root))),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or len(boxes)
    values = np.zeros((total, len(ARKIT)), np.float32)
    valid = np.zeros(total, bool)
    index = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while index < total:
            ok, frame = capture.read()
            if not ok:
                break
            row = min(index, len(boxes) - 1)
            if row >= 0 and observed[row] and np.isfinite(boxes[row]).all():
                height, width = frame.shape[:2]
                a, b, c, d = head_box(boxes[row], width, height)
                crop = frame[b:d, a:c]
                if crop.size:
                    big = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
                    result = landmarker.detect_for_video(image, int(index * 1000 / fps))
                    if result.face_blendshapes:
                        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
                        values[index] = [scores.get(name, 0.0) for name in ARKIT]
                        valid[index] = True
            index += 1
            if index % 15 == 0 or index == total:
                progress(index, total)
    capture.release()
    values, valid = values[:index], valid[:index]
    _hold(values, valid)
    out = take / FILE
    partial = take / (FILE + '.partial.npz')
    np.savez_compressed(partial, values=values.astype(np.float16), valid=valid,
                        names=np.array(ARKIT), fps=np.float32(fps))
    partial.replace(out)
    return dict(take=str(take), frames=int(index), found=int(valid.sum()))


def _hold(values, valid):
    """Frames without a face keep the nearest valid frame instead of neutral."""
    found = np.flatnonzero(valid)
    if not len(found):
        return
    for index in np.flatnonzero(~valid):
        nearest = found[np.argmin(np.abs(found - index))]
        values[index] = values[nearest]


def main():
    args = sys.argv[2:]
    root = Path(__file__).resolve().parents[1]
    takes = [Path(args[i + 1]) for i, a in enumerate(args) if a == '--take' and i + 1 < len(args)]
    if not takes:
        emit('failed', stage='face', message='No take was given.')
        return 1
    summary = []
    for number, take in enumerate(takes):
        def progress(done, total, number=number):
            emit('progress', stage='face', current=number * 1000 + int(1000 * done / max(total, 1)),
                 total=len(takes) * 1000, message='Face capture %d/%d' % (number + 1, len(takes)))
        try:
            summary.append(extract(take, root, progress))
        except Exception as error:  # report per take, keep going
            summary.append(dict(take=str(take), error=str(error)))
    emit('done', stage='face', takes=summary)
    return 0 if all('error' not in row for row in summary) else 1


if __name__ == '__main__':
    sys.exit(main())
