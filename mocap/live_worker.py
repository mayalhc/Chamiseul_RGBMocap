"""Live inference worker: camera frames in, solved body rows out.

A separate process on purpose - the app never lets an upstream model tree
leak into the GUI. Per camera frame the worker detects the person (YOLO) and
runs SAM 3D Body's body decoder only (``inference_type='body'`` - the hand
decoders cost ten times the body and the live rig needs none of it). The
MHR70 3D keypoints land as camera-space metres; ``live_body_solve`` maps them
onto the app's 22 joints, smooths them, and aims the base character's own rig
at them, so the live view shows the same skinned character a reconstruction
does.

The model load is minutes (once per live session); each frame then costs a
few hundred milliseconds - the character in the 3D view follows at a real
live rate.

Run as ``python -m mocap.live_worker --channel <name>`` (the ExMR runtime);
``--video`` reads a file instead so the solver can be verified against a take.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
import traceback

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='', help='shared-memory channel name (live mode)')
    parser.add_argument('--video', default='', help='read frames from a video instead (verification)')
    parser.add_argument('--yolo-every', type=int, default=4, help='re-detect the person every N frames')
    parser.add_argument('--max-frames', type=int, default=0, help='stop after N solved frames (0: until stopped)')
    parser.add_argument('--detect-only', action='store_true',
                        help='person boxes for the live view only: YOLO, no SAM')
    parser.add_argument('--preload', action='store_true',
                        help='load the models now and wait for the GUI to hand over a live channel')
    parser.add_argument('--rig', default='',
                        help="the open take's prepared character rig (its asset npz)")
    parser.add_argument('--binding', default='',
                        help="the open take's prepared character skin binding")
    parser.add_argument('--progress', default='',
                        help='file the GUI reads for the start-up step and its numbers')
    return parser.parse_args()


MARKER = ('cache', 'live-warm-channel.txt')


def attach_when_handed_over():
    """Wait for the GUI to publish a channel name, then attach to it.

    A preloaded worker is started with the app, long before any live source:
    it loads the models, then idles here. The name only appears once the GUI
    has created the channel, so the attach never races its creation.
    """
    from pathlib import Path
    from .live_channel import LiveChannel
    from .paths import HOME
    marker = Path(HOME).joinpath(*MARKER)
    while True:
        try:
            name = marker.read_text(encoding='utf-8').strip()
        except OSError:
            name = ''
        if name:
            try:
                return LiveChannel(name)
            except (OSError, ValueError):
                # The session ended between the write and this attach.
                name = ''
        time.sleep(0.2)


def main():
    args = parse_args()
    from pathlib import Path
    from .live_channel import Progress
    # The 3D view counts these steps out while the character is coming up.
    progress = Progress(args.progress)
    progress.step(1, 'Preparing the solver')
    if args.video:
        # The engine activation below changes the working directory.
        args.video = str(Path(args.video).resolve())
    from .exmr_backend import activate
    activate()
    import torch
    import cv2
    # The crop shapes are constant, so benchmark every convolution once.
    torch.backends.cudnn.benchmark = True

    from .live_body_solve import LiveBody, Smoother, cached_rest
    from .models import BY_KEY
    from .paths import HOME, UPSTREAM

    channel = None
    if args.channel:
        from .live_channel import LiveChannel
        channel = LiveChannel(args.channel)
    elif not (args.preload or args.video):
        raise ValueError('A --channel or --video source is required')
    if args.video:
        source = cv2.VideoCapture(args.video)
        if not source.isOpened():
            raise RuntimeError('Cannot open ' + args.video)
        source_kind = 'video'
    elif channel is not None or args.preload:
        # Preloaded: the frames arrive once a live source hands over a channel.
        source, source_kind = None, 'channel'
    else:
        raise ValueError('A --channel or --video source is required')

    # The character the 3D view already shows: the open take's own prepared rig
    # and skin binding. Without a take, the neutral base character stands in.
    progress.step(2, 'Preparing the character')
    supplied = {}
    if args.rig:
        with np.load(args.rig, allow_pickle=False) as data:
            supplied = dict(rest=data['rest'], parents=data['parents'], ends=data['ends'],
                            alignment=data['anatomical_calibration'])
    body = LiveBody(rig=supplied) if supplied else LiveBody(cached_rest(HOME, BY_KEY['smplx'].path))
    smoother = Smoother()
    take_binding = None
    if args.binding:
        with np.load(args.binding, allow_pickle=False) as data:
            take_binding = {key: data[key] for key in
                            ('vertices', 'frames', 'local', 'rest', 'parents', 'betas')}

    # The follower renders the BASE CHARACTER: the same packed SMPL-X mesh
    # frames a take's mesh preview uses, skinned from the solved joints.
    import numpy as _np
    _zeros_betas = _np.zeros((1, 10), _np.float32)
    mesh_binding = None
    mesh_model = None
    def mesh_setup():
        nonlocal mesh_binding, mesh_model
        if take_binding is not None:
            # The take's character: its binding is already the one the result
            # view renders with, so nothing has to be rebuilt or published.
            mesh_binding = take_binding
            return None
        import smplx
        model_path = str(BY_KEY['smplx'].path)
        smpl = smplx.SMPLX(model_path, gender='neutral', num_pca_comps=12,
                           flat_hand_mean=False, batch_size=1).eval()
        from .rig_mesh import make_binding
        mesh_binding = make_binding(smpl, body.rig, _zeros_betas)
        mesh_model = smpl
        # The 3D view skins this base character on the GPU and never imports
        # torch, so the small canonical binding travels as a file beside the
        # shared mesh buffers the GUI already loads.
        binding_file = Path(HOME) / 'cache' / 'live-binding.npz'
        temporary = binding_file.with_name(binding_file.name + '.partial.npz')
        binding_file.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(temporary, **mesh_binding)
        os.replace(temporary, binding_file)
        from .mesh_data import MESH_VERSION, adjacency
        asset = Path(model_path).parent / f'.mesh-lbs-v{MESH_VERSION}-{Path(model_path).stat().st_size}-{Path(model_path).stat().st_mtime_ns}.npz'
        if not asset.exists():
            packed = np.concatenate((smpl.v_template.numpy().reshape(-1),
                torch.cat((smpl.shapedirs, smpl.expr_dirs), dim=-1).numpy().transpose(2, 0, 1).reshape(-1),
                smpl.posedirs.numpy().reshape(-1), smpl.lbs_weights.numpy().reshape(-1))).astype(np.float32)
            offsets, neighbors = adjacency(smpl.faces, len(smpl.v_template))
            temporary = asset.with_suffix('.partial.npz')
            _np.savez(temporary, packed=packed, faces=smpl.faces.astype(_np.uint32),
                      offsets=offsets, neighbors=neighbors, vertex_count=_np.array(len(smpl.v_template)))
            os.replace(temporary, asset)
        return asset
    def mesh_frame(row):
        """The posed base-character frame for one solved (52, 4, 4) rig row.

        Only the pose travels: the GUI skins the packed SMPL-X mesh on the GPU
        with it. Measured at 0.3 ms per frame, so the live rate is unaffected.
        """
        from .rig_mesh import mesh_frames
        frame, _error = mesh_frames(mesh_model, body.rig, row[None, :, :3, 3],
                                    row[None, :, :3, :3], mesh_binding['betas'], mesh_binding)
        return frame[0]

    # Before the first inference: the base character can be shown as soon as
    # the body model is loaded, not after a multi-minute detector start.
    mesh_setup()

    progress.step(3, 'Loading the person detector')
    print('LIVE-WORKER: loading YOLO', flush=True)
    from pathlib import Path as _Path
    from ultralytics import YOLO
    import ultralytics.nn.tasks as yolo_tasks
    # This old upstream release predates PyTorch's weights_only default.
    # Limit full-object loading to our explicitly selected official detector.
    def load_official_detector(weight):
        selected = _Path(weight).resolve()
        if selected != BY_KEY['yolo'].path.resolve():
            raise ValueError('Unexpected detector checkpoint path')
        return torch.load(selected, map_location='cpu', weights_only=False), str(selected)
    yolo_tasks.torch_safe_load = load_official_detector
    detector = YOLO(str(BY_KEY['yolo'].path))

    focal = None
    principal = None
    last_box = None
    last_seq = 0

    if args.detect_only:
        # The live view's tracking box: YOLO on every fed frame, no body
        # model - the character still comes only from the reconstruction.
        def detect(image):
            nonlocal last_box, focal, principal
            height, width = image.shape[:2]
            focal = float(math.hypot(width, height))
            principal = (width / 2.0, height / 2.0)
            small = cv2.resize(image, (480, max(1, round(480 * height / width))))
            result = detector.predict(small, classes=[0], imgsz=480, conf=0.35, iou=0.5,
                                      max_det=8, verbose=False,
                                      device='cuda' if torch.cuda.is_available() else 'cpu')[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            if len(boxes):
                scale = width / small.shape[1]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                last_box = boxes[int(np.argmax(areas))].astype(np.float64) * scale
            return last_box

        def next_frame():
            if source_kind == 'channel':
                return channel.read_frame(last_seq)
            ok, frame = source.read()
            if not ok:
                return None, 0
            return frame, 1

        if channel is not None:
            from .live_channel import READY
            channel.set_state(READY)
        print('LIVE-WORKER: ready (detect-only)', flush=True)
        sequence = 0
        while True:
            if channel is not None and channel.state() in (3, 4):
                break
            frame, tag = next_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            last_seq = tag
            try:
                sequence += 1
                found = detect(frame)
                if found is not None:
                    last_box = found
                if last_box is not None and channel is not None:
                    channel.publish_box(last_box, sequence)
            except Exception:
                if channel is not None:
                    channel.fail(''.join(traceback.format_exc()[-450:]))
                raise
        if channel is not None:
            channel.set_state(0)
        print('LIVE-WORKER: stopped after %d detections' % sequence, flush=True)
        return

    progress.step(4, 'Loading the body model')
    print('LIVE-WORKER: loading SAM 3D Body (once; this takes a few minutes)', flush=True)
    flow_root = str(UPSTREAM / 'FlowHMR')
    if flow_root not in sys.path:
        sys.path.insert(0, flow_root)
    from .flow_sam import build_extractor
    extractor = build_extractor(torch.device('cuda'), fast=True, low_memory=False)
    estimator = extractor.estimator
    progress.step(5, 'Computing the first pose')

    def detect(image):
        """The largest person in a small detection image (~15 ms)."""
        nonlocal last_box, focal, principal
        height, width = image.shape[:2]
        focal = float(math.hypot(width, height))
        principal = (width / 2.0, height / 2.0)
        small = cv2.resize(image, (480, max(1, round(480 * height / width))))
        result = detector.predict(small, classes=[0], imgsz=480, conf=0.35, iou=0.5,
                                  max_det=8, verbose=False,
                                  device='cuda' if torch.cuda.is_available() else 'cpu')[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        if len(boxes):
            scale = width / small.shape[1]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            last_box = boxes[int(np.argmax(areas))].astype(np.float64) * scale
        return last_box

    def solve_frame(image, box):
        """SAM 3D Body body-only -> the app's 22 joints."""
        height, width = image.shape[:2]
        k = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = estimator.process_one_image(image, bboxes=box.astype(np.float32),
                                                  cam_int=torch.from_numpy(k[None]),
                                                  inference_type='body')
        person = outputs[0] if isinstance(outputs, list) else outputs
        return body.positions(person['pred_keypoints_3d'], person['pred_cam_t'])

    def next_frame():
        if source_kind == 'channel':
            return channel.read_frame(last_seq)
        ok, frame = source.read()
        if not ok:
            return None, 0
        return frame, 1

    focal = None
    principal = None
    last_box = None
    last_seq = 0

    if channel is None and args.preload:
        # One synthetic frame now. The first inference is the expensive one
        # (kernel selection and memory setup), and paying it here keeps it out
        # of the moment the character is called in.
        try:
            blank = np.full((960, 640, 3), 32, np.uint8)
            focal = float(math.hypot(blank.shape[1], blank.shape[0]))
            principal = (blank.shape[1] / 2.0, blank.shape[0] / 2.0)
            started = time.monotonic()
            detect(blank)
            with contextlib.redirect_stdout(io.StringIO()):
                warmed = solve_frame(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB),
                                     np.array([160., 80., 480., 880.]))
            body.rig_row(warmed)
            print('LIVE-WORKER: warmed up in %.1f s' % (time.monotonic() - started), flush=True)
        except Exception:
            # A warm-up is an optimisation: never let it stop the worker.
            print('LIVE-WORKER: warm-up skipped: ' + traceback.format_exc()[-200:], flush=True)
        print('LIVE-WORKER: preloaded, waiting for a live source', flush=True)
        channel = attach_when_handed_over()
    if channel is not None:
        from .live_channel import READY
        channel.set_state(READY)
    progress.step(5, 'Computing the first pose')
    print('LIVE-WORKER: ready (base-character follower)', flush=True)

    published = 0
    started = time.monotonic()
    marked = started
    sequence = 0

    while True:
        if channel is not None and channel.state() in (3, 4):  # stop requested or failed
            break
        if args.max_frames and published >= args.max_frames:
            break
        frame, tag = next_frame()
        if frame is None:
            if source_kind == 'video':
                break
            time.sleep(0.005)
            continue
        last_seq = tag
        try:
            sequence += 1
            if last_box is None or sequence % args.yolo_every == 0:
                if detect(frame) is None and last_box is None:
                    continue  # no person yet
            if last_box is not None and channel is not None:
                channel.publish_box(last_box, sequence)
            raw = solve_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), last_box)
            points = smoother.step(raw)
            row, placed = body.rig_row(points)
            if mesh_binding is not None and channel is not None:
                try:
                    channel.publish_mesh(mesh_frame(row))
                except Exception:
                    # The body is cosmetic: a mesh that cannot be built must not
                    # take the skeleton down with it.
                    mesh_binding = None
                    print('LIVE-WORKER: mesh follower disabled: '
                          + traceback.format_exc()[-300:], flush=True)
            if channel is not None:
                now = time.monotonic()
                if not channel.publish(published, placed, row,
                                       fps=(published + 1) / max(now - started, 1e-3)):
                    break
            published += 1
            now = time.monotonic()
            if now - marked >= 2.0:
                print('LIVE-WORKER: %d frames, %.1f fps' % (published, published / (now - started)), flush=True)
                marked = now
        except Exception:
            if channel is not None:
                channel.fail(''.join(traceback.format_exc()[-450:]))
            raise

    if channel is not None:
        channel.set_state(0)  # idle: the GUI learns the worker exited
    print('LIVE-WORKER: stopped after %d frames' % published, flush=True)


if __name__ == '__main__':
    main()
