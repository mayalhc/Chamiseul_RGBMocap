"""InstantHMR as the live body estimator: a person box in, MHR70 camera-space keypoints out.

The network and the Momentum Human Rig body graph run in ONNX Runtime on the CUDA provider.
The result is the same 70-keypoint layout and camera frame (metres, x right, y down, z forward)
that SAM 3D Body gives the live solver, at a fraction of the cost per frame.

ONNX Runtime GPU lives in its own folder (see tools/setup_instanthmr.py) and is put on the path
only here, after PyTorch is imported, so that both share PyTorch's CUDA libraries.
"""
from __future__ import annotations

import sys

from .instanthmr_models import BODY, DEFAULT_VARIANT, VARIANTS
from .paths import ROOT

OVERLAY = ROOT / 'runtime_instanthmr'
SOURCE = ROOT / 'third_party' / 'InstantHMR'
ENGINES = ('sam', 'instanthmr')
DEFAULT_ENGINE = 'sam'


def overlay_ready():
    return (OVERLAY / 'onnxruntime' / '__init__.py').is_file()


def missing_models(variant=DEFAULT_VARIANT):
    from .models import BY_KEY, ready
    return [BY_KEY[key] for key in (VARIANTS[variant], BODY) if not ready(BY_KEY[key])]


def problem(variant=DEFAULT_VARIANT):
    """Why InstantHMR cannot start, in words for the person, or None when it can."""
    if variant not in VARIANTS:
        return 'Unknown InstantHMR model: ' + str(variant)
    if not overlay_ready():
        return 'InstantHMR needs ONNX Runtime GPU. Close the app and run Setup_InstantHMR.cmd.'
    absent = missing_models(variant)
    if absent:
        return 'InstantHMR model files are missing: ' + ', '.join(model.name for model in absent) \
            + '. Open Model Manager.'
    return None


def load(variant=DEFAULT_VARIANT, device='cuda'):
    """The ready InstantHMR network. Raises RuntimeError with a readable reason otherwise."""
    reason = problem(variant)
    if reason:
        raise RuntimeError(reason)
    import torch  # noqa: F401  (first, so ONNX Runtime reuses its CUDA libraries)
    for folder in (str(SOURCE), str(OVERLAY)):
        if folder not in sys.path:
            sys.path.insert(0, folder)
    from instanthmr import InstantHMR
    from .models import BY_KEY
    model = InstantHMR(BY_KEY[VARIANTS[variant]].path, BY_KEY[BODY].path, device=device, max_batch=1)
    if device == 'cuda' and model.provider != 'CUDAExecutionProvider':
        raise RuntimeError('ONNX Runtime did not start the CUDA provider (got %s).' % model.provider)
    model.warmup()
    return model


def keypoints(model, rgb, box):
    """(70, 3) MHR70 keypoints in metres for the person inside `box`, camera frame."""
    return model.predict(rgb, [box])[0].joints_3d_cam


def settings():
    """(engine, variant) from the local settings file."""
    from .resource_safety import path, read_settings
    try:
        saved = read_settings(path())
    except (OSError, ValueError):
        saved = {}
    engine = saved.get('live_engine')
    variant = saved.get('live_instanthmr_variant')
    return (engine if engine in ENGINES else DEFAULT_ENGINE,
            variant if variant in VARIANTS else DEFAULT_VARIANT)


def save(engine, variant):
    if engine not in ENGINES or variant not in VARIANTS:
        raise ValueError('Invalid live estimator selection.')
    from .resource_safety import path, read_settings
    from .storage import atomic_json
    target = path()
    saved = read_settings(target)
    saved['live_engine'] = engine
    saved['live_instanthmr_variant'] = variant
    atomic_json(target, saved)
