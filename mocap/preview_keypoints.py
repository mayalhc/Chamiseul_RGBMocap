"""COCO-17 2D points drawn over the video panel.

The video panel draws two things from the project it displays: the tracked box
from ``tracking.npz`` and a COCO-17 skeleton from ``keypoints.npz``. Only the
GVHMR pose analysis used to write that skeleton, so the same take showed bones
with one model and only a box with another, and person rows that point at a
folder without the file showed no bones at all.

ExMR computes COCO-63 points (17 body + 6 feet + 20 left + 20 right hand,
see ``third_party/ExMR/exmr/preprocessing/keypoints/mappings.py``) before it
reconstructs anything. Its first 17 entries are exactly the body points the
overlay draws, so saving them next to the take gives every ExMR take the same
skeleton the GVHMR takes already show. FlowHMR predicts no 2D points, so its
takes keep the box only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BODY_POINTS = 17


def from_coco63(points):
    """ExMR ``(frames, 63, 3)`` points -> ``(frames, 17, 2)`` and confidence."""
    array = points.detach().float().cpu().numpy() if hasattr(points, "detach") else np.asarray(points, np.float32)
    if array.ndim != 3 or array.shape[1] < BODY_POINTS or array.shape[2] < 3:
        raise ValueError(f"ExMR 2D points have an unexpected shape {array.shape}")
    body = np.ascontiguousarray(array[:, :BODY_POINTS, :], np.float32)
    if not np.isfinite(body).all():
        raise ValueError("ExMR 2D points contain non-finite values")
    return body[:, :, :2], body[:, :, 2]


def has_points(project) -> bool:
    return (Path(project) / "keypoints.npz").is_file()


def save(project, keypoints, confidence) -> Path:
    """Write ``keypoints.npz`` so the overlay can map it without a re-run."""
    project = Path(project)
    keypoints = np.ascontiguousarray(keypoints, np.float32)
    confidence = np.ascontiguousarray(confidence, np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (BODY_POINTS, 2):
        raise ValueError(f"Invalid 2D point shape {keypoints.shape}")
    if confidence.shape != keypoints.shape[:2]:
        confidence = confidence.reshape(keypoints.shape[:2])
    target = project / "keypoints.npz"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, keypoints=keypoints, confidence=confidence)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def save_exmr(project, points) -> Path:
    """Save a take's ExMR COCO-63 points as the overlay's COCO-17 skeleton."""
    return save(project, *from_coco63(points))
