"""Continuous local ground height for flat-ground foot correction.

One physical floor is the model, but a reconstructed take drifts vertically:
the same planted sole slowly rises over a long clip, so a single global plane
either floats the actor in one half of it or sinks the actor in the other. This
module recovers where the sole actually rests, fits a stiff smooth curve through
those settled samples and never lets a jump, a kick or a lifted foot raise the
ground.

The curve owns the *reference*, not the contact labels: contact detection and
the anatomical solver keep their existing responsibilities. Grounding against
the curve makes the published clearance above the floor equal to the source
clearance above the local ground, so the correction removes drift without
changing how high the feet really are.
"""
import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.linalg import solve_banded

WINDOW = 1.0        # seconds of video searched for the local lowest sole
NEAR = .08          # a settled sole lies this close to the local lowest point
SLOW = .25          # metres per second of support-material travel
STIFFNESS = 20000.  # second-difference penalty; keeps drift, drops stance bob
ROBUST = .05        # residual beyond this is treated as a bad sample, in metres
TRUST = .02         # weight of frames without a settled sample
ANCHOR_SECONDS = 1.  # the reference plane is measured over this first settled span


def settled(heights, speed, fps):
    """Frames where a sole rests near the local lowest point and barely moves.

    The rolling minimum is only a prior for "locally lowest"; its lag does not
    bias the fitted curve because the fit itself is anchored on samples.
    """
    reach = max(2, round(float(fps) * WINDOW))
    prior = minimum_filter1d(heights.min(1), size=2 * reach + 1, mode='nearest')
    return (heights - prior[:, None] < NEAR) & (speed < SLOW)


def _fit(samples, weight, stiffness):
    """Weighted smoothing spline with a second-difference penalty (banded)."""
    n = len(samples)
    if n < 4:
        return samples.copy()
    diagonal = weight + stiffness * np.r_[1., 5., np.full(n - 4, 6.), 5., 1.]
    first = -2. * stiffness * np.r_[1., np.full(n - 3, 2.), 1.]
    bands = np.zeros((5, n))
    bands[2] = diagonal
    bands[1, 1:] = first
    bands[3, :-1] = first
    bands[0, 2:] = stiffness
    bands[4, :-2] = stiffness
    return solve_banded((2, 2), bands, weight * samples, check_finite=False)


def _speed(probes, fps):
    points = np.asarray(probes).reshape(-1, 2, 2, 3).mean(2)
    speed = np.zeros((len(points), 2))
    if len(points) > 1:
        speed[1:] = np.linalg.norm(np.diff(points[..., [0, 2]], axis=0), axis=-1) * fps
        speed[0] = speed[1]
    return speed


def reference(probes, fps):
    """Local resting height per frame plus the plane the take starts from.

    ``anchor`` is measured over the first settled span instead of the whole
    take, so a drifting clip keeps its opening floor and never starts with a
    vertical jump. It is only a default: a measured or shared scene floor
    replaces it.
    """
    heights = np.asarray(probes)[..., 1].reshape(-1, 2, 2).min(2)
    mask = settled(heights, _speed(probes, fps), fps)
    fit = curve(heights, mask, fps)
    if fit is None:
        return None
    known = np.flatnonzero(mask.any(1))
    span = known[:max(1, min(len(known), round(float(fps) * ANCHOR_SECONDS)))]
    fit['anchor'] = float(np.median(fit['level'][span])) if len(span) else float(fit['level'][0])
    return fit


def curve(heights, mask, fps):
    """Smooth ground height per frame, or None when nothing ever settles."""
    n, frames = len(heights), int(mask.sum())
    if frames < max(2, round(float(fps) * .1)):
        return None
    sample = np.where(mask, heights, np.inf).min(1)
    known = np.isfinite(sample)
    index = np.flatnonzero(known)
    filled = np.interp(np.arange(n), index, sample[index])
    stiffness = STIFFNESS * (float(fps) / 30.) ** 4
    weight = np.where(known, 1., TRUST)
    fitted = _fit(filled, weight, stiffness)
    # One robust pass: a jittered or mis-shaped sole must not drag the ground.
    residual = np.abs(sample - fitted)
    outlier = known & (residual > ROBUST)
    if outlier.any() and (~outlier).sum() >= max(2, round(float(fps) * .1)):
        weight = weight.copy()
        weight[outlier] = .0
        fitted = _fit(filled, weight, stiffness)
    clean = known & ~outlier
    return dict(level=fitted, frames=frames, samples=int(frames),
                residual_p95=float(np.quantile(residual[clean], .95)) if clean.any() else 0.,
                outliers=int(outlier.sum()))


def clearance(probes, fps):
    """Lowest-sole height above the local ground, independent of any drift."""
    probes = np.asarray(probes)
    ref = reference(probes, fps)
    if ref is None:
        return None, None
    heights = probes[..., 1].reshape(-1, 2, 2).min(2)
    return heights.min(1) - ref['level'], ref


def audit(probes, fps, floor=None):
    """Visible hover and airborne share, plus the drift that was present.

    ``floor`` anchors the report to the plane the app draws. Without one the
    local resting height is used, which is correct for a single take but not
    comparable across two different outputs unless the same floor is passed.
    """
    probes = np.asarray(probes)
    heights = probes[..., 1].reshape(-1, 2, 2).min(2)
    lowest = heights.min(1)
    ref = reference(probes, fps)
    if ref is None:
        return {}
    level = ref['level'] if floor is None else np.full(len(lowest), float(floor))
    value = lowest - level
    hover = (value > .02) & (value <= .08)
    seconds = max(1, round(float(fps)))
    minima = np.array([value[i:i + seconds].min() for i in range(0, len(value), seconds)])
    return dict(reference_plane=float(level[0]) if floor is None else float(floor),
                ground_drift_m=float(np.ptp(ref['level'])), ground_samples=ref['samples'],
                ground_residual_p95_m=ref['residual_p95'],
                hover_frames=int(hover.sum()), hover_pct=100. * float(hover.mean()),
                hover_p95_m=float(np.quantile(value[hover], .95)) if hover.any() else 0.,
                grounded_pct=100. * float((np.abs(value) <= .015).mean()),
                sink_pct=100. * float((value < -.015).mean()),
                airborne_over_8cm=100. * float((value > .08).mean()),
                per_second_float_range_m=float(np.ptp(minima)))
