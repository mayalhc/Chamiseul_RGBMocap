"""Rebuild motion only where the person detector lost the actor.

The tracker interpolates a missed bounding box, so the reconstruction inside a
detection hole comes from the body prior. This stage replaces those frames from
the observed motion on both sides: the pelvis follows a velocity-matched Hermite
path, body rotations cross-fade in local space, and a foot that is locked on both
sides stays planted. Frames outside a hole are never touched.

Quaternions here are w,x,y,z, matching the corrected arrays in `correction.py`.
"""
import numpy as np

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
MIN_FRAMES = 2
MAX_SECONDS = 2.0
MAX_SPEED = 2.5          # m/s of extrapolation at a clip edge
MAX_SHIFT = .5           # m of root correction for one pinned foot
PIN_WINDOW = 3           # observed frames averaged on each side of a hole
VELOCITY_WINDOW = 4


def _qmul(a, b):
    av, bv = a[..., 1:], b[..., 1:]
    w = a[..., :1] * b[..., :1] - np.sum(av * bv, axis=-1, keepdims=True)
    v = a[..., :1] * bv + b[..., :1] * av + np.cross(av, bv)
    return np.concatenate((w, v), axis=-1)


def _qconj(q):
    return q * np.array([1., -1., -1., -1.])


def _qnorm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _nlerp(a, b, t):
    """Shortest-arc normalised blend; t broadcasts over the last axis."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    return _qnorm((1. - t) * a + t * b)


def _local(global_quat, frame):
    """Parent-relative rotations of one frame, so a pose can blend cleanly."""
    q = np.asarray(global_quat[frame], np.float64)
    local = np.empty_like(q)
    local[0] = q[0]
    for joint in range(1, len(q)):
        local[joint] = _qmul(_qconj(q[PARENTS[joint]]), q[joint])
    return local


def _speed(roots, index, other, fps):
    """Velocity at an anchor from one neighbouring frame, bounded for stability."""
    if index == other:
        return np.zeros(3)
    value = (np.asarray(roots[index], np.float64) - np.asarray(roots[other], np.float64)) / ((index - other) / fps)
    length = float(np.linalg.norm(value))
    return value if length <= MAX_SPEED else value * (MAX_SPEED / length)


def _hermite(p0, m0, p1, m1, duration, u):
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * duration * m0
            + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * duration * m1)


def plan(spans, frames, fps, maximum_seconds=MAX_SECONDS):
    """Split detection holes into rebuilt and skipped spans."""
    limit = max(MIN_FRAMES, round(float(fps) * maximum_seconds))
    fillable, skipped = [], []
    if spans is None:
        return fillable, skipped
    for start, end in spans.gaps:
        start, end = max(0, int(start)), min(int(frames), int(end))
        if end - start < MIN_FRAMES:
            continue
        if end - start > limit or (start == 0 and end == frames):
            skipped.append((start, end))
        else:
            fillable.append((start, end))
    return fillable, skipped


def fill(roots, quats, spans, sample, fps=30., segments=(), floor=None,
         maximum_seconds=MAX_SECONDS):
    """Rewrite root and joint rotations inside detection holes, in place.

    `sample(roots, quats)` returns (frames, 4, 3) sole probes; it must be
    translation-equivariant, which makes one root shift exact for a foot pin.
    Returns (report, mask) where mask marks the frames that were replaced.
    """
    n = len(roots)
    mask = np.zeros(n, bool)
    report = dict(version=1, gaps=0, filled_frames=0, skipped_frames=0,
                  skipped_gaps=0, pinned_feet=0, max_shift_m=0.,
                  max_speed_mps=0., pin_correction_max_m=0., warning='')
    fillable, skipped = plan(spans, n, fps, maximum_seconds)
    report['skipped_gaps'] = len(skipped)
    report['skipped_frames'] = int(sum(end - start for start, end in skipped))
    if skipped:
        report['warning'] = (f'{len(skipped)} detection gap(s) longer than '
                             f'{maximum_seconds:.0f} s stayed as reconstructed.')
    locked = {}
    for segment in segments or ():
        try:
            locked.setdefault(int(segment['foot']), []).append(
                (int(segment['start']), int(segment['end'])))
        except (KeyError, TypeError, ValueError):
            continue
    speeds = []
    for start, end in fillable:
        prev, nxt = start - 1, end
        if prev < 0 and nxt >= n:
            continue
        count = end - start
        if prev >= 0 and nxt < n:
            duration = (nxt - prev) / fps
            m0 = _speed(roots, prev, max(0, prev - VELOCITY_WINDOW), fps)
            m1 = _speed(roots, nxt, min(n - 1, nxt + VELOCITY_WINDOW), fps)
            u = (np.arange(start, end, dtype=np.float64) - prev) / (nxt - prev)
            gap_roots = np.stack([_hermite(np.asarray(roots[prev], np.float64), m0,
                                           np.asarray(roots[nxt], np.float64), m1,
                                           duration, t) for t in u])
            local = _nlerp(_local(quats, prev)[None], _local(quats, nxt)[None],
                           u[:, None, None])
            root_quat = _nlerp(np.asarray(quats[prev, 0], np.float64)[None, None],
                               np.asarray(quats[nxt, 0], np.float64)[None, None],
                               u[:, None])[:, 0]
        elif nxt < n:
            # The clip starts inside a hole: coast backwards from the first anchor.
            anchor = np.asarray(roots[nxt], np.float64)
            velocity = _speed(roots, min(n - 1, nxt + VELOCITY_WINDOW), nxt, fps)
            gap_roots = np.stack([anchor - velocity * ((nxt - index) / fps)
                                  for index in range(start, end)])
            local = np.repeat(_local(quats, nxt)[None], count, 0)
            root_quat = np.repeat(np.asarray(quats[nxt, 0], np.float64)[None], count, 0)
        else:
            anchor = np.asarray(roots[prev], np.float64)
            velocity = _speed(roots, prev, max(0, prev - VELOCITY_WINDOW), fps)
            gap_roots = np.stack([anchor + velocity * ((index - prev) / fps)
                                  for index in range(start, end)])
            local = np.repeat(_local(quats, prev)[None], count, 0)
            root_quat = np.repeat(np.asarray(quats[prev, 0], np.float64)[None], count, 0)
        gap_quats = np.empty((count, len(PARENTS), 4), np.float64)
        gap_quats[:, 0] = root_quat
        for joint in range(1, len(PARENTS)):
            gap_quats[:, joint] = _qmul(gap_quats[:, PARENTS[joint]], local[:, joint])
        # One root shift per locked foot keeps the sole on its observed patch.
        if prev >= 0 and nxt < n:
            for foot, intervals in locked.items():
                if not any(a <= prev and b >= nxt for a, b in intervals):
                    continue
                low = max(0, prev - PIN_WINDOW + 1)
                before = sample(np.asarray(roots[low:prev + 1]),
                                np.asarray(quats[low:prev + 1]))[0]
                high = min(n, nxt + PIN_WINDOW)
                after = sample(np.asarray(roots[nxt:high]),
                               np.asarray(quats[nxt:high]))[0]
                pair = slice(2 * foot, 2 * foot + 2)
                target = np.concatenate((before[:, pair], after[:, pair])).reshape(-1, 3).mean(0)
                probes = sample(gap_roots, gap_quats)[0][:, pair].mean(1)
                shift = np.clip(target - probes, -MAX_SHIFT, MAX_SHIFT)
                gap_roots = gap_roots + shift
                report['pinned_feet'] += 1
                report['pin_correction_max_m'] = max(report['pin_correction_max_m'],
                                                     float(np.abs(shift).max()))
                report['max_shift_m'] = max(report['max_shift_m'], float(np.abs(shift).max()))
        if floor is not None:
            lowest = sample(gap_roots, gap_quats)[0][..., 1].min(1)
            lift = np.maximum(float(floor) - lowest, 0.)
            if lift.any():
                gap_roots = gap_roots + lift[:, None]
                report['max_shift_m'] = max(report['max_shift_m'], float(lift.max()))
        # Boundary steps belong to this hole; separate holes are unrelated.
        if prev >= 0:
            steps = [np.linalg.norm(gap_roots[0] - np.asarray(roots[prev], np.float64))]
        else:
            steps = []
        if count > 1:
            steps.extend(np.linalg.norm(np.diff(gap_roots, axis=0), axis=1).tolist())
        if nxt < n:
            steps.append(np.linalg.norm(np.asarray(roots[nxt], np.float64) - gap_roots[-1]))
        speeds.extend(steps)
        roots[start:end] = gap_roots
        quats[start:end] = gap_quats
        mask[start:end] = True
    if fillable:
        report['gaps'] = len(fillable)
        report['filled_frames'] = int(mask.sum())
        report['max_speed_mps'] = float(max(speeds) * fps) if speeds else 0.
    return {key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in report.items()}, mask
