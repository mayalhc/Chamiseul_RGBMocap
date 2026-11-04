"""Frames whose actor box was interpolated instead of detected.

The person tracker stores one boolean per frame: True when the detector saw the
actor. A missed frame keeps a box filled by interpolation, so the motion there
comes from the body prior rather than from an observation. These spans let the
app show estimated time apart from measured time without changing the motion.
"""
from bisect import bisect_right

import numpy as np

CHUNK = 4096
COLOR = '#c9922f'


class ObservedSpans:
    """Half-open runs of frames the person detector did not observe."""

    def __init__(self, gaps, frames, observed):
        self.gaps = gaps
        self.frames = int(frames)
        self.observed = int(observed)
        self.starts = [start for start, _ in gaps]

    @classmethod
    def from_array(cls, values):
        """Scan once in bounded chunks; never copy a whole recording."""
        if values is None:
            return None
        frames = len(values)
        if not frames:
            return None
        gaps, observed, start = [], 0, None
        for first in range(0, frames, CHUNK):
            stop = min(first + CHUNK, frames)
            seen = np.asarray(values[first:stop], dtype=bool).reshape(-1)
            if seen.shape != (stop - first,):
                return None
            observed += int(seen.sum())
            # Carry the run state across chunks so a long hole stays one span.
            carried = 0 if start is not None else 1
            edges = np.flatnonzero(np.diff(np.concatenate(([carried], seen)).astype(np.int8)))
            for edge in edges:
                frame = first + int(edge)
                if start is None:
                    start = frame
                else:
                    gaps.append((start, frame))
                    start = None
        if start is not None:
            gaps.append((start, frames))
        return cls(gaps, frames, observed)

    def gap_at(self, frame):
        if not self.gaps:
            return False
        index = bisect_right(self.starts, int(frame)) - 1
        return index >= 0 and self.gaps[index][0] <= frame < self.gaps[index][1]

    def gaps_in(self, first, end):
        """Spans overlapping [first, end), clipped; used for labels only."""
        if not self.gaps:
            return []
        index = max(0, bisect_right(self.starts, first) - 1)
        result = []
        while index < len(self.gaps) and self.gaps[index][0] < end:
            begin, finish = self.gaps[index]
            if finish > first:
                result.append((max(begin, first), min(finish, end)))
            index += 1
        return result

    def summary(self):
        return dict(frames=self.frames, observed=self.observed,
                    ratio=self.observed / max(1, self.frames),
                    gaps=len(self.gaps),
                    longest=max((end - start for start, end in self.gaps), default=0))


def label(spans, fps=None):
    """Short status text: how much of the take was actually observed."""
    if spans is None:
        return ''
    data = spans.summary()
    text = f'{100 * data["ratio"]:.0f}% seen'
    if data['gaps']:
        text += f' · {data["gaps"]} gap' + ('' if data['gaps'] == 1 else 's')
        if fps and data['longest'] >= max(1., float(fps) / 2):
            text += f' · longest {data["longest"] / float(fps):.1f} s'
    return text
