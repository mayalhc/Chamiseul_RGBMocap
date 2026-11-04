"""Give the video overlay's COCO-17 points to ExMR takes already on disk.

New ExMR runs save ``keypoints.npz`` next to the take, but projects captured
before that change only have ``exmr_keypoints.pt``. This walks a projects
folder and derives the overlay file from those points, so the skeleton shows up
without re-running the Sapiens pass.

    ./runtime/python.exe tools/backfill-2d-keypoints.py [PATH ...] [--force]

With no path it scans ``projects/``. ``--force`` rewrites files that exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mocap.paths import PROJECTS  # noqa: E402
from mocap.preview_keypoints import has_points, save_exmr  # noqa: E402


def load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    # Project names come from video titles and can hold any script; a cp949
    # console must not abort the run halfway through a folder.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[PROJECTS])
    parser.add_argument("--force", action="store_true", help="Rewrite existing keypoints.npz files.")
    options = parser.parse_args()
    written, skipped, failed = 0, 0, 0
    for root in options.paths:
        for source in sorted(Path(root).rglob("exmr_keypoints.pt")):
            project = source.parent
            if has_points(project) and not options.force:
                skipped += 1
                continue
            try:
                save_exmr(project, load(source))
            except (OSError, ValueError, RuntimeError) as error:
                failed += 1
                print(f"FAILED {project}: {error}")
                continue
            written += 1
            print(f"wrote {project.relative_to(Path(root).parent)}\\keypoints.npz")
    print(f"{written} written, {skipped} already present, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
