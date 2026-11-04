"""Serve a take's base body the way the app's live link does.

  runtime/python.exe -B tools/live-base-server.py TAKE_FOLDER [--port 9241]

Used by tools/verify-live-base.py: the same ``mocap.live_link`` server and the
same body publishing path the GUI uses, so the Blender side is tested against
the app's real stream instead of a hand-written packet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from mocap.live_body import for_project  # noqa: E402
from mocap.live_link import LiveLink  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take', help='Take folder, or a project workspace')
    parser.add_argument('--port', type=int, default=9241)
    parser.add_argument('--seconds', type=float, default=40.0)
    options = parser.parse_args()
    body = for_project(Path(options.take))
    if body is None:
        raise SystemExit('No committed body export under ' + options.take)
    link = LiveLink(port=options.port)
    ok, message = link.start()
    if not ok:
        raise SystemExit(message)
    print('SERVING %d frames, body %d vertices, %.2f m'
          % (body.count, len(body.skin['vertices']), body.height), flush=True)
    # The app sends the character first, then its pose: the same order and the
    # same call the GUI makes when Live is switched on.
    link.publish_character(body.rest_file(), body.names, 'RGBMoCap Body', None, 0,
                           body.fps, body.count, 'Person 1', None, body.height,
                           rest_format='npz')
    deadline = time.time() + options.seconds
    index = 0
    try:
        while time.time() < deadline:
            link.publish_character(body.rest_file(), body.names, 'RGBMoCap Body',
                                   body.matrices(index), index, body.fps, body.count,
                                   'Person 1', None, body.height, rest_format='npz')
            index = (index + 3) % body.count
            time.sleep(1.0 / 30.0)
    finally:
        link.stop()
    print('STOPPED', flush=True)


if __name__ == '__main__':
    sys.exit(main())
