"""Check the link preview's playhead: length, loop and seeking.

Serves a short clip over HTTP (no download to disk by the app) and drives
``live_url.UrlSession`` exactly as the window does:

  * the clip's length is known, so the position control has a range;
  * frames keep arriving across the clip's end while Loop is on;
  * a seek reopens the stream at that second, and the playhead follows;
  * turning Loop off leaves the clip parked on its last frame instead.
"""
import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The GUI packages (PySide6) live in the gui overlay, as launch_source loads them.
sys.path.insert(0, str(ROOT / 'runtime_gui'))

from mocap.live_url import UrlSession            # noqa: E402
from mocap.video import executable              # noqa: E402

CLIP_SECONDS = 6
HERE = ROOT / 'cache' / 'probe-link-playback'


def make_clip():
    HERE.mkdir(parents=True, exist_ok=True)
    clip = HERE / 'clip.mp4'
    subprocess.run([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=25',
                    '-t', str(CLIP_SECONDS), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(clip)], check=True,
                   creationflags=0x08000000)
    return clip


def wait(session, predicate, timeout=25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def main():
    report = {'failures': []}
    failures = report['failures']
    clip = make_clip()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.RequestHandlerClass.log_message = lambda *a, **k: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d/clip.mp4' % server.server_address[1]
    session = None
    try:
        session = UrlSession(url)
        report['duration'] = round(session.duration, 2)
        report['loop_default'] = session.loop
        if abs(session.duration - CLIP_SECONDS) > 0.5:
            failures.append('the clip length was not read for the position control')

        session.start_preview()
        if not wait(session, lambda: session.frames >= 5):
            failures.append('no frames arrived from the link')
        report['start'] = session.report()

        # Loop on: frames keep coming past the clip's end.
        first = session.frames
        if not wait(session, lambda: session.frames > first + 40, timeout=20.0):
            failures.append('a looping clip stopped delivering frames at its end')
        report['after_one_pass'] = session.report()
        if session.finished:
            failures.append('a looping clip was reported as finished')

        # Seek: reopening at 4 s of a 6 s clip must show that part. A frame has
        # to arrive from the reopened stream, and the playhead has to sit there.
        before = session.frames
        session.seek(4.0)
        if not wait(session, lambda: session.frames > before and session.position() >= 3.9,
                    timeout=25.0):
            failures.append('seeking did not reopen the stream at that second')
        report['after_seek'] = session.report()

        # Loop off: the clip runs to its end and stays there.
        session.set_loop(False)
        if not wait(session, lambda: session.finished, timeout=20.0):
            failures.append('with Loop off the clip did not report its end')
        report['loop_off'] = session.report()
        frames = session.frames
        time.sleep(1.0)
        report['frames_after_end'] = session.frames - frames
        if session.frames != frames:
            failures.append('with Loop off frames kept arriving after the end')

        # Loop back on resumes the parked clip from where it stopped.
        session.set_loop(True)
        if not wait(session, lambda: not session.finished and session.frames > frames, timeout=15.0):
            failures.append('turning Loop on again did not resume the clip')
        report['loop_on_again'] = session.report()

        # Stopping while parked must not wait on a stream that already ended.
        session.set_loop(False)
        if not wait(session, lambda: session.finished, timeout=25.0):
            failures.append('the parked clip did not report its end before stopping')
        began = time.monotonic()
        session.stop()
        report['stop_seconds_when_parked'] = round(time.monotonic() - began, 2)
        if report['stop_seconds_when_parked'] > 3.0:
            failures.append('stopping a parked clip waited on the finished stream')
        session = None
    finally:
        if session is not None:
            session.stop()
        server.shutdown()
        server.server_close()
    report['pass'] = not failures
    print('PROBE-LINK-PLAYBACK ' + json.dumps(report, ensure_ascii=False))
    shutil.rmtree(HERE, ignore_errors=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
