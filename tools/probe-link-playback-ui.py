"""Check the link playback strip in the real window.

  runtime/python.exe -B tools/probe-link-playback-ui.py

Previews a link in the actual workspace and reads the controls the user sees:
the strip appears only for a link, the playhead follows the stream, Loop
toggles the session, and releasing the slider reopens the stream at that second.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
BASE = APP.parent
sys.path.insert(0, str(APP))
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE='1',
                  HF_HOME=str(APP / 'cache/huggingface'), TORCH_HOME=str(APP / 'cache/torch'),
                  XDG_CACHE_HOME=str(APP / 'cache'))

HERE = APP / 'cache' / 'probe-link-playback-ui'
CLIP_SECONDS = 12
report = {'failures': []}


def pump(app, seconds, until=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.03)
    return until() if until is not None else True


def make_clip():
    HERE.mkdir(parents=True, exist_ok=True)
    clip = HERE / 'clip.mp4'
    from mocap.video import executable
    subprocess.run([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=25',
                    '-t', str(CLIP_SECONDS), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(clip)], check=True,
                   creationflags=0x08000000)
    return clip


def main():
    from mocap.runtime_registry import activate

    activate(APP, 'gui')
    from PySide6.QtWidgets import QApplication
    from mocap.gui import MainWindow
    from mocap.live_url import UrlSession

    clip = make_clip()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.RequestHandlerClass.log_message = lambda *a, **k: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d/clip.mp4' % server.server_address[1]
    failures = report['failures']
    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1200, 700)
    window.show()
    session = None
    try:
        app.processEvents()
        report['strip_before_any_source'] = window.link_playback.isVisible()
        if window.link_playback.isVisible():
            failures.append('the link strip showed before a link was previewed')

        session = UrlSession(url)
        session.start_preview()
        if not pump(app, 25.0, lambda: session.frames >= 3):
            failures.append('no frames came from the link')
        window.start_live_camera(session, record=False)
        pump(app, 1.0)
        report['visible'] = window.link_playback.isVisible()
        report['slider_enabled'] = window.link_position.isEnabled()
        report['length'] = window.link_time.text()
        if not window.link_playback.isVisible():
            failures.append('the link strip stayed hidden for a link preview')
        if not window.link_position.isEnabled():
            failures.append('the position slider was disabled for a clip of known length')
        if '/' not in window.link_time.text():
            failures.append('the strip did not show the clip length')

        # The playhead follows the stream.
        pump(app, 2.5)
        first = window.link_position.value()
        report['slider_after_2s'] = first
        if first <= 0:
            failures.append('the playhead did not move with the stream')

        # Loop is a real toggle on the session.
        window.link_loop.setChecked(False)
        app.processEvents()
        report['loop_after_uncheck'] = session.loop
        if session.loop:
            failures.append('the Loop box did not turn looping off')
        window.link_loop.setChecked(True)
        app.processEvents()
        if not session.loop:
            failures.append('the Loop box did not turn looping back on')

        # Releasing the slider reopens the stream at that second.
        frames = session.frames
        window.link_position.setValue(500)
        window.link_position_released()
        if not pump(app, 30.0, lambda: session.frames > frames and session.position() >= 5.4):
            failures.append('releasing the slider did not reopen the stream at that second')
        report['position_after_release'] = round(session.position(), 2)
        report['time_after_release'] = window.link_time.text()

        window.stop_live_camera()
        pump(app, 1.0)
        report['visible_after_stop'] = window.link_playback.isVisible()
        if window.link_playback.isVisible():
            failures.append('the link strip stayed visible after stopping the link')
    finally:
        if session is not None:
            session.stop()
        window.close()
        server.shutdown()
        server.server_close()
        shutil.rmtree(HERE, ignore_errors=True)
    report['pass'] = not failures
    print('PROBE-LINK-PLAYBACK-UI ' + json.dumps(report, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
