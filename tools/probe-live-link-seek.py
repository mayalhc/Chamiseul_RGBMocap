"""Check the live character keeps moving across a link seek, with the real model.

  runtime/python.exe -B tools/probe-live-link-seek.py

Runs the actual path - a link previewed as the live source, Find person, the
body model solving every frame - then moves the playhead with the window's own
slider. The character must keep receiving solved frames and the 3D view must
keep painting new poses, not freeze on the frame it was showing.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'runtime_gui'))
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE='1',
                  HF_HOME=str(APP / 'cache/huggingface'), TORCH_HOME=str(APP / 'cache/torch'),
                  XDG_CACHE_HOME=str(APP / 'cache'))

CLIP = Path(os.environ.get('PROBE_CLIP') or 'C:/tmp/live-test.mp4')
REPORT = APP.parent / 'screenshots/live-link-seek-report.json'
report = {'clip': str(CLIP), 'failures': []}


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(CLIP.parent))
    handler.log_message = lambda *a, **k: None
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    httpd, port = serve()
    url = 'http://127.0.0.1:%d/%s' % (port, CLIP.name)
    report['url'] = url
    from mocap.gui import MainWindow
    window = MainWindow(None)
    window.resize(1200, 760)
    window.show()
    app.processEvents()

    failures = report['failures']
    window.live_from_url(url)
    session = getattr(window, 'camera_session', None)
    if session is None:
        report['pass'] = False
        failures.append('the link did not become the live source')
        print('PROBE-LIVE-LINK-SEEK ' + json.dumps(report, ensure_ascii=False), flush=True)
        os._exit(1)
    report['duration'] = round(session.duration, 2)
    report['controls_visible'] = window.link_playback.isVisible()

    def pose_digest(view):
        """The pose the 3D view is actually drawing, as a short fingerprint."""
        joints = getattr(view, 'joints', None)
        if joints is None:
            return None
        try:
            rounded = np.round(np.asarray(joints, dtype='float32'), 3)
            return hashlib.md5(rounded.tobytes()).hexdigest()[:10]
        except (TypeError, ValueError):
            return None

    def counters():
        channel = getattr(window, 'live_channel', None)
        view = window.motion_view
        return dict(frames=int(session.frames),
                    rig=0 if channel is None else int(channel.count()),
                    mesh=0 if channel is None else int(channel.mesh_count()),
                    presented=getattr(view, 'mesh_presented_index', None),
                    pose=pose_digest(view),
                    position=round(session.position(), 2),
                    state=None if channel is None else channel.state(),
                    status=window.status.text()[:70])

    def wait_until(predicate, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.processEvents()
            time.sleep(0.08)
            if predicate():
                return True
        return False

    def move_to(fraction):
        window.link_position.setValue(int(fraction * 1000))
        window.link_position_released()
        app.processEvents()

    try:
        window.start_live_preview(delay=1.0)
        # The model has to load first; that is the one long wait.
        live = wait_until(lambda: (getattr(window, 'live_channel', None) is not None
                                   and window.live_channel.mesh_count() >= 3), 600.0)
        report['character_live'] = live
        if not live:
            failures.append('the character never came up for the link')
            raise SystemExit
        report['before'] = counters()

        # Baseline: the character is moving on its own before any seek.
        base = counters()
        time.sleep(0.1)
        wait_until(lambda: counters()['mesh'] >= base['mesh'] + 2, 30.0)
        report['baseline_motion'] = dict(rig=counters()['rig'] - base['rig'],
                                         mesh=counters()['mesh'] - base['mesh'],
                                         presented_changed=counters()['pose'] != base['pose'])
        if report['baseline_motion']['mesh'] <= 0:
            failures.append('the character was not moving before the seek')
        if not report['baseline_motion']['presented_changed']:
            failures.append('the 3D view did not repaint a new pose before the seek')

        # Move the playhead to the last quarter of the clip through the window's
        # own control, then watch the character for a while.
        before_seek = counters()
        asked = time.monotonic()
        target = session.duration * 0.75
        move_to(0.75)
        report['seek_requested'] = round(target, 2)
        # The pass in flight can still deliver a frame or two, so the interruption
        # is measured from the restart itself to the first frame of the reopened
        # stream, not from the request to any frame.
        if not wait_until(lambda: abs(getattr(session, '_start_at', 0.0) - target) < 0.2, 30.0):
            failures.append('the stream never reopened at the requested second')
        at_restart = session.frames
        if not wait_until(lambda: session.frames > at_restart, 30.0):
            failures.append('no frame came from the reopened stream')
        report['reopen_seconds'] = round(time.monotonic() - asked, 2)
        time.sleep(1.0)
        step = counters()
        wait_until(lambda: counters()['mesh'] >= step['mesh'] + 3, 40.0)
        after_seek = counters()
        report['after_seek'] = after_seek
        report['across_seek'] = dict(rig=after_seek['rig'] - before_seek['rig'],
                                     mesh=after_seek['mesh'] - before_seek['mesh'],
                                     frames=after_seek['frames'] - before_seek['frames'],
                                     presented_changed=after_seek['pose'] != before_seek['pose'])
        if report['across_seek']['frames'] <= 0:
            failures.append('no frames arrived after the seek')
        if report['across_seek']['mesh'] < 3:
            failures.append('the character did not keep getting solved frames after the seek')
        if not report['across_seek']['presented_changed']:
            failures.append('the 3D view stopped repainting after the seek')
        if after_seek['position'] < session.duration * 0.7:
            failures.append('the playhead did not stay at the requested second')

        # A second move back to the start: repeated seeks must not wedge it.
        second = counters()
        asked = time.monotonic()
        move_to(0.0)
        wait_until(lambda: abs(getattr(session, '_start_at', 1.0)) < 0.2, 30.0)
        at_restart = session.frames
        wait_until(lambda: session.frames > at_restart, 30.0)
        reopen = round(time.monotonic() - asked, 2)
        wait_until(lambda: counters()['mesh'] >= second['mesh'] + 3, 45.0)
        third = counters()
        report['second_seek'] = dict(rig=third['rig'] - second['rig'],
                                     mesh=third['mesh'] - second['mesh'],
                                     position=third['position'],
                                     reopen_seconds=reopen,
                                     presented_changed=third['pose'] != second['pose'])
        if reopen > 5.0:
            failures.append('the stream took too long to reopen at the requested second')
        if third['mesh'] - second['mesh'] < 3:
            failures.append('the character stopped after moving the playhead back to the start')
        if not report['second_seek']['presented_changed']:
            failures.append('the 3D view stopped repainting after the second move')
        report['worker_alive'] = bool(getattr(window, 'live_worker', None)
                                      and window.live_worker.poll() is None)
        if not report['worker_alive']:
            failures.append('the live character worker exited during the move')
    finally:
        window.stop_live_camera()
        app.processEvents()
        log = APP / 'cache/live-worker.log'
        report['worker_log_tail'] = log.read_text(encoding='utf-8', errors='replace')[-600:] \
            if log.is_file() else ''
        report['pass'] = not failures
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('PROBE-LIVE-LINK-SEEK ' + json.dumps(report, ensure_ascii=False), flush=True)
        httpd.shutdown()
    os._exit(0 if report.get('pass') else 1)


if __name__ == '__main__':
    main()
