"""Reproduce: live link preview + Find person, then report what the 3D view got."""
from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
import http.server
import socketserver
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
# The GUI packages (PySide6, yt-dlp) live in the gui overlay, as launch_source loads them.
sys.path.insert(0, str(APP / 'runtime_gui'))
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE='1',
                  HF_HOME=str(APP / 'cache/huggingface'), TORCH_HOME=str(APP / 'cache/torch'),
                  XDG_CACHE_HOME=str(APP / 'cache'))

CLIP = Path(os.environ.get('PROBE_CLIP') or '/c/tmp/live-test.mp4')
REPORT = APP.parent / 'screenshots/live-url-preview-report.json'
report = {'clip': str(CLIP)}


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(CLIP.parent))
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    httpd, port = serve()
    url = f'http://127.0.0.1:{port}/{CLIP.name}'
    report['url'] = url
    from mocap.gui import MainWindow
    window = MainWindow(None)
    window.resize(1200, 760)
    window.show()
    app.processEvents()

    window.live_from_url(url)
    report['status_after_preview'] = window.status.text()
    report['download_enabled'] = window.source_download.isEnabled()
    report['source_url'] = getattr(window, 'source_url', '')
    session = getattr(window, 'camera_session', None)
    report['session'] = None if session is None else dict(
        kind=type(session).__name__, running=bool(session.running), frames=int(session.frames),
        preview=[session.preview_width, session.preview_height],
        size=[session.width, session.height])
    if session is None:
        report['pass'] = False
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('PROBE', json.dumps(report), flush=True)
        return 1

    # The real Find person path, with a 1 s countdown instead of 20.
    window.start_live_preview(delay=1.0)
    deadline = time.time() + 420
    steps = []
    grabbed = False
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.1)
        channel = getattr(window, 'live_channel', None)
        view = window.motion_view
        snapshot = dict(
            t=round(time.time() - (deadline - 420), 1),
            state=None if channel is None else channel.state(),
            count=0 if channel is None else channel.count(),
            mesh=0 if channel is None else channel.mesh_count(),
            joints=view.joints is not None,
            live_mode=bool(view.live_body_mode),
            mesh_frames=view.mesh_frames is not None,
            mesh_asset=view.mesh_asset is not None,
            mesh_binding=view.mesh_binding is not None,
            error=view.mesh_error,
            status=window.status.text()[:80],
            filename=window.filename.text()[:90],
        )
        if not steps or steps[-1]['mesh'] != snapshot['mesh'] or steps[-1]['joints'] != snapshot['joints']:
            steps.append(snapshot)
        if snapshot['mesh'] >= 3 and not grabbed:
            app.processEvents()
            image = window.motion_view.grab()
            target = APP.parent / 'screenshots/live-url-character.png'
            image.save(str(target))
            report['screenshot'] = str(target)
            grabbed = True
            steps.append(dict(snapshot, grabbed=True))
            # The clip is 8 s long and the model takes longer than that: the
            # stream must keep playing so the character keeps moving.
            first_frames, first_count = session.frames, channel.count()
            until = time.time() + 20
            while time.time() < until:
                app.processEvents()
                time.sleep(0.1)
            report['looped'] = dict(frames_before=first_frames, frames_after=session.frames,
                                    count_before=first_count, count_after=channel.count(),
                                    still_running=bool(session.running))
            break
        if channel is not None and channel.state() in ('failed', 2):
            break
    report['steps'] = steps[-8:]
    log = APP / 'cache/live-worker.log'
    report['worker_log'] = log.read_text(encoding='utf-8', errors='replace')[-800:] if log.is_file() else ''
    view = window.motion_view
    report['final'] = dict(joints=view.joints is not None,
                           mesh_frames=view.mesh_frames is not None,
                           mesh_asset=view.mesh_asset is not None,
                           mesh_binding=view.mesh_binding is not None,
                           mesh_error=view.mesh_error,
                           presented=getattr(view, 'mesh_presented_index', None))
    looped = report.get('looped') or {}
    report['pass'] = bool(report['final']['joints'] and report['final']['mesh_frames']
                          and report['final']['mesh_asset']
                          and looped.get('frames_after', 0) > looped.get('frames_before', 0)
                          and looped.get('count_after', 0) > looped.get('count_before', 0))
    window.stop_live_camera()
    app.processEvents()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PROBE', json.dumps(report), flush=True)
    httpd.shutdown()
    os._exit(0 if report['pass'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
