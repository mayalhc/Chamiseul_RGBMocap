"""Check the header Download button keeps a previewed link and opens it."""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'runtime_gui'))
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE='1',
                  HF_HOME=str(APP / 'cache/huggingface'), TORCH_HOME=str(APP / 'cache/torch'),
                  XDG_CACHE_HOME=str(APP / 'cache'))

CLIP = Path(os.environ.get('PROBE_CLIP') or 'C:/tmp/live-test.mp4')
REPORT = APP.parent / 'screenshots/source-download-report.json'


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(CLIP.parent))
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{httpd.server_address[1]}/{CLIP.name}'
    from mocap.gui import MainWindow
    window = MainWindow(None)
    window.resize(1100, 720)
    window.show()
    app.processEvents()
    report = {'url': url, 'before': dict(enabled=window.source_download.isEnabled(),
                                        text=window.source_download.text())}
    # No link yet: the button must refuse instead of guessing a file.
    window.download_source()
    report['without_link_status'] = window.status.text()
    window.source_url = url
    window.source_download.setEnabled(True)
    window.download_source()
    deadline = time.time() + 90
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
        if window.video_path is not None and Path(window.video_path).suffix == CLIP.suffix:
            break
        if not window.source_download.isEnabled() and 'could not be downloaded' in window.status.text():
            break
    report['path'] = str(window.video_path) if window.video_path else ''
    report['status'] = window.status.text()
    report['enabled_after'] = window.source_download.isEnabled()
    report['saved'] = bool(report['path']) and Path(report['path']).is_file()
    report['size'] = Path(report['path']).stat().st_size if report['saved'] else 0
    report['pass'] = bool(report['saved'] and report['size'] == CLIP.stat().st_size
                          and report['enabled_after'])
    window.close()
    httpd.shutdown()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PROBE', json.dumps(report), flush=True)
    if report['saved']:
        Path(report['path']).unlink(missing_ok=True)   # leave no test clip behind
    os._exit(0 if report['pass'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
