"""Check the app describes the character on screen before sending poses.

  runtime/python.exe -B tools/verify-live-character-app.py

Turns on the live link in the real window with a character take in the scene and
reads the first message. It must name the character's rest file, palette and
placement (so Blender can import it), and no pose packet may precede it. The
Blender side, including bone playback, is checked by
tools/verify-live-character.py.
"""
from __future__ import annotations

import json
import os
import socket
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

CHARACTER = APP / ('projects/Download_35b14c2294807c_project/characters'
                   '/1d86e8a4c3dd4e8f8dd981709cc4e466')
ANIMATION = 'animation-988febee65e14e66a4ce17ed082dd3f7'
TAKE = CHARACTER / ANIMATION / 'take-0'
PORT = 9236
os.environ['RGB_MOCAP_LIVE_PORT'] = str(PORT)
REPORT = BASE / 'screenshots/live-character-app-report.json'
report = {'character': str(CHARACTER), 'port': PORT}
handshake = {}
packets = []


def read_client(stop):
    try:
        client = socket.create_connection(('127.0.0.1', PORT), timeout=5)
    except OSError as error:
        report['client_error'] = str(error)
        return
    client.settimeout(0.5)
    buffer = b''
    try:
        while b'\n' not in buffer and not stop.is_set():
            try:
                buffer += client.recv(65536)
            except socket.timeout:
                continue
        line, _, buffer = buffer.partition(b'\n')
        if line:
            handshake.update(json.loads(line.decode('utf-8')))
        report['bytes_before_pose'] = len(buffer)
        while not stop.is_set():
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                continue
            if chunk:
                packets.append(len(chunk))
    finally:
        client.close()


def main():
    from mocap.runtime_registry import activate

    activate(APP, 'gui')
    from PySide6.QtWidgets import QApplication
    from mocap.gui import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(900, 600)
    window.show()
    # A character take, exactly as the character list stores it.
    window.character_root = APP / 'projects/Download_35b14c2294807c_project'
    # Character paths in the index are relative to the project's characters folder.
    window.characters = [dict(id='check', name='Person 1', asset=CHARACTER.name,
                              animation=CHARACTER.name + '/' + ANIMATION, visible=True,
                              takes=[dict(folder='take-0', start=0, frames=822, fps=30,
                                          transform=None)])]
    report['characters'] = [dict(name=c['name'], takes=len(c['takes'])) for c in window.characters]
    stop = threading.Event()
    thread = threading.Thread(target=read_client, args=(stop,), daemon=True)
    thread.start()
    time.sleep(0.3)
    window.live_link_action.setChecked(True)
    report['status_after_enable'] = window.status.text()
    deadline = time.time() + 6.0
    while time.time() < deadline and not handshake:
        app.processEvents()
        time.sleep(0.05)
    report['handshake'] = dict(kind=handshake.get('kind'), version=handshake.get('version'),
                               rest_only=handshake.get('rest_only'),
                               blend=handshake.get('blend'),
                               blend_exists=Path(str(handshake.get('blend') or '')).is_file(),
                               armature=handshake.get('armature'),
                               palette=len(handshake.get('palette') or []),
                               height=handshake.get('height'), label=handshake.get('label'),
                               transform=handshake.get('transform'))
    report['link_status'] = window.live_link.status() if getattr(window, 'live_link', None) else None
    try:
        report['log'] = window.log.toPlainText()[-600:]
    except AttributeError:
        report['log'] = str(getattr(window, 'log', ''))[-600:]
    window.live_link_action.setChecked(False)
    stop.set()
    thread.join(2)
    # The pose packet must not be squashed in with the description.
    report['posted_pose_with_description'] = bool(report.get('bytes_before_pose'))
    ok = (report['handshake']['kind'] == 'character' and report['handshake']['rest_only'] is True
          and report['handshake']['blend_exists'] and report['handshake']['palette'] > 0
          and report['handshake']['armature'] and not report['posted_pose_with_description'])
    report['pass'] = bool(ok)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFY-LIVE-CHARACTER-APP', json.dumps(report), flush=True)
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    raise SystemExit(main())
