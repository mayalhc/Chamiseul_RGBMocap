# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: MIT (see LICENSE in the repository root)
"""Check the app side of the Cascadeur live link with the real window.

  runtime/python.exe -B tools/verify-cascadeur-app.py [project] [seconds]

Opens a take, enables File > Live Link to Cascadeur, and presents a run of
frames. The link is real: the receiver is started inside a running Cascadeur,
so Cascadeur must be open (a character in the scene is needed for anything to
be written, and the report says how many joints bound). Verifies that the menu
action connects, that frames actually leave the app, that Cascadeur reports
writing them, and that unchecking the action stops the receiver.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
BASE = APP.parent
sys.path.insert(0, str(APP))
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(APP),
                  PYTHONDONTWRITEBYTECODE='1',
                  HF_HOME=str(APP / 'cache/huggingface'),
                  TORCH_HOME=str(APP / 'cache/torch'),
                  XDG_CACHE_HOME=str(APP / 'cache'))

DEFAULT_PROJECT = APP / 'projects/Download_35b14c2294807c_project'

from mocap.runtime_registry import activate  # noqa: E402

activate(APP, 'gui')

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from mocap.gui import MainWindow  # noqa: E402

PROJECT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECT
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
REPORT = BASE / 'screenshots/cascadeur-app-report.json'
report = {'project': str(PROJECT)}


def main():
    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1200, 800)
    window.load_project(PROJECT)
    window.show()
    state = {'waited': 0.0, 'frames': []}

    def finish(code, error=''):
        if error:
            report['error'] = error
        action = getattr(window, 'cascadeur_link_action', None)
        if action is not None and action.isChecked():
            action.setChecked(False)
            app.processEvents()
        link = getattr(window, 'cascadeur_link', None)
        report['final'] = {'active': bool(link and link.active),
                           'status': window.status.text()}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=1)[:2500])
        app.exit(code)

    def wait_for_take():
        state['waited'] += 0.25
        if window.info is None or not getattr(window, 'video_readable', False):
            if state['waited'] > 30:
                return finish(1, 'project video never became readable')
            return QTimer.singleShot(250, wait_for_take)
        enable()

    def enable():
        action = getattr(window, 'cascadeur_link_action', None)
        report['menu_action'] = dict(present=action is not None,
                                     text=action.text() if action else None,
                                     checkable=bool(action.isCheckable()) if action else None)
        if action is None:
            return finish(1, 'menu action missing')
        report['action_tooltip'] = action.toolTip()[:120]
        action.setChecked(True)
        report['status_after_enable'] = window.status.text()
        QTimer.singleShot(300, wait_for_link)

    def wait_for_link():
        state['waited'] += 0.3
        link = getattr(window, 'cascadeur_link', None)
        text = window.status.text()
        if link is not None and link.active:
            report['status_linked'] = text
            return stream()
        if text.startswith('Cascadeur:') and not text.startswith('Cascadeur: delayed'):
            # A refusal is a real answer, not something to wait out.
            return finish(1, text)
        if state['waited'] > 40:
            return finish(1, 'link never became active: ' + text)
        QTimer.singleShot(300, wait_for_link)

    def stream():
        link = window.cascadeur_link
        report['start'] = {'names': len(link.names), 'root': link.root,
                           'body_joints': len(link.body_index)}
        start_frame = getattr(window.motion_view, 'index', 0) or 0
        count = max(8, int(SECONDS * 20))
        for step in range(count):
            frame = start_frame + step
            window.present_frame(frame)
            app.processEvents()
            time.sleep(0.05)
            state['frames'].append(frame)
        QTimer.singleShot(700, check)

    def check():
        link = window.cascadeur_link
        receiver = link.status() or {}
        report['presented'] = len(state['frames'])
        report['app_sent'] = link.sent
        report['receiver'] = {k: receiver.get(k) for k in
                              ('version', 'bound', 'written', 'dropped',
                               'write_ms', 'errors', 'at')}
        report['link_error'] = link.error
        ok = (link.sent > 0 and receiver.get('written', 0) > 0
              and receiver.get('bound', 0) > 0)
        if not ok:
            return finish(1, 'nothing was written: %s' % (report['receiver'],))
        disable()

    def disable():
        action = window.cascadeur_link_action
        action.setChecked(False)
        app.processEvents()
        time.sleep(0.8)
        link = getattr(window, 'cascadeur_link', None)
        report['stopped'] = {'active': bool(link and link.active),
                             'receiver': (link.status() if link else None),
                             'status': window.status.text()}
        ok = not (link and link.active)
        finish(0 if ok else 1, '' if ok else 'receiver still running after uncheck')

    QTimer.singleShot(250, wait_for_take)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
