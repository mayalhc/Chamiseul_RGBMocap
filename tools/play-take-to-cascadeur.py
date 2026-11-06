# Copyright (c) 2026 Chamiseul. All rights reserved.
# Proprietary: redistribution prohibited. See LICENSE-CHAMISEUL.md in the repository root.
"""Stream the whole take through the real app window into Cascadeur.

  runtime/python.exe -B tools/play-take-to-cascadeur.py [project] [fps]

Opens the app's real window, loads the project, enables File > Live Link to
Cascadeur through its actual menu action, and presents every frame of the
take at the given rate (30 fps default). The receiver inside Cascadeur keys
each frame where it belongs on the timeline, so when the run ends the whole
take is animated there and can be played in Cascadeur itself.

The window is left open with the link checked: from then on it is the
user's session - play here and Cascadeur follows.
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

from mocap.runtime_registry import activate  # noqa: E402

activate(APP, 'gui')

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from mocap.gui import MainWindow  # noqa: E402

PROJECT = (Path(sys.argv[1]) if len(sys.argv) > 1 else
           APP / 'projects/Download (2)_23dd0cf4a379e7_project')
FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
REPORT = BASE / 'screenshots/cascadeur-full-take-report.json'
report = {'project': str(PROJECT), 'fps': FPS}


def main():
    application = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1200, 800)
    window.load_project(PROJECT)
    window.show()
    state = {'waited': 0.0, 'frame': 0, 'frames': [], 'total': 0,
             'interval': 1.0 / FPS, 'deadline': 0.0}

    def save_report():
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def finish(code, error=''):
        if error:
            report['error'] = error
        save_report()
        print(json.dumps(report, indent=1)[:2000])
        if error:
            application.exit(code)

    def wait_for_take():
        state['waited'] += 0.25
        if window.info is None or not getattr(window, 'video_readable', False):
            if state['waited'] > 60:
                return finish(1, 'project video never became readable')
            return QTimer.singleShot(250, wait_for_take)
        enable()

    def enable():
        action = getattr(window, 'cascadeur_link_action', None)
        report['menu_action'] = action is not None
        if action is None:
            return finish(1, 'menu action missing')
        action.setChecked(True)
        QTimer.singleShot(300, wait_for_link)

    def wait_for_link():
        state['waited'] += 0.3
        link = getattr(window, 'cascadeur_link', None)
        text = window.status.text()
        if link is not None and link.active:
            return stream()
        if text.startswith('Cascadeur:') and 'delayed' not in text:
            return finish(1, text)
        if state['waited'] > 60:
            return finish(1, 'link never became active: ' + text)
        QTimer.singleShot(300, wait_for_link)

    def stream():
        link = window.cascadeur_link
        report['start'] = {'names': len(link.names),
                           'body_joints': len(link.body_index)}
        state['total'] = int(link.body.count)
        report['take_frames'] = state['total']
        state['deadline'] = time.time()
        QTimer.singleShot(0, step)

    def step():
        link = window.cascadeur_link
        if link is None or not link.active:
            return finish(1, 'link dropped mid-stream')
        # Pace to the writer: never run more than a few frames ahead, or the
        # receiver's bounded queue drops the oldest frames and the take ends
        # up with holes that hold stale poses.
        receiver = link.status() or {}
        if (receiver.get('queued', 0) > 4
                or receiver.get('written', 0) + receiver.get('dropped', 0)
                < state['frame']):
            return QTimer.singleShot(20, step)
        window.present_frame(state['frame'])
        application.processEvents()
        state['frames'].append(state['frame'])
        state['frame'] += 1
        if state['frame'] >= state['total']:
            return QTimer.singleShot(1500, check)
        delay = state['deadline'] + state['interval'] - time.time()
        state['deadline'] += state['interval']
        if delay < -2.0:            # fell behind: catch up, keep going
            state['deadline'] = time.time()
            delay = 0.0
        QTimer.singleShot(max(0, int(delay * 1000)), step)

    def check():
        # The writer inside Cascadeur commits at its own pace (a scene write
        # costs more than a frame at 30 fps), so wait for the queue to drain
        # before judging: presented frames must all be written or dropped.
        state['drained'] = 0.0
        poll()

    def poll():
        state['drained'] += 1.0
        link = window.cascadeur_link
        receiver = (link.status() or {}) if link else {}
        written = receiver.get('written', 0)
        dropped = receiver.get('dropped', 0)
        if written + dropped < state['total'] - 2 and state['drained'] < 420:
            window.status.setText('Cascadeur: keying %d / %d frames...'
                                  % (written, state['total']))
            return QTimer.singleShot(1000, poll)
        report['presented'] = len(state['frames'])
        report['app_sent'] = link.sent
        report['receiver'] = {key: receiver.get(key) for key in
                              ('version', 'bound', 'written', 'dropped',
                               'write_ms', 'errors', 'at')}
        report['ok'] = bool(written >= state['total'] - 4
                            and dropped <= 4
                            and not receiver.get('errors'))
        save_report()
        print(json.dumps(report, indent=1)[:2000])
        if not report['ok']:
            return finish(1, 'receiver missed frames: %s' % report['receiver'])
        window.status.setText(
            'Cascadeur: full take keyed (%d frames written, %d dropped). '
            'This window stays open - play freely, Cascadeur follows.'
            % (written, dropped))
        # The window stays open with the link checked: the session is the
        # user's now. Closing the window ends this process normally.

    QTimer.singleShot(250, wait_for_take)
    return application.exec()


if __name__ == '__main__':
    raise SystemExit(main())
