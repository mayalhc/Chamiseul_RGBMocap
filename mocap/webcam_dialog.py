"""Live webcam preview and capture.

The preview and the clip come from one capture session: Windows gives a camera
to a single process, so opening it once and splitting the frames is the only way
to see the live image while it records.

Two modes share that session:

* **Single clip** - Record stops after the chosen length and the clip is opened
  in the workspace.
* **Near-live windows** - Record keeps taking clip after clip of the chosen
  length, so a rehearsal leaves a queue of short takes instead of one long file
  that has to be cut first.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QListWidget, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout)

from . import camera_capture
from .paths import ROOT


class WebcamDialog(QDialog):
    def __init__(self, parent=None, on_window=None):
        super().__init__(parent)
        self.setWindowTitle('Live Webcam')
        self.recorded = None
        self.live_session = None
        self.on_window = on_window
        self.clips = []
        self._enumeration = {}
        self._closed = False
        self._session = None
        self._recording = False
        self._window_started = 0.0
        self._window_seconds = 10

        self.preview = QLabel('Camera preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet('background:#111; color:#777; border:1px solid #2a2a2a;')

        self.device = QComboBox()
        self.resolution = QComboBox()
        self.fps = QComboBox()
        self.rotation = QComboBox()
        for label, value in (('0°', 0), ('90° clockwise', 90),
                             ('90° counter-clockwise', 270), ('180°', 180)):
            self.rotation.addItem(label, value)
        self.rotation.setCurrentIndex(1)
        self.rotation.setToolTip('Turn the capture upright. Stand the camera on its '
                                 'side and pick the 90° that makes the picture level; '
                                 'the view, the live character and the recorded windows '
                                 'all follow.')
        self.seconds = QSpinBox()
        self.seconds.setRange(1, 300)
        self.seconds.setValue(10)
        self.seconds.setSuffix(' s')
        self.keep_windows = QCheckBox('Keep taking windows (near-live)')
        self.keep_windows.setToolTip('Record one clip after another of the length above. '
                                     'Each finished window is listed here; open the last one '
                                     'in the workspace to reconstruct it.')
        self.auto_process = QCheckBox('Load each finished window automatically')
        self.auto_process.setChecked(True)
        self.auto_process.setToolTip('When a window finishes, it is loaded into the '
                                     'workspace right away - the next window records '
                                     'while this one is being processed.')
        self.refresh = QPushButton('Refresh cameras')
        self.refresh.clicked.connect(self.find_devices)

        device_row = QHBoxLayout()
        device_row.addWidget(self.device, 1)
        device_row.addWidget(self.refresh)
        form = QFormLayout()
        form.addRow('Camera', device_row)
        form.addRow('Resolution', self.resolution)
        form.addRow('Frame rate', self.fps)
        form.addRow('Rotation', self.rotation)
        form.addRow('Window', self.seconds)
        form.addRow('', self.keep_windows)
        form.addRow('', self.auto_process)

        self.clip_list = QListWidget()
        self.clip_list.setMaximumHeight(90)
        self.clip_list.itemDoubleClicked.connect(self.open_selected)

        self.status = QLabel('Looking for cameras...')
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)

        self.buttons = QDialogButtonBox()
        self.record_button = self.buttons.addButton('Record', QDialogButtonBox.AcceptRole)
        self.stop_button = self.buttons.addButton('Stop', QDialogButtonBox.RejectRole)
        self.open_button = self.buttons.addButton('Open last clip', QDialogButtonBox.ActionRole)
        self.live_source_button = self.buttons.addButton('Start live',
                                                         QDialogButtonBox.ActionRole)
        self.live_source_button.setToolTip('Hand the camera to the workspace and start '
                                          'the live capture. Find person then checks the '
                                          'character, and Record window keeps a take.')
        self.buttons.addButton(QDialogButtonBox.Close)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.live_source_button.setEnabled(False)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.open_button.clicked.connect(self.open_last)
        self.live_source_button.clicked.connect(self.use_as_live)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addLayout(form)
        layout.addWidget(self.clip_list)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        self.device.currentIndexChanged.connect(self.find_formats)
        QTimer.singleShot(0, self.find_devices)

    # -- camera list ------------------------------------------------------
    def find_devices(self):
        self.release_preview()
        self.refresh.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status.setText('Looking for cameras...')
        self.device.clear()
        self.resolution.clear()
        self.fps.clear()

        def work():
            self._enumeration['devices'] = camera_capture.devices()

        threading.Thread(target=work, name='rgbmocap-webcam-devices', daemon=True).start()
        QTimer.singleShot(150, self.poll_devices)

    def poll_devices(self):
        if self._closed:
            return
        if 'devices' not in self._enumeration:
            QTimer.singleShot(150, self.poll_devices)
            return
        found = self._enumeration.pop('devices')
        self.refresh.setEnabled(True)
        if not found:
            self.status.setText('No camera was found. Connect one, close other apps that use '
                                'it (Discord, Teams, OBS), then press Refresh cameras.')
            return
        self.device.blockSignals(True)
        for name in found:
            self.device.addItem(name, name)
        self.device.blockSignals(False)
        self.find_formats()

    def find_formats(self, *_args):
        device = self.device.currentData()
        self.resolution.clear()
        self.fps.clear()
        self.record_button.setEnabled(False)
        if not device:
            return
        self.status.setText('Reading what this camera supports...')

        def work():
            self._enumeration['formats'] = camera_capture.formats(device)

        threading.Thread(target=work, name='rgbmocap-webcam-formats', daemon=True).start()
        QTimer.singleShot(150, self.poll_formats)

    def poll_formats(self):
        if self._closed:
            return
        if 'formats' not in self._enumeration:
            QTimer.singleShot(150, self.poll_formats)
            return
        found = self._enumeration.pop('formats')
        if not found:
            self.status.setText('This camera reported no capture format.')
            return
        for entry in found:
            self.resolution.addItem('%d x %d' % (entry['width'], entry['height']),
                                    (entry['width'], entry['height']))
            self.fps.addItem('%d fps' % round(entry['fps']), round(entry['fps']))
        preferred = -1
        for index in range(self.resolution.count()):
            data = self.resolution.itemData(index)
            if data and tuple(data) == (1920, 1080):
                preferred = index
                break
        if preferred >= 0:
            self.resolution.setCurrentIndex(preferred)
        self.record_button.setEnabled(True)
        self.live_source_button.setEnabled(True)
        self.status.setText('Live preview started. Record, or use it as the live source.')
        self.start_preview()

    # -- preview ----------------------------------------------------------
    def start_preview(self):
        device = self.device.currentData()
        size = self.resolution.currentData()
        fps = self.fps.currentData()
        if not device or not size or not fps:
            return
        self.release_preview()
        session = camera_capture.Session(
            camera_capture.default_path(ROOT), device, size[0], size[1], fps,
            preview_width=min(960, size[0]), rotation=self.rotation.currentData())
        try:
            session.start_preview()
        except Exception as error:          # noqa: BLE001 - shown to the user
            self.status.setText('Preview failed: ' + str(error))
            return
        self._session = session

    def release_preview(self):
        session, self._session = self._session, None
        if session is not None and not self._recording:
            session.stop()

    def tick(self):
        if self._closed:
            return
        session = self._session
        if session is not None:
            frame = session.snapshot()
            if frame is not None:
                height, width, _ = frame.shape
                image = QImage(np.ascontiguousarray(frame).data, width, height,
                               3 * width, QImage.Format_BGR888)
                self.preview.setPixmap(QPixmap.fromImage(image.copy()).scaled(
                    self.preview.width(), self.preview.height(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
            if self._recording:
                if session.started and not session.running and session.frames == 0:
                    # ffmpeg died on open (the camera held by another app):
                    # say so now instead of waiting out the whole window.
                    self.finish_window()
                    return
                elapsed = session.elapsed()
                self.progress.setValue(min(1000, int(1000 * elapsed / max(1.0, self._window_seconds))))
                if elapsed >= self._window_seconds:
                    self.finish_window()

    # -- recording --------------------------------------------------------
    def start_recording(self):
        device = self.device.currentData()
        size = self.resolution.currentData()
        fps = self.fps.currentData()
        if not device or not size or not fps:
            return
        self.release_preview()
        self._window_seconds = float(self.seconds.value())
        # The camera is being handed over from the preview: Windows keeps it
        # busy for a beat after the previous ffmpeg exits, so the first open
        # can come up empty. Detect a capture that died at once and retry.
        session = None
        for attempt in range(3):
            session = camera_capture.Session(
                camera_capture.default_path(ROOT), device, size[0], size[1], fps,
                preview_width=min(960, size[0]), rotation=self.rotation.currentData())
            try:
                session.start()
            except Exception as error:      # noqa: BLE001 - shown to the user
                self.status.setText('Recording failed: ' + str(error))
                return
            time.sleep(1.2)
            if session.running:
                break
            session.stop()
            session = None
            time.sleep(0.4)
        if session is None:
            self.status.setText('The camera would not open for recording. '
                                'Close other apps that use it and try again.')
            return
        self._session = session
        self._recording = True
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.keep_windows.setEnabled(False)
        self.seconds.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText('Recording... move freely.')

    def finish_window(self):
        session, self._session = self._session, None
        if session is None:
            return
        self._recording = False
        report = session.stop()
        path = report.get('path')
        if path and Path(path).is_file() and report.get('bytes'):
            self.clips.append(report)
            self.clip_list.addItem('%s  -  %.1f fps, %.1f s'
                                   % (Path(path).name, report.get('fps', 0), report.get('seconds', 0)))
            self.clip_list.setCurrentRow(self.clip_list.count() - 1)
            self.recorded = path
            self.open_button.setEnabled(True)
            self.status.setText('Saved %s (%.1f fps, %.1f s)'
                                % (Path(path).name, report.get('fps', 0), report.get('seconds', 0)))
        else:
            self.status.setText('The camera produced no frames. Close other apps that use it.')
        self.progress.setValue(0)

        if self.on_window is not None and self.auto_process.isChecked()                 and self.keep_windows.isChecked() and path:
            try:
                self.on_window(Path(path))
            except Exception as error:      # noqa: BLE001 - the recording stands
                self.status.setText('Auto-load failed: ' + str(error))
        if self.keep_windows.isChecked() and not self._closed:
            self.start_recording()
            return
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.keep_windows.setEnabled(True)
        self.seconds.setEnabled(True)

        if not self.keep_windows.isChecked() and self.recorded:
            self.accept()

    def stop_recording(self):
        if self._recording:
            self.finish_window()

    # -- opening ----------------------------------------------------------
    def open_selected(self, item):
        row = self.clip_list.row(item)
        if 0 <= row < len(self.clips):
            self.recorded = self.clips[row]['path']
            self.accept()

    def open_last(self):
        if self.clips:
            self.recorded = self.clips[-1]['path']
            self.accept()

    def use_as_live(self):
        """Hand the running preview session to the workspace: it goes live
        in the video view, and the File menu records windows from it."""
        session, self._session = self._session, None
        self.recorded = None
        self.live_session = session
        self._closed = True
        self.timer.stop()
        super().done(QDialog.Accepted)

    def done(self, result):
        # Polling and preview timers outlive the widgets once the dialog closes;
        # a tick after close must not paint into a deleted label.
        self._closed = True
        self.timer.stop()
        self._release_session()
        super().done(result)

    def _release_session(self):
        session, self._session = self._session, None
        self._recording = False
        if session is not None:
            try:
                session.stop()
            except Exception:               # noqa: BLE001 - closing anyway
                pass

    def reject(self):
        self._closed = True
        self.timer.stop()
        self._release_session()
        super().reject()
