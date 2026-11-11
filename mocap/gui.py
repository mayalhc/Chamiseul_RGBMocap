from __future__ import annotations
from .tooltips import tooltip_text
from .ui_refresh import batch_readiness
from .validation_snapshot import read_only_checks

import json
import codecs
import math

import numpy as np
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
import time

from PySide6.QtCore import QElapsedTimer, QEvent, QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QShortcut, QTextOption
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from .models import MODELS_LIST, import_body_model, missing, ready
from .paths import ROOT, PROJECTS, worker_command
from .video import select_preview
from .viewer import VideoView, MotionView
from .playback import playback_frame, PlaybackTimerResolution
from .source_audio import SourceAudio
from .timeline import TimelineSlider
from .preview_data import PreviewDataLoader, project_signature
from .motion_tools import MOTION_TOOLS, RECONSTRUCTION_STAGES, selected_export, RESULTS, result_kind, result_options, result_is_current, base_result, body_result, has_handflow

def live_marker(root):
    """The file through which a preloaded worker learns its live channel."""
    return Path(root) / 'cache' / 'live-warm-channel.txt'


def link_clock(seconds, length=0.0):
    """A playhead for the link strip: m:ss, with the clip length when known."""
    def stamp(value):
        value = max(0, int(value))
        return '%d:%02d' % divmod(value, 60) if value < 3600 else '%d:%02d:%02d' % (
            value // 3600, value % 3600 // 60, value % 60)
    return stamp(seconds) + (' / ' + stamp(length) if length else '')


STAGES = {"prepare": "Prepare Video", "track": "Track Actor", "pva": "Analyze 2D", "gvhmr": "Initial 3D Reconstruction", "refine": "Refine Motion", "export": "Export", "download": "Model Manager", "diagnose": "Check System"}

from .theme import STYLE
STAGES["mesh"] = "Prepare SMPL-X Mesh"
STAGES['rig']='Prepare Bone Preview'
STAGES['save_fbx']='Save FBX'
STAGES['copy_bvh']='Save BVH'
STAGES['save_blend']='Save Blender Project'
STAGES['save_blend_scene']='Save MAMMA Blender Scene'
STAGES["footmr"] = "Refine Ankles / FootMR"
STAGES["handflow"] = "Infer Fingers / HandFlow"
STAGES['correct'] = 'Motion Correction'
from .workspace_ui import build_workspace
from .people_ui import PeopleWorkspace
from .project_save_ui import ProjectSaving
from .project_workspace_ui import ModelWorkspace
from .mamma_ui import MammaWorkspace
from .prompthmr_ui import PromptHMRWorkspace
STAGES.update(mamma_prepare='Prepare Multi-view Videos',mamma_masks='MAMMA Person Masks',mamma_landmarks='MAMMA Surface Landmarks',mamma_camera='MAMMA Cameras',mamma_fit='MAMMA 3D Fit',mamma_publish='MAMMA Motion Correction and Preview')
STAGES.update(prompthmr_prepare='Prepare PromptHMR Video',prompthmr_track='PromptHMR People and Masks',prompthmr_pose='PromptHMR Camera, Keypoints and Motion',prompthmr_world='PromptHMR Shared World and Post-optimization',prompthmr_publish='PromptHMR Motion Correction and Preview')
from .gemx_ui import GemXWorkspace
STAGES['gemx_reconstruct']='Reconstruct Native SOMA / GEM-X'
STAGES['gemx_correct']='Motion Correction / GEM-X'
STAGES['save_project']='Save Project As'
STAGES.update(people='Find People',person='Prepare Person Motion')
STAGES.update(flow_camera='Estimate Scene Camera / VGGT-Omega', flow_features='Extract Body and Hand Features / SAM', flowhmr='Reconstruct Motion / FlowHMR', flow_export='Prepare FlowHMR Preview and FBX', import_model='Import Model')
STAGES.update(exmr_preprocess='ExMR Body, Hand and Foot Keypoints',exmr_features='ExMR Body and Hand Features',exmr='Reconstruct Motion / ExMR')

STAGES.update(flow_batch='Process FlowHMR Motion',exmr_batch='Process ExMR Motion',gvhmr_batch='GVHMR Multi-person',features='GVHMR Image Observations')

class ModelDownloadDialog(QDialog):
    """Readable, selectable provider instructions with bounded scrolling."""
    def __init__(self, parent, key, manager=None):
        super().__init__(parent)
        self.manager = manager
        self.model_key = key
        from .models import BY_KEY, manual_download_page, manual_instructions_html
        self.setWindowTitle(BY_KEY[key].name + " - Manual Download")
        self.setMinimumSize(360, 260)
        area = self.screen().availableGeometry()
        self.resize(min(680, max(360, area.width() - 80)),
                    min(560, max(260, area.height() - 100)))
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setStyleSheet(
            'QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:6px; }')
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.browser.setHtml(manual_instructions_html(key))
        self.browser.anchorClicked.connect(QDesktopServices.openUrl)
        layout.addWidget(self.browser)
        self.import_status = QLabel()
        self.import_status.setWordWrap(True)
        self.import_status.setMaximumHeight(64)
        self.import_status.hide()
        layout.addWidget(self.import_status)
        actions = QHBoxLayout(); actions.addStretch()
        self.website = QPushButton("Open Download Page && Auto-Import")
        self.website.setToolTip(tooltip_text('Open the provider website in your browser. Sign in and download there; the app notices the finished file in your Downloads folder, checks it, stores it in the models folder and removes the original. No need to move anything.'))
        self.website.clicked.connect(self.open_and_watch)
        actions.addWidget(self.website)
        self.import_file = QPushButton("Import...")
        self.import_file.setToolTip('Select the downloaded model file to check and import it. These instructions stay open.')
        self.import_file.clicked.connect(self.import_download)
        actions.addWidget(self.import_file)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        close.setToolTip('Close these download instructions.')
        actions.addWidget(close); layout.addLayout(actions)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.sync_import_status)
        self.sync_import_status()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_import_status()
        self.status_timer.start()

    def hideEvent(self, event):
        if getattr(self, 'watcher', None) is not None:
            self.watcher.stop()
        self.status_timer.stop()
        super().hideEvent(event)

    def open_and_watch(self):
        from .models import manual_download_page
        QDesktopServices.openUrl(QUrl(manual_download_page(self.model_key)))
        if self.manager is None:
            return
        if getattr(self, 'watcher', None) is None:
            from .model_signin import DownloadWatcher
            self.watcher = DownloadWatcher(self.manager, self, self.show_watch)
        self.watcher.start()

    def show_watch(self, text):
        self.import_status.setText(text)
        self.import_status.show()

    def import_download(self):
        if self.manager is not None:
            self.manager.import_local_model(self.model_key, picker_parent=self)
            self.sync_import_status()

    def sync_import_status(self):
        manager = self.manager
        self.import_file.setEnabled(manager is not None and not manager.window.busy)
        if manager is None:
            return
        manager.sync_task_status()
        message = manager.task_status.toPlainText() if manager._observing_setup or manager.window.busy else ''
        if self.import_status.toolTip() != message:
            self.import_status.setTextFormat(Qt.PlainText)
            self.import_status.setText(message[:500])
            self.import_status.setToolTip(message)
            self.import_status.setVisible(bool(message))


class ModelsDialog(QDialog):
    def __init__(self, window, tool_key=None):
        super().__init__(window)
        self.window = window
        self.tool = MOTION_TOOLS[tool_key or window.model_choice.currentData()]
        from .models import STAGE_REQUIREMENTS
        keys = ({key for key in STAGE_REQUIREMENTS[self.tool.key+'_assets']} if self.tool.key in ('mamma','prompthmr','pear','gemx') else
                {key for stage in (*self.tool.stages(refine=True, footmr=True, handflow=True),'people') for key in STAGE_REQUIREMENTS.get(stage, ())})
        if self.tool.key=='pear':keys.update(STAGE_REQUIREMENTS['pear_align'])
        if self.tool.key=='prompthmr':keys.update(STAGE_REQUIREMENTS['prompthmr_camera_assets'])
        if self.tool.key=='flowhmr':
            keys.add('flow_metadata')
            if window.flow_hamer.isChecked():
                from .flow_hamer import MODEL_KEYS
                keys.update(MODEL_KEYS)
        # Keep public automatic downloads together, followed by manual-only files.
        # Stable sorting preserves the existing component order within each group.
        self.models = sorted((model for model in MODELS_LIST if model.key in keys),
                             key=lambda model: not bool(model.drive_id or model.download_url))
        self.setWindowTitle(f"{self.tool.label} Model Manager")
        self.resize(850, min(540, max(360, self.screen().availableGeometry().height() - 80)))
        layout = QVBoxLayout(self)
        description = QLabel(f"Models for {self.tool.label} and compatible add-ons.\nUse Auto Download for public files, or Manual Download and Import for any model.\nRegistered body model files are not redistributed by this app.")
        description.setWordWrap(True)
        layout.addWidget(description)
        # Every model file lives under this one folder. Import writes each checked
        # file to the path shown in its row, so users can also place a downloaded
        # file there themselves before importing it.
        from .paths import MODELS
        folder = QHBoxLayout()
        self.models_path = QLabel('Model folder: ' + str(MODELS))
        self.models_path.setWordWrap(True)
        self.models_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.models_path.setToolTip(tooltip_text('The app reads models only from this folder. Each row shows the exact file path it uses; hover a row, or open Manual Download, to see it.'))
        folder.addWidget(self.models_path, 1)
        self.open_models_folder_button = QPushButton('Open Folder')
        self.open_models_folder_button.setToolTip(tooltip_text('Open this folder in Windows Explorer so you can copy a downloaded file into place.'))
        self.open_models_folder_button.clicked.connect(self.open_models_folder)
        folder.addWidget(self.open_models_folder_button)
        layout.addLayout(folder)
        if self.tool.key=='mamma':
            from .mamma_manager import runtime_panel
            layout.addWidget(runtime_panel(self))
        elif self.tool.key=='prompthmr':
            from .prompthmr_manager import runtime_panel
            layout.addWidget(runtime_panel(self))
        elif self.tool.key=='gemx':
            from .gemx_runtime import configuration
            try:
                configuration()
            except RuntimeError as error:
                message=str(error)
            else:
                message='GEM-X runtime is ready.'
            note=QLabel(message+' Download All prepares nine files. Review the upstream terms before downloading. '
                '<a style="color:#8ed2ff" href="https://huggingface.co/nvidia/GEM-X">GEM-X</a> · '
                '<a style="color:#8ed2ff" href="https://huggingface.co/nvidia/SOMA-X">SOMA-X</a> · '
                '<a style="color:#8ed2ff" href="https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose">Detector</a>')
            note.setWordWrap(True)
            note.setOpenExternalLinks(True)
            layout.addWidget(note)
        elif self.tool.key=='pear':
            from .pear_runtime import status
            note=QLabel(status()[1]+'\nPEAR support is included in install.cmd and update.cmd. Uses the existing shared runtime; Camera Align uses a small Flow-runtime support overlay. No CUDA compiler is needed. VGGT and GeoCalib are optional for multiple fixed views; model files are prepared separately below.')
            note.setWordWrap(True);layout.addWidget(note)
        self.table = QTableWidget(len(self.models), 3)
        self.table.setHorizontalHeaderLabels(["Model", "Status", "Action"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 300)
        self.table.setColumnWidth(1, 130)
        layout.addWidget(self.table)
        self.task_status = QTextBrowser()
        self.task_status.setFixedHeight(64)
        self.task_status.setOpenLinks(False)
        self.task_status.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.task_status.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.task_status.setStyleSheet(
            'QTextBrowser { font-family:"' + QApplication.font().family() + '"; font-size:12px; padding:4px; }')
        self.task_status.setPlainText("Ready to download or import models.")
        layout.addWidget(self.task_status)
        self.task_progress = QProgressBar()
        self.task_progress.setTextVisible(False)
        self.task_progress.setRange(0, 100); self.task_progress.setValue(0)
        layout.addWidget(self.task_progress)
        actions = QHBoxLayout()
        self.cancel_task = QPushButton("Cancel")
        self.cancel_task.setToolTip('Stop the current download or import queue. Completed files are kept.')
        self.cancel_task.setEnabled(False)
        self.cancel_task.clicked.connect(self.cancel_model_job)
        actions.addWidget(self.cancel_task)
        actions.addStretch()
        self.download_all = QPushButton("Download All")
        self.download_all.setToolTip(tooltip_text(
            "Download missing public model files. Files requiring registration must be downloaded manually and imported. Completed downloads are kept if you cancel or retry."))
        self.download_all.clicked.connect(self.start_download_all)
        actions.addWidget(self.download_all)
        close = QPushButton("Close")
        close.setToolTip(tooltip_text("Close Model Manager. Any running download or import continues in the main workspace."))
        close.clicked.connect(self.accept); actions.addWidget(close)
        layout.addLayout(actions)
        self._batch_total = 0
        self._observing_setup = False
        self._row_state = None
        self._status_state = None
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.sync_task_status)
        self.refresh()

    def open_models_folder(self):
        from .paths import MODELS
        MODELS.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS)))

    def refresh(self):
        from .models import STAGE_REQUIREMENTS
        for i, model in enumerate(self.models):
            name = model.name + (" (Optional: HTD / 2D)" if self.tool.key == "gvhmr" and model.key == "pva" else "")
            if self.tool.key=='pear' and model.key in STAGE_REQUIREMENTS['pear_align']:name+=' (Optional: Camera Align)'
            available = ready(model)
            for column, text in ((0, name), (1, "Ready" if available else "Missing")):
                item = self.table.item(i, column)
                if item is None:
                    item = QTableWidgetItem(); self.table.setItem(i, column, item)
                item.setText(text)
                if column==0:item.setToolTip(str(model.path))
                item.setForeground(QColor("#d5d5d5" if available else "#ffb45c"))
                font = item.font(); font.setBold(not available); item.setFont(font)
                item.setToolTip(tooltip_text(("Registered. " if available else
                    "Missing. Download it from the provider and use Import in this row; the app checks it, then registers it at:\n") +
                    str(model.path)))
            cell = self.table.cellWidget(i, 2)
            if cell is None or cell.property("model_key") != model.key:
                cell = QWidget(); cell.setProperty("model_key", model.key)
                row = QHBoxLayout(cell); row.setContentsMargins(0, 0, 0, 0)
                cell.automatic = None
                if model.drive_id or model.download_url:
                    cell.automatic = QPushButton("Auto Download")
                    cell.automatic.setToolTip(tooltip_text("Download this file automatically. If it fails, use Manual Download or Import an existing copy."))
                    cell.automatic.clicked.connect(lambda checked=False, key=model.key: self.start_download(key))
                    row.addWidget(cell.automatic)
                from .models import PROVIDER_FILES
                if cell.automatic is None and model.key in PROVIDER_FILES:
                    cell.automatic = QPushButton("Auto Download")
                    cell.automatic.setToolTip(tooltip_text("Sign in with your own free account and the file is downloaded and installed here automatically. Your password is used only for this download and is not stored."))
                    cell.automatic.clicked.connect(lambda checked=False, key=model.key: self.start_provider_download(key))
                    row.addWidget(cell.automatic)
                cell.manual = QPushButton("Manual Download...")
                cell.manual.setToolTip(tooltip_text("Show download links and instructions. Download in your browser, then use Import in the instructions or this row."))
                cell.manual.clicked.connect(lambda checked=False, key=model.key: self.manual_setup(key))
                row.addWidget(cell.manual)
                cell.local = QPushButton("Import...")
                cell.local.setToolTip(tooltip_text("Choose a model file you already downloaded. The app checks it before replacing any existing file."))
                cell.local.clicked.connect(lambda checked=False, key=model.key: self.import_local_model(key))
                row.addWidget(cell.local)
                self.table.setCellWidget(i, 2, cell)
            if cell.automatic is not None:
                cell.automatic.setEnabled(not self.window.busy and not available)
            cell.manual.setEnabled(True)
            cell.local.setEnabled(not self.window.busy)
        self.download_all.setEnabled(not self.window.busy and bool(self.pending_downloads()))
        self._row_state = (self.window.busy, id(self.window.process))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh(); self.sync_task_status()
        self.status_timer.start()

    def hideEvent(self, event):
        self.status_timer.stop()
        super().hideEvent(event)

    def sync_task_status(self):
        # Poll only cheap GUI state; filesystem readiness is refreshed once per
        # worker transition, never on each progress update or every timer tick.
        w = self.window
        row_state = (w.busy, id(w.process))
        if row_state != self._row_state:
            self.refresh()
        setup = w.current_stage in {"download", "import_model"}
        if w.busy and setup:
            self._observing_setup = True
        state = (w.busy, setup, w.cancelled, w.failed, w.status.text(),w.task_info.text(),
                 w.progress.minimum(), w.progress.maximum(), w.progress.value(), len(w.jobs))
        if state == self._status_state:
            return
        self._status_state = state
        active = w.busy and setup
        self.cancel_task.setEnabled(active and not w.cancelled)
        if setup and self._observing_setup:
            message = w.status.text()
            if active:message+='\n'+w.task_info.text()
            if active and self._batch_total > 1 and not w.cancelled and not w.failed:
                index = min(self._batch_total, max(1, self._batch_total - len(w.jobs)))
                message = f"Model {index}/{self._batch_total}: " + message
            self.task_status.setPlainText(message if len(message) <= 4096 else message[:4096] + "\nSee Console for the complete message.")
            self.task_status.setStyleSheet(
                'QTextBrowser { font-family:"' + QApplication.font().family() +
                '"; font-size:12px; padding:4px;' +
                ('color:#ffb45c;' if w.failed else '') + ' }')
            if active:
                self.task_progress.setRange(w.progress.minimum(), w.progress.maximum())
                self.task_progress.setValue(w.progress.value())
            else:
                self.task_progress.setRange(0, 100)
                self.task_progress.setValue(0 if w.failed or w.cancelled else 100)
        elif w.busy:
            self.task_status.setPlainText("Another task is running. You can still open manual download instructions.")
            self.task_progress.setRange(0, 0)
        else:
            self.task_status.setPlainText("Ready to download or import models.")
            self.task_progress.setRange(0, 100); self.task_progress.setValue(0)

    def start_model_jobs(self, jobs):
        if self.window.busy:
            return
        self._batch_total = len(jobs)
        self._observing_setup = True
        self._status_state = None
        self.window.start_jobs(jobs)
        self.sync_task_status()

    def cancel_model_job(self):
        if self.window.busy and self.window.current_stage in {"download", "import_model"}:
            self.window.cancel_job()
            self.sync_task_status()

    def pending_downloads(self):
        return [model for model in self.models
                if (model.drive_id or model.download_url) and not ready(model)]

    def start_download_all(self):
        if self.window.busy:
            return
        jobs = [("download", ["--model", model.key]) for model in self.pending_downloads()]
        if not jobs:
            self.refresh()
            return
        self.start_model_jobs(jobs)

    def start_download(self, key):
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        self.start_model_jobs([("download", ["--model", key])])

    def start_provider_download(self, key):
        """Sign in with the person's own account and install a registration-gated file."""
        if self.window.busy or key not in {model.key for model in self.models}:
            return
        from .model_signin import download_with_sign_in
        self._provider_download = download_with_sign_in(self, key, self, self.task_status.setPlainText)

    def manual_setup(self, key):
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        dialog = ModelDownloadDialog(self, key, manager=self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def import_local_model(self, key, picker_parent=None):
        from .models import BY_KEY
        if self.window.busy:
            return
        if key not in {model.key for model in self.models}:
            raise ValueError("This model is not part of the selected motion tool.")
        model = BY_KEY[key]
        name, _ = QFileDialog.getOpenFileName(picker_parent or self, 'Import ' + model.name, '',
            ('PEAR assets (assets.zip);;All files (*)' if key=='pear_geometry' else f'Model file ({model.path.name});;All files (*)'))
        if name:
            self.start_model_jobs([('import_model', ['--model', key, '--video', name])])



from .character_ui import CharacterWorkspace
from .mamma_ui import MammaWorkspace
from .prompthmr_ui import PromptHMRWorkspace
from .pear_ui import PearWorkspace
from .preview_export_ui import PreviewExportWorkspace

class MainWindow(PreviewExportWorkspace,PearWorkspace,GemXWorkspace,PromptHMRWorkspace,MammaWorkspace,CharacterWorkspace,ModelWorkspace,ProjectSaving,PeopleWorkspace,QMainWindow):
    def __init__(self, video: Path | None = None):
        super().__init__()
        self.init_mamma()
        self.setWindowTitle("RGB MoCap | Motion Workspace")
        self.resize(1440, 880)
        self.setMinimumSize(1100, 700)
        self.video_path = None
        self.init_mamma()
        self.init_pear()
        self.project = None
        self.workspace_root=None
        self._workspace_loading=False
        self._workspace_video_reference=False
        self._workspace_camera=None
        self.init_people()
        self.active_result = "corrected_footmr_refined"
        self.info = None
        self.source_offset = 0.0
        self.source_end = 0.0
        self.video_readable = False
        self.busy = False
        self.jobs = []
        self.process = None
        self.cancel_process = None
        self.buffer = ""
        self.failed = False
        self.camera_session = None
        self.live_warmup = False
        self._live_person_row = False
        self.live_window_seconds = 10.0
        self.preview_volume = 1.0
        self._volume_before_mute = 100
        self.preload_worker = None
        self.preload_log = None
        self.cancelled = False
        self.hardware_summary = None
        self.current_stage = None
        self._batch_started_at = None
        self._batch_has_motion = False
        from .task_progress import TaskProgress
        self.task_state=TaskProgress()
        self._task_manifests={};self._queued_task_estimates=[]
        self.play_clock = QElapsedTimer()
        self.timer_resolution = PlaybackTimerResolution()
        self.play_anchor = 0
        self.playback_tick = False
        self.presented_times = deque()
        self.source_audio = SourceAudio(self)
        self.preview_data = None
        self.preview_signature = None
        from .contact_writer import ContactWriter
        self.contact_tracks={};self.contact_saving=False;self.contact_save_token=0
        self.contact_writer=ContactWriter(self)
        self.contact_writer.saved.connect(self.foot_contacts_saved)
        self.contact_writer.failed.connect(self.foot_contacts_failed)
        self.contact_writer.settled.connect(self.foot_contacts_settled)
        self.data_loader = PreviewDataLoader(self)
        self.data_loader.ready.connect(self.preview_ready)
        self.data_loader.failed.connect(self.preview_failed)
        self.data_loader.tracked_actor_ready.connect(self.tracked_actor_ready)
        build_workspace(self)
        self.init_characters()
        self.video_view.frame_presented.connect(self.refresh_preview_export_action)
        self.scene_loader.ready.connect(self.refresh_preview_export_action)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.next_frame)
        self.shortcuts = []
        for key, action in [("Space", self.toggle_play), ("Left", lambda: self.step_frames(-1)),
                            ("Right", lambda: self.step_frames(1)),
                            ("Shift+Left", lambda: self.step_frames(-round(self.info['fps']) if self.info else -30)),
                            ("Shift+Right", lambda: self.step_frames(round(self.info['fps']) if self.info else 30)),
                            ("Home", lambda: self.step_frames(-self.slider.maximum())),
                            ("End", lambda: self.step_frames(self.slider.maximum()))]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        self.setAcceptDrops(True)
        self.data_loader.start()
        self.contact_writer.start()
        # Load the live models in the background from the start: the first live
        # session then moves the character instead of waiting for them.
        QTimer.singleShot(2500, self.preload_live_solver)
        self.processing_guard = None
        self.guard_timer = QTimer(self)
        self.guard_timer.setInterval(500)
        self.guard_timer.timeout.connect(self.check_processing_resources)
        self.task_timer=QTimer(self)
        self.task_timer.setInterval(500)
        self.task_timer.timeout.connect(self.update_task_progress)
        # Initialize the same model-specific visibility used on selector changes.
        # Otherwise the empty GVHMR stack reserves the FlowHMR page height.
        self.select_model()
        if video:
            self.load_video(video)

    @read_only_checks
    def update_readiness(self):
        self.export_preview_action.setEnabled(self.preview_export_available())
        if self._pear_active:
            if hasattr(self,'camera_tools'):self.camera_tools.sync()
            self.pear_sync_controls();return
        if getattr(self,'_readiness_depth',0):
            self._readiness_pending=True
            return
        if hasattr(self,'character_loader'):self.sync_characters()
        self.update_people_readiness()
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        self.compute_mode.setEnabled(not self.busy)
        self.resource_safety.setEnabled(not self.busy)
        stages = tool.stages(self.htd_refine.isChecked(), self.footmr_refine.isChecked(), handflow=self.handflow_refine.isChecked())
        from .models import STAGE_REQUIREMENTS, BY_KEY
        required = {key for stage in stages for key in STAGE_REQUIREMENTS.get(stage, ())}
        if self.flow_hamer.isChecked():
            for checkbox in (self.fast_sam,self.flow_hand_stride4):
                blocked=checkbox.blockSignals(True);checkbox.setChecked(True);checkbox.blockSignals(blocked)
            if tool.key=='flowhmr':
                from .flow_hamer import MODEL_KEYS
                required.update(MODEL_KEYS)
        if tool.key=='flowhmr' and self.flow_sampling.currentData()==-1:required.discard('flow_vggt')
        if tool.key=='mamma' and self.mamma_camera_mode.currentData()=='manual':required.difference_update(('flow_vggt','pear_geocalib'))
        if tool.key=='prompthmr' and self.prompthmr_camera.currentData()=='moving':required.update(STAGE_REQUIREMENTS['prompthmr_camera_assets'])
        unavailable = [BY_KEY[key] for key in sorted(required) if not ready(BY_KEY[key])]
        count = len(required) - len(unavailable)
        text = f"{tool.label}: {count}/{len(required)} models ready"
        if unavailable:
            noun = "model" if len(unavailable) == 1 else "models"
            text += f"\nWarning: {len(unavailable)} required {noun} missing.\nOpen Model Manager to download or import."
        self.readiness.setStyleSheet("color:#ffb45c; font-weight:600;" if unavailable else "")
        self.readiness.setToolTip(tooltip_text("Missing required models:\n" + "\n".join(model.name for model in unavailable) if unavailable else ""))
        self.readiness.setText(text)
        for button in (self.models_button, self.analyze_button, self.run_button):
            button.setEnabled(not self.busy)
        self.cancel_button.setEnabled(self.busy)
        self.check_system_action.setEnabled(not self.busy)
        self.model_choice.setEnabled(not self.busy and not self.contact_saving)
        self.htd_refine.setEnabled(not self.busy)
        self.footmr_refine.setEnabled(not self.busy)
        self.handflow_refine.setEnabled(not self.busy)
        self.motion_correction.setEnabled(not self.busy)
        self.correction_controls.setEnabled(not self.busy and self.motion_correction.isChecked())
        flat=self.correction_mode.currentData()=='flat'
        self.floor_board.setEnabled(not self.busy and flat)
        for widget in (self.floor_board_paper, self.floor_board_page):
            widget.setEnabled(not self.busy and flat)
        self.auto_floor.setEnabled(not self.busy and flat)
        # Typing a height switches the automatic estimate off, so the field never needs unlocking first.
        self.floor_height.setEnabled(not self.busy and flat)
        self.correction_button.setEnabled(not self.busy and not self.contact_saving and self.project is not None and
            (result_is_current(self.project, 'initial') if tool.key=='gemx' else
             (self.project/RESULTS[base_result(self.active_result)][0]).exists()))
        self.slider.trim_enabled = not self.busy and self.info is not None
        self.slider.contact_edit_enabled=not self.busy and not self.contact_saving and self.motion_correction.isChecked() and self.slider.contacts_available
        self.slider.contact_track_enabled=(self.foot_lock.isChecked(),)*2+(self.hand_lock.isChecked(),)*2
        self.slider.update()
        self.reset_footlocks_button.setEnabled(self.slider.contact_edit_enabled)
        self.reset_trim_button.setEnabled(self.slider.trim_enabled)
        character=self.selected_character()
        motion_ready, motion_reason = self.motion_export_state()
        if character:
            self.export_action.setText('Export Character...')
        elif motion_ready:
            self.export_action.setText('Export Motion (FBX, BVH, Blender)...')
        else:
            self.export_action.setText('Export Motion (FBX, BVH, Blender)...')
        self.export_action.setEnabled(not self.busy and (
            bool(character.get('takes')) and not character.get('needs_retarget')
            if character else motion_ready))
        self.export_action.setToolTip(tooltip_text(
            "Choose a character export format and whether to include its mesh. PMX characters also support VMD motion."
            if character else
            "Export the displayed motion as FBX, BVH or a portable Blender project. "
            "Multiple people export separately; MAMMA also offers a whole-scene Blender bundle."
            + ('' if motion_ready else '\n\nUnavailable: ' + motion_reason)))
        # FBX and GLB have their own File menu entries and stay visible whether
        # or not a character row is selected.
        exportable_character = not self.busy and not self.contact_saving and (
            bool(character.get('takes')) if character else any(c.get('takes') for c in getattr(self,'characters',[])))
        self.export_character_action.setEnabled(exportable_character)
        self.export_character_glb_action.setEnabled(exportable_character)
        if exportable_character:
            character_reason = 'Choose the format and whether to include the mesh.'
        elif not getattr(self, 'characters', []):
            character_reason = ('No character is imported. These entries export an imported character; '
                                'the reconstructed person is exported with Export Motion above. '
                                'Use File > Import Character to add one.')
        elif not any(c.get('takes') for c in getattr(self, 'characters', [])):
            character_reason = 'The character has no motion yet. Retarget a reconstructed person onto it first.'
        else:
            character_reason = 'Finish the current job or contact edit, then export again.'
        for action in (self.export_character_action, self.export_character_glb_action):
            base = action.toolTip().split('\n\nUnavailable:')[0]
            action.setToolTip(tooltip_text(base + ('' if exportable_character
                                                  else '\n\nUnavailable: ' + character_reason)))
        self.play_button.setEnabled(self.info is not None and self.video_readable)
        for action in self.open_actions:
            action.setEnabled(not self.busy)
        self.save_project_action.setEnabled(not self.busy and not self.contact_saving and self.video_path is not None and self.info is not None)
        self.mesh_button.setEnabled(self.preview_preparation_available())
        self.import_character_action.setEnabled(not self.busy and not self.contact_saving)
        self.flow_sampling.setEnabled(not self.busy)
        self.fast_sam.setEnabled(not self.busy and not self.flow_hamer.isChecked())
        self.flow_hand_stride4.setEnabled(not self.busy and not self.flow_hamer.isChecked())
        self.flow_hamer.setEnabled(not self.busy)
        self.flow_acceleration.setVisible(self.fast_sam.isChecked())
        self.flow_acceleration.setEnabled(not self.busy)
        self.exmr_camera.setEnabled(not self.busy)
        self.exmr_smoothing.setEnabled(not self.busy)
        self.flowhmr_background.setEnabled(not self.busy)
        self.exmr_background.setEnabled(not self.busy)
        self.analyze_button.setVisible(tool.key=='gvhmr')
        self.sync_correction_floor_preview()
        self.sync_mamma_controls()
        self.sync_prompthmr_controls()
        self.sync_gemx_controls()
        if hasattr(self,'camera_tools'):self.camera_tools.sync()

    def sync_correction_floor_preview(self):
        self.motion_view.correction_floor_applicable=self.correction_mode.currentData()=='flat'
        enabled=self.motion_correction.isChecked() and self.motion_view.correction_floor_applicable
        pending=bool(self.project is not None and enabled and
                     not result_is_current(self.project,'corrected_'+base_result(self.active_result)))
        self.motion_view.set_correction_floor(self.floor_height.value() if self.project is not None else None,
            automatic=self.auto_floor.isChecked(),pending=pending,enabled=enabled)
        self.refresh_camera_hint()

    def refresh_camera_hint(self):
        """Recommend a camera mode when the clip itself moves."""
        from .camera_motion import cached
        if self.project is None:
            self.camera_hint.hide()
            return
        value = cached(self.project)
        if value is None:
            attempted = getattr(self, 'camera_hint_attempted', None)
            if attempted is None:
                attempted = self.camera_hint_attempted = set()
            key = str(Path(self.project).resolve())
            if key not in attempted:
                from .camera_check_ui import CameraCheck
                if getattr(self, 'camera_check', None) is None:
                    self.camera_check = CameraCheck(self)
                    self.camera_check.ready.connect(self.camera_hint_ready)
                attempted.add(key)
                self.camera_check.check(self.project)
        self.apply_camera_hint(value)

    def camera_hint_ready(self, path, value):
        if self.project is None or Path(path).resolve() != Path(self.project).resolve():
            return
        self.apply_camera_hint(value)

    def apply_camera_hint(self, value):
        from .camera_motion import advice
        tool = self.model_choice.currentData()
        if tool == 'exmr':
            static = self.exmr_camera.currentData() is True
        elif tool == 'flowhmr':
            static = self.flow_sampling.currentData() == -1
        else:
            static = True
        text = advice(value, tool, static) or ''
        if text != self.camera_hint.text():self.camera_hint.setText(text)
        self.camera_hint.setVisible(bool(text))

    def save_floor_board_page(self):
        from .floor_board import save_page
        paper=self.floor_board_paper.currentData()
        name, _ = QFileDialog.getSaveFileName(self, "Save Floor Reference Board",
                                              str(Path.home() / f'rgbmocap-floor-board-{paper}.png'), "PNG (*.png)")
        if not name:
            return
        try:
            path = save_page(name, paper=paper)
        except OSError as error:
            QMessageBox.warning(self, "Floor Reference Board", str(error))
            return
        self.log.appendPlainText(f"Saved the {paper.upper()} floor reference board: " + str(path))
        self.log.appendPlainText("Print it at 100% (actual size), lay it flat on the floor and show it once in the video.")
        self.log.appendPlainText("The app reads the sheet size from the printed sheet, so any supported size can be used later.")

    def show_model_guide(self):
        from .model_guide import ModelGuideDialog
        dialog = ModelGuideDialog(self, self.model_choice.currentData())
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def select_model(self, *_):
        if self.pear_select_model():return
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        if self.workspace_root is not None and not self._workspace_loading and tool.key!=getattr(self,'_selected_tool',None):
            if self.busy or self.contact_saving:
                self.model_choice.blockSignals(True)
                self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool))
                self.model_choice.blockSignals(False)
                return
            self.switch_workspace_model(tool.key)
            return
        previous=getattr(self,'_selected_tool','gvhmr')
        choices=getattr(self,'_tool_refinement',{'gvhmr':(True,True,True,True)})
        if previous!=tool.key:
            if previous=='gvhmr':choices['gvhmr']=(self.htd_refine.isChecked(),self.footmr_refine.isChecked(),self.motion_correction.isChecked(),self.handflow_refine.isChecked())
            else:choices[previous]=(False,False,self.motion_correction.isChecked(),False)
            values=choices.get(tool.key,(tool.key=='gvhmr',tool.key=='gvhmr',True,tool.key=='gvhmr'))
            for control,value in zip((self.htd_refine,self.footmr_refine,self.motion_correction,self.handflow_refine),values):
                control.blockSignals(True);control.setChecked(value);control.blockSignals(False)
        self._selected_tool=tool.key;self._tool_refinement=choices
        self.active_result=self.selected_result()
        if self.project is not None:
            state=json.loads((self.project/'project.json').read_text(encoding='utf-8'))
            if state.get('pipeline',{}).get('model','gvhmr')!=tool.key:
                # Keep source timing/camera but never show another engine's result.
                self.stop_playback();self.clear_preview();self.reset_people()
                self.project=None
                self.active_result=self.selected_result()
                self.status.setText(f'{tool.label} selected. Reconstruct this clip or open a {tool.label} project.')
        for index in range(self.model_pages.count()):
            if self.model_pages.widget(index).property("motion_tool") == tool.key:
                self.model_pages.setCurrentIndex(index)
                break
        self.model_pages.setVisible(tool.key not in ('gvhmr', 'gemx'))
        self.htd_refine.setVisible(bool(tool.refinement))
        self.refine_description.setVisible(bool(tool.refinement))
        self.footmr_refine.setVisible(tool.key == "gvhmr")
        self.handflow_refine.setVisible(tool.key == "gvhmr")
        self.footmr_description.setVisible(tool.key == "gvhmr")
        if not tool.refinement:
            self.refine_preview_note.hide()
        self.models_group.setTitle(tool.label+' Models')
        self.correction_group.show()
        self.slider.contacts_supported=True
        self.slider._cache_key=None
        self.slider.update()
        self.correction_description.setText('Only checked contact types are corrected. Uses the selected '+tool.label+' result.')
        self.analyze_button.setToolTip(tooltip_text('Find body points in the video for GVHMR refinement. Reconstruct 3D normally includes this step when needed.'))
        self.mamma_mode_changed(previous,tool.key)
        self.update_readiness()

    def selected_result(self, correction=None):
        gvhmr=self.model_choice.currentData()=='gvhmr'
        return result_kind(gvhmr and self.htd_refine.isChecked(),gvhmr and self.footmr_refine.isChecked(),
            self.motion_correction.isChecked() if correction is None else correction,
            handflow=gvhmr and self.handflow_refine.isChecked())

    def refinement_changed(self, enabled):
        self.set_active_result(self.selected_result())

    def unavailable_result_note(self, kind):
        """Explain which selected variant has no committed preview yet."""
        result = {"initial":MOTION_TOOLS[self.model_choice.currentData()].label, "refined":"HTD",
                  "footmr_initial":"FootMR", "footmr_refined":"HTD + FootMR"}[body_result(kind)]
        if has_handflow(kind):result += " + HandFlow"
        if has_handflow(kind) and not result_is_current(self.project,base_result(kind)):
            return 'No current HandFlow preview. Run Reconstruct 3D to infer finger motion.'
        if kind.startswith('corrected_'):
            return 'Correction settings changed or result unavailable.\nUse Apply Correction to update the preview.'
        return f"No {result} preview yet. Run Reconstruct 3D to create it."

    def preview_display_fallbacks(self, kind):
        """Closest displayable variants of a result that was never computed."""
        from .motion_tools import RESULTS, base_result, body_result
        candidates=[]
        for candidate in (kind.replace('handflow_','',1), base_result(kind),
                          body_result(kind), body_result(base_result(kind)), 'initial'):
            if candidate in RESULTS and candidate!=kind and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def displayable_result(self, project=None, kind=None, ignore_contacts=False):
        """Closest committed variant of the selected result, for display only.

        The selection itself never changes: a variant such as HandFlow that has
        no committed preview yet resolves to the same stack without it, so one
        unchecked option cannot blank the viewport or drop actors from a scene.
        """
        from .result_artifacts import artifacts_current
        project = self.project if project is None else project
        kind = self.active_result if kind is None else kind
        if project is None:
            return kind
        for candidate in (kind, *self.preview_display_fallbacks(kind)):
            if artifacts_current(project, candidate) and result_is_current(project, candidate, ignore_contacts):
                return candidate
        return kind

    def current_export(self, kind=None, project=None):
        if self._pear_active:return None
        kind=self.active_result if kind is None else kind
        project=self.project if project is None else project
        info=self.contact_tracks.get(base_result(kind))
        if kind.startswith('corrected_') and (self.contact_saving or (info and info.get('unsaved'))):return None
        if project is not None and result_is_current(project, kind):
            return selected_export(project, kind)
        return None

    def render_changed(self):
        self.motion_view.show_bones=self.bone_overlay.isChecked()
        self.motion_view.show_character_bones=self.character_bones.isChecked()
        self.follow_source=self.follow_source_box.isChecked()
        self.motion_view.update()
        if hasattr(self,'camera_tools'):self.camera_tools.canvas.update()

    def motion_export_state(self):
        """Whether the displayed motion can be exported, and why not when it cannot."""
        if self.project is None:
            return False, 'Open a project or a video and reconstruct it first.'
        try:
            from .motion_tools import RESULTS, base_result, result_is_current
            from .mixamo_data import current as preview_current
        except Exception:                   # noqa: BLE001 - reported as unavailable
            return False, 'Export support is unavailable in this installation.'
        kind = self.active_result
        for pending in ('exmr_pending.json', 'flowhmr_pending.json', 'flow_world_pending.json'):
            if (self.project / pending).exists():
                return False, ('The last reconstruction did not finish (it left '
                               + pending + '). Run Reconstruct 3D again.')
        if not (self.project / RESULTS[base_result(kind)][0]).is_file():
            return False, 'There is no reconstruction in this project yet. Run Reconstruct 3D.'
        if not result_is_current(self.project, kind):
            return False, ('The stored motion no longer matches this project. '
                           'Apply Correction, or run Reconstruct 3D again.')
        if self.current_export() is None:
            if preview_current(self.project, kind):
                return False, 'The export file is missing. Prepare Preview again.'
            # This is what an update leaves behind: the preview was built by
            # earlier code, so it is refused rather than exported silently.
            return False, ('This preview was made by an earlier version of the app. '
                           'Press Prepare Preview to rebuild it (no inference is rerun), '
                           'then export.')
        return True, ''

    def preview_preparation_available(self):
        if self.model_choice.currentData()=='gemx':return False
        if self.project is None or self.busy or self.contact_saving:return False
        if not (self.project/RESULTS[base_result(self.active_result)][0]).is_file():return False
        info=self.contact_tracks.get(base_result(self.active_result),{})
        if self.active_result.startswith('corrected_') and info.get('unsaved'):return False
        if self.model_choice.currentData()=='flowhmr':
            from .flow_placement import needs_update
            if needs_update(self.project):return True
        return ((self.project/RESULTS[self.active_result][0]).is_file()
                and result_is_current(self.project,self.active_result))

    def prepare_target_preview(self):
        if self.model_choice.currentData()=='gemx':return
        if self.project is None or self.busy or self.contact_saving:return
        from .mixamo_data import current
        projects=[self.project]
        if self.people_state:
            from .people_store import safe_child
            projects += [safe_child(self.people_root,take['project']) for person in self.people_state['people']
                if person.get('checked') for take in person.get('takes',[])]
        jobs=[]
        for project in dict.fromkeys(projects):
            from .flow_placement import needs_update
            result=self.displayable_result(project)
            placement_old=self.model_choice.currentData()=='flowhmr' and needs_update(project)
            if (project/RESULTS[base_result(result)][0]).is_file() and (placement_old or result_is_current(project,result) and not current(project,result)):
                jobs.append(('rig',['--project',str(project),'--result',result]))
        if jobs:self.start_jobs(jobs)

    def open_model_manager(self):
        dialog = ModelsDialog(self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def set_active_result(self, kind):
        """Select the cached preview and sync the HTD option without inference."""
        htd, foot = result_options(kind)
        self.active_result = kind
        self.handflow_refine.blockSignals(True)
        self.handflow_refine.setChecked(has_handflow(kind))
        self.handflow_refine.blockSignals(False)
        self.htd_refine.blockSignals(True)
        self.htd_refine.setChecked(htd)
        self.htd_refine.blockSignals(False)
        self.footmr_refine.blockSignals(True)
        self.footmr_refine.setChecked(foot)
        self.footmr_refine.blockSignals(False)
        self.motion_correction.blockSignals(True)
        self.motion_correction.setChecked(kind.startswith('corrected_'))
        self.motion_correction.blockSignals(False)
        self.refresh_results()
        self.request_people_scene()
        self.update_readiness()

    def choose_video(self):
        if self._pear_active:
            names,_=QFileDialog.getOpenFileNames(self,'Open PEAR Camera Videos','', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
            if names:self.load_pear_videos(names)
            return
        if self.model_choice.currentData()=='mamma':
            names,_=QFileDialog.getOpenFileNames(self,'Open MAMMA Camera Videos','', 'Videos (*.mp4 *.mov *.avi *.mkv *.webm)')
            if names:self.load_mamma_videos(names)
            return
        name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)")
        if name:
            self.load_video(Path(name))

    def new_session(self):
        """Start an empty workspace: a new project without restarting the app.

        The open video or link, its decoder, the live solver and every
        per-project pane are released, so the next Open starts from the same
        state as a freshly launched window. Saved project folders are not
        touched: nothing on disk is deleted.
        """
        if self.busy:
            self.status.setText('Cancel the running job before starting a new project.')
            return
        if getattr(self, 'camera_session', None) is not None:
            self.stop_live_camera()
        if getattr(self, 'workspace_root', None) is not None:
            self.remember_workspace_model()
        self.workspace_root = None
        self._workspace_video_reference = False
        self._workspace_camera = None
        self.stop_playback()
        self.clear_preview()
        self.reset_people()
        self.motion_view.set_motion(None, None, None)
        self.motion_view.auto_fit = True
        # The decoder, its cache and its frames belong to the project left behind.
        self.video_view.shutdown()
        self.video_path = self.project = self.info = None
        self.source_url = ''
        self.source_download.setEnabled(False)
        self.source_offset = self.source_end = 0.0
        self.video_readable = False
        self.slider.setRange(0, 0)
        self.slider.reset_trim()
        self.clip_changed(0, 0)
        self.active_result = self.selected_result()
        self.restore_correction_options({})
        self.filename.setText('No clip loaded')
        self.fps_label.setText('No clip loaded')
        self.time_label.setText('0.00 / 0.00 s')
        self.setWindowTitle('RGB MoCap | Motion Workspace')
        self.update_readiness()
        self.status.setText('New project. Open a video or a video link to begin.')

    def record_webcam(self):
        """Record a short clip from a webcam and open it like any video."""
        if self.busy:
            return
        from .webcam_dialog import WebcamDialog
        dialog = WebcamDialog(self, on_window=self.load_video)
        try:
            accepted = dialog.exec() == QDialog.Accepted
            if dialog.live_session is not None:
                # The window length is chosen in the dialog; Record window
                # keeps using it once the workspace owns the live camera.
                self.live_window_seconds = float(dialog.seconds.value())
                self.start_live_camera(dialog.live_session)
                return
            if accepted and dialog.recorded:
                self.load_video(Path(dialog.recorded))
        finally:
            dialog.deleteLater()

    def start_live_camera(self, session, record=True):
        """Show the handed-over live source: preview only.

        Nothing is written to disk here. This is the rehearsal view - Find
        person starts the character so its motion can be checked against the
        performer, and only Record window captures a take. A web link
        (``record=False``) is a stream with no local capture to cut a window
        from, so its Record window stays hidden.
        """
        self.camera_session = session
        self.video_view.open_camera(session)
        label = getattr(session, 'label', 'Live camera')
        self._live_source_ended = False
        self._live_person_row = False
        self.live_record_button.setVisible(bool(record))
        self.live_stop_button.setVisible(True)
        self.start_live_box()
        # Load the solver while the source is already playing: Find person then
        # moves the character at once instead of waiting for a model load.
        self.warm_live_solver()
        self.fps_label.setText(f"{label} · {session.width}×{session.height}")
        self.filename.setText(
            f"{label} · {session.width}×{session.height} "
            + ("· Find person checks the character, ● Record window keeps the take" if record
               else "· Find person moves the character with the video · nothing is saved"))
        # Find person works on the live feed. The character is only called
        # in BY Find person - the order is: live, find, then the character.
        self.update_people_readiness()
        self.show_link_playback(session)

    def show_link_playback(self, session):
        """Show the link's playhead and Loop, or hide them for a camera.

        Only a web link carries these controls: it is the one live source that
        can be restarted at a chosen second, and the one that can run out.
        """
        controls = getattr(self, 'link_playback', None)
        timer, self.link_playback_timer = getattr(self, 'link_playback_timer', None), None
        if timer is not None:
            timer.stop()
        if controls is None:
            return
        seekable = session is not None and hasattr(session, 'seek') and hasattr(session, 'report')
        controls.setVisible(bool(seekable))
        if not seekable:
            return
        self._link_dragging = False
        self.link_loop.blockSignals(True)
        self.link_loop.setChecked(bool(getattr(session, 'loop', True)))
        self.link_loop.blockSignals(False)
        length = float(getattr(session, 'duration', 0.0) or 0.0)
        self.link_position.setEnabled(bool(length))
        self.link_position.blockSignals(True)
        self.link_position.setValue(0)
        self.link_position.blockSignals(False)
        timer = QTimer(self)
        timer.setInterval(200)
        timer.timeout.connect(self.update_link_playback)
        timer.start()
        self.link_playback_timer = timer
        self.update_link_playback()

    def update_link_playback(self):
        """Follow the link's playhead; the character's own state is untouched."""
        session = getattr(self, 'camera_session', None)
        if session is None or not hasattr(session, 'report'):
            timer, self.link_playback_timer = getattr(self, 'link_playback_timer', None), None
            if timer is not None:
                timer.stop()
            return
        report = session.report()
        length = float(report.get('duration') or 0.0)
        position = float(report.get('position') or 0.0)
        if not getattr(self, '_link_dragging', False):
            self.link_position.blockSignals(True)
            self.link_position.setValue(int(round(1000 * position / length)) if length else 0)
            self.link_position.blockSignals(False)
            self.link_time.setText(link_clock(position, length))

    def link_position_pressed(self):
        # Hold the playhead still while the handle is being dragged, so the
        # stream's own progress does not fight the thumb.
        self._link_dragging = True

    def link_position_moved(self, value):
        session = getattr(self, 'camera_session', None)
        length = float(getattr(session, 'duration', 0.0) or 0.0)
        if length:
            self.link_time.setText(link_clock(length * value / 1000.0, length))

    def link_position_released(self):
        self._link_dragging = False
        session = getattr(self, 'camera_session', None)
        length = float(getattr(session, 'duration', 0.0) or 0.0)
        value = self.link_position.value()
        if session is None or not length or not hasattr(session, 'seek'):
            return
        seconds = length * value / 1000.0
        if session.seek(seconds):
            # Say where the playhead is going now; the reopened stream confirms
            # it within a frame or two.
            self.link_time.setText(link_clock(seconds, length))
            self.status.setText('Link preview moving to ' + link_clock(seconds, length)
                                + ' (the stream reopens there).')

    def set_link_loop(self, enabled):
        session = getattr(self, 'camera_session', None)
        if session is None or not hasattr(session, 'set_loop'):
            return
        session.set_loop(enabled)
        self.status.setText('Link preview will play again at its end.' if enabled
                            else 'Link preview will stop on its last frame.')

    def live_from_url(self, url=''):
        """Drive the live character from a web video, without downloading it.

        Only the live preview works this way: it needs each frame once, in
        order, which a stream provides. Reconstruct 3D re-reads the whole clip
        several times and still needs the file, so it keeps downloading once.
        """
        if self.busy:
            return
        if getattr(self, 'camera_session', None) is not None:
            self.status.setText('A live source is already running. Press Stop first.')
            return
        if not url:
            # One link window for both entries: it previews, never downloads.
            self.choose_video_url()
            return
        if not url:
            return
        # Remembered so Download beside the video title can keep this clip for
        # the stages that re-read it; the preview itself saves nothing.
        self.source_url = url
        self.source_download.setEnabled(True)
        self.source_download.setToolTip(
            'Save this link into the downloads folder and open it as a video. '
            'Reconstruct 3D re-reads the clip, so it needs the file; the live '
            'preview does not.')
        self.status.setText('Connecting to the video link...')
        QApplication.processEvents()
        from .live_url import UrlSession
        try:
            session = UrlSession(url)
            session.start_preview()
        except Exception as error:          # noqa: BLE001 - reported, not raised
            self.status.setText('The link could not be opened: ' + str(error)[:200])
            return
        # The first frame proves the stream really plays before the view and
        # the workers are pointed at it.
        deadline = time.monotonic() + 20.0
        while session.frames == 0 and session.running and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)
        if session.frames == 0:
            detail = session.error or 'no frames arrived'
            session.stop()
            self.status.setText('The link did not play: ' + detail[:200])
            return
        self.start_live_camera(session, record=False)
        self.start_link_audio(session)
        self.status.setText('Live link ready. Press Find person to move the character with the video.')

    def start_live_box(self):
        """Track the performer in the live view with a box.

        A separate worker on YOLO only: the box has to appear within a second
        of going live, and the body solver takes minutes to load. Once the
        solver publishes its own boxes this worker is retired - one detector
        tracking the performer is enough.
        """
        if getattr(self, 'live_box_channel', None) is not None:
            return
        session = getattr(self, 'camera_session', None)
        if session is None:
            return
        from .live_channel import LiveChannel
        try:
            channel = LiveChannel('rgbmocap-box-%s' % uuid.uuid4().hex[:12],
                                  session.preview_width, session.preview_height, create=True)
        except OSError as error:
            self.fps_label.setText('Live tracking box unavailable: ' + str(error)[:120])
            return
        self.live_box_channel = channel
        python = ROOT / 'runtime_exmr' / 'python.exe'
        if not python.is_file():
            python = Path(sys.executable)
        log = (ROOT / 'cache' / 'live-box-worker.log').open('wb')
        self.live_box_log = log
        from .camera_capture import CREATE_NO_WINDOW, assign_to_job
        self.live_box_worker = subprocess.Popen(
            [str(python), '-B', '-m', 'mocap.live_worker', '--channel', channel.name,
             '--detect-only'],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        assign_to_job(self.live_box_worker)
        self.start_live_feed()
        self.live_box_timer = QTimer(self)
        self.live_box_timer.setInterval(100)
        self.live_box_timer.timeout.connect(self.poll_live_box)
        self.live_box_timer.start()

    def start_live_feed(self):
        """Publish the camera frames to the live workers.

        One timer for every live worker: the box detector, the body solver, or
        both while the solver is still loading.
        """
        if getattr(self, 'live_feed_timer', None) is not None:
            return
        self.live_feed_timer = QTimer(self)
        self.live_feed_timer.setInterval(33)
        self.live_feed_timer.timeout.connect(self.feed_live_frames)
        self.live_feed_timer.start()

    def stop_live_feed(self):
        timer, self.live_feed_timer = getattr(self, 'live_feed_timer', None), None
        if timer is not None:
            timer.stop()

    def close_live_box_worker(self):
        """Stop the box-only detector but keep painting boxes."""
        from .live_channel import STOP
        worker, self.live_box_worker = getattr(self, 'live_box_worker', None), None
        channel, self.live_box_channel = getattr(self, 'live_box_channel', None), None
        if channel is not None:
            channel.set_state(STOP)
        if worker is not None:
            try:
                worker.wait(10.0)
            except OSError:
                pass
            if worker.poll() is None:
                worker.kill()
            try:
                worker.wait(5.0)
            except OSError:
                pass
        log, self.live_box_log = getattr(self, 'live_box_log', None), None
        if log is not None:
            log.close()
        if channel is not None:
            channel.close()

    def stop_live_box(self):
        """Stop live tracking: the box, its worker and its timer."""
        timer, self.live_box_timer = getattr(self, 'live_box_timer', None), None
        if timer is not None:
            timer.stop()
        self.close_live_box_worker()
        self.video_view.set_live_box(None)

    def poll_live_box(self):
        """Paint the performer's box: the body solver's once it tracks, the
        detector's before that."""
        box = None
        solver = getattr(self, 'live_channel', None)
        if solver is not None:
            box = solver.box()
            if box is not None:
                self.close_live_box_worker()
        if box is None:
            boxes = getattr(self, 'live_box_channel', None)
            box = boxes.box() if boxes is not None else None
        self.video_view.set_live_box(None if box is None else self.live_box_pixels(box))

    def live_box_pixels(self, box):
        """A worker box (preview pixels) in the capture pixels the view draws in.

        The workers see the small preview frame the channel carries, while the
        video view scales overlay boxes against the full capture size - so a
        box passed through unscaled lands at half size in the top-left quarter.
        """
        session = getattr(self, 'camera_session', None)
        if session is None or not session.preview_width:
            return box[:4]
        scale = session.width / float(session.preview_width)
        return [float(value) * scale for value in box[:4]]

    def live_person_icon(self, box=None):
        """A small picture of the performer for the live People row."""
        session = getattr(self, 'camera_session', None)
        frame = session.snapshot() if session is not None else None
        if frame is None:
            return None
        import cv2
        crop = frame
        if box is not None:
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
            x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
            y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
            if x1 > x0 and y1 > y0:
                crop = frame[y0:y1, x0:x1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format_RGB888).copy()
        return image.scaled(48, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def live_character_files(self):
        """The open take's own prepared character: its rig asset and skin.

        A prepared result writes both beside the motion data, and the live
        preview reuses them so it presents that person's character - the same
        body the 3D view shows for the result - instead of a neutral stand-in.
        """
        project = getattr(self, 'project', None)
        if project is None:
            return None
        project = Path(project)
        # This project belongs to the motion model currently selected in the AI
        # Motion Model control, so its own prepared rigs are that model's.
        # FlowHMR prepares its native SMPL-H rig with a generation token
        # (initial_flow-<token>-rig-asset.npz); every other model writes
        # <result>_rig-<token>-asset.npz. Both are this take's own character.
        assets = (list(project.glob('*_rig-*-asset.npz'))
                  + list(project.glob('*-rig-asset.npz')))
        if not assets:
            return None
        wanted = ''
        try:
            manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
            wanted = (manifest.get('pipeline') or {}).get('result') or ''
        except (OSError, ValueError):
            wanted = ''
        def kind(path):
            prefix = path.name[:-len('-asset.npz')].split('_rig-')[0]
            return prefix.split('_flow-')[0]
        chosen = [asset for asset in assets if wanted and kind(asset) == wanted]
        asset = (chosen or sorted(assets, key=lambda path: path.stat().st_mtime_ns))[-1]
        binding = project / (asset.name[:-len('-asset.npz')] + '-display-binding.npz')
        # FlowHMR keeps its own SMPL-H surface, so there is no SMPL-X display
        # binding beside it: the live worker then skins the base surface with
        # this take's rig, which still brings that person's proportions.
        return (asset, binding if binding.is_file() else None)

    def start_live_body(self):
        """Present the live character: feed it frames and follow its solves.

        The solver may already be loading from when the source opened (see
        warm_live_solver), so this only has to hand the stream over unless
        nothing was warmed.
        """
        if getattr(self, 'live_channel', None) is None and not self._start_live_worker():
            return
        self.live_warmup = False
        self._live_character_ready = False
        self.live_shown = 0
        self.start_live_feed()
        timer = getattr(self, 'live_poll_timer', None)
        if timer is None:
            timer = self.live_poll_timer = QTimer(self)
            timer.setInterval(120)
            timer.timeout.connect(self.poll_live_body)
        timer.start()
        # The first pose cannot appear until the solver has warmed up its own
        # frames, so the view counts that wait out in numbers (step, percent,
        # seconds) instead of staying bare.
        self._live_progress_started = time.monotonic()
        self.update_live_progress()
        session = getattr(self, 'camera_session', None)
        self.filename.setText(f'{getattr(session, "label", "Live camera")} · '
                              'the character is loading in the 3D view')

    def update_live_progress(self):
        """The 3D view's wait numbers: the worker's step, percent, elapsed time.

        The model load and the first inference are the only slow part of a live
        session; naming the step and counting the seconds tells the user the
        solver is working rather than stuck.
        """
        view = self.motion_view
        started = getattr(self, '_live_progress_started', None)
        seconds = time.monotonic() - started if started is not None else 0.0
        report = {}
        path = getattr(self, '_live_progress_file', None)
        if path is not None:
            try:
                report = json.loads(Path(path).read_text(encoding='utf-8'))
            except (OSError, ValueError, TypeError):
                report = {}
        if not isinstance(report, dict):
            report = {}
        try:
            step = int(report.get('step', 0) or 0)
            steps = int(report.get('of', 0) or 0) or 5
            within = min(max(float(report.get('within', 0.0) or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            step, steps, within = 0, 5, 0.0
        percent = 0.0 if step <= 0 else (min(step, steps) - 1 + within) / steps * 100.0
        view.live_progress = {
            'title': 'Live character is starting',
            'stage': str(report.get('detail') or 'Waiting for the solver'),
            'percent': percent,
            'seconds': seconds,
            'step': step,
            'steps': steps,
        }
        view.update()

    def warm_live_solver(self):
        """Present the live character with the source, no button needed.

        The channel exists as soon as the source opens, so the solver works on
        the video while it plays, and the view shows the character from the
        first solved pose on. Find person is then a confirmation, not the
        switch that starts it.
        """
        session = getattr(self, 'camera_session', None)
        if session is None or not session.running:
            return False
        if getattr(self, 'live_channel', None) is not None:
            return False
        self.start_live_body()
        self.live_warmup = True
        self.status.setText('The live character follows the source once its first '
                            'pose is ready.')
        return True

    def preloaded_worker_alive(self):
        worker = getattr(self, 'preload_worker', None)
        return worker is not None and worker.poll() is None

    def preload_live_solver(self):
        """Load the live models at app start, before any source is opened.

        The worker loads YOLO and SAM 3D Body, then idles until a live channel
        is handed over, so the first Find person of the session moves the
        character instead of waiting for a model load.
        """
        if getattr(self, 'camera_session', None) is not None:
            return False
        if getattr(self, 'live_channel', None) is not None:
            return False
        if self.preloaded_worker_alive():
            return True
        if getattr(self, 'busy', False):
            # A reconstruction owns the GPU; preloading would compete with it.
            return False
        python = ROOT / 'runtime_exmr' / 'python.exe'
        if not python.is_file():
            python = Path(sys.executable)
        try:
            log = (ROOT / 'cache' / 'live-preload.log').open('wb')
        except OSError:
            return False
        from .camera_capture import CREATE_NO_WINDOW, assign_to_job
        from .live_channel import progress_file
        try:
            worker = subprocess.Popen([str(python), '-B', '-m', 'mocap.live_worker', '--preload',
                                       '--progress', str(progress_file(ROOT, 'preload'))],
                                      cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=CREATE_NO_WINDOW)
        except OSError:
            log.close()
            return False
        assign_to_job(worker)
        self.preload_worker, self.preload_log = worker, log
        self.preload_progress_file = progress_file(ROOT, 'preload')
        return True

    def adopt_preloaded_worker(self):
        """Hand the live channel to a worker whose models are already loaded."""
        if not self.preloaded_worker_alive():
            self.stop_preloaded_worker()
            return False
        marker = ROOT / 'cache' / 'live-warm-channel.txt'
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(self.live_channel.name, encoding='utf-8')
        except OSError:
            return False
        self.live_worker, self.preload_worker = self.preload_worker, None
        self.live_worker_log, self.preload_log = getattr(self, 'preload_log', None), None
        # The models are already loaded: the wait from here is the first pose,
        # and this worker reports that step itself.
        self._live_progress_file = getattr(self, 'preload_progress_file', None)
        return True

    def stop_preloaded_worker(self):
        """Release the app-start worker, or forget one that already exited."""
        worker, self.preload_worker = getattr(self, 'preload_worker', None), None
        log, self.preload_log = getattr(self, 'preload_log', None), None
        live_marker(ROOT).unlink(missing_ok=True)
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
                worker.wait(5.0)
            except (OSError, subprocess.TimeoutExpired):
                worker.kill()
                try:
                    worker.wait(3.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if log is not None:
            try:
                log.close()
            except OSError:
                pass
        return worker is not None

    def _start_live_worker(self):
        """One worker process solving SAM 3D Body per frame, not yet presented.

        Preview only: the timeline is left alone and no frame is stored, so a
        rehearsal costs nothing and Record window is the only capture.
        """
        from .live_body_solve import LiveBody, cached_rest
        from .live_channel import LiveChannel, progress_file
        from .models import BY_KEY
        if getattr(self, 'live_channel', None) is not None:
            return getattr(self, 'live_worker', None) is not None
        session = getattr(self, 'camera_session', None)
        if session is None or not session.running:
            return False
        try:
            channel = LiveChannel('rgbmocap-live-%s' % uuid.uuid4().hex[:12], session.preview_width,
                                  session.preview_height, create=True)
        except OSError as error:
            self.filename.setText(f'Live character unavailable: {error}')
            return False
        self.live_channel = channel
        self._live_mesh_asset = None
        self._live_mesh_binding = None
        # The character the view already presents for the open take IS the live
        # character: its prepared rig brings that person's proportions, and its
        # skin binding is the one the result view renders with. Only without a
        # take does the neutral base character stand in.
        character_binding = getattr(self.motion_view, 'mesh_binding', None)
        binding_path = ''
        try:
            if character_binding and character_binding.get('key'):
                binding_path = str(character_binding['key'][0])
        except (AttributeError, IndexError, TypeError):
            binding_path = ''
        prepared = self.live_character_files()
        rig_path = str(prepared[0]) if prepared else ''
        # A binding left over from another model's project would dress the live
        # character as that model's person, so only this project's own skin or
        # the take's prepared binding counts.
        project = Path(self.project).resolve() if getattr(self, 'project', None) else None
        if binding_path and project is not None and not Path(binding_path).resolve().is_relative_to(project):
            binding_path = ''
        binding_path = binding_path if Path(binding_path).is_file() else (
            str(prepared[1]) if prepared and prepared[1] else '')
        supplied = {}
        if rig_path and Path(rig_path).is_file():
            try:
                with np.load(rig_path, allow_pickle=False) as data:
                    supplied = dict(rest=data['rest'], parents=data['parents'], ends=data['ends'],
                                    alignment=data['anatomical_calibration'])
            except (OSError, ValueError, KeyError):
                supplied = {}
        if supplied and binding_path:
            try:
                with np.load(binding_path, allow_pickle=False) as data:
                    self._live_mesh_binding = {key: data[key] for key in
                                               ('vertices', 'frames', 'local', 'rest',
                                                'parents', 'betas')}
                self._live_mesh_binding['key'] = ('live-binding', Path(binding_path).stat().st_mtime_ns)
            except (OSError, ValueError, KeyError):
                self._live_mesh_binding = None
                supplied = {}
        try:
            body = (LiveBody(rig=supplied) if supplied
                    else LiveBody(cached_rest(ROOT, BY_KEY['smplx'].path)))
        except Exception as error:
            self.filename.setText(f'Live character unavailable: {error}')
            self.stop_live_body()
            return False
        # Imported characters are posed from this same rig, in the 3D view's own
        # character layer: they move with the video as each pose arrives.
        self.clear_live_characters()
        self._live_source_rest = np.asarray(body.rig['rest'][:22], dtype=np.float64).copy()
        if getattr(self, 'follow_source', False):
            self.prepare_live_characters()
        # Bones and character are the view's own display modes; live only feeds
        # them, so the overlay and mesh toggle behave as they do for a result.
        from .bone_overlay import geometry
        self._live_bones_visible = self.motion_view.show_bones
        self.live_bones_asset = geometry(body.rig['rest'], body.rig['parents'], body.rig['ends'],
                                         ('live', channel.name))
        self.motion_view.target_asset = {'bones': self.live_bones_asset,
                                         'key': self.live_bones_asset['key'], 'mode': 'rig'}
        self.motion_view.target_frames = channel.rig_frames
        self.motion_view.frame_stamp = 0
        # The solver needs the ExMR runtime's packages (braceexpand, ultralytics);
        # fall back to this process's interpreter in developer layouts.
        # Models loaded at app start need only the new channel: hand it over
        # instead of loading them a second time.
        if not supplied and self.adopt_preloaded_worker():
            self._live_character_ready = False
            return True
        self.stop_preloaded_worker()
        python = ROOT / 'runtime_exmr' / 'python.exe'
        if not python.is_file():
            python = Path(sys.executable)
        log = (ROOT / 'cache' / 'live-worker.log').open('wb')
        self.live_worker_log = log
        from .camera_capture import CREATE_NO_WINDOW, assign_to_job
        self._live_progress_file = progress_file(ROOT, channel.name)
        command = [str(python), '-B', '-m', 'mocap.live_worker', '--channel', channel.name,
                   '--progress', str(self._live_progress_file)]
        if supplied:
            command += ['--rig', rig_path]
            if binding_path and Path(binding_path).is_file():
                command += ['--binding', binding_path]
        self.live_worker = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        assign_to_job(self.live_worker)
        self._live_character_ready = False
        return True

    def feed_live_frames(self):
        channel = getattr(self, 'live_channel', None)
        boxes = getattr(self, 'live_box_channel', None)
        session = getattr(self, 'camera_session', None)
        if session is None or (channel is None and boxes is None):
            return
        if getattr(session, 'ended', False) and not getattr(self, '_live_source_ended', False):
            # A clip link ends by itself; a camera never does. Say so instead
            # of leaving a frozen frame under a live label.
            self._live_source_ended = True
            self.filename.setText('Live link finished · open the link again to replay it')
        frame = session.snapshot()
        if frame is None:
            return
        if channel is not None:
            channel.write_frame(frame)
        if boxes is not None:
            boxes.write_frame(frame)

    def poll_live_body(self):
        channel = getattr(self, 'live_channel', None)
        if channel is None:
            return
        from .live_channel import CALIBRATING, FAILED
        if channel.state() == CALIBRATING:
            self.fps_label.setText('Live camera · the solver is calibrating depth')
            return
        if channel.state() == FAILED:
            self.live_poll_timer.stop()
            self.filename.setText('Live solver failed: ' + channel.error()[:140])
            return
        self.update_live_progress()
        count = channel.count()
        if count <= 0:
            worker = getattr(self, 'live_worker', None)
            if worker is not None and worker.poll() is not None:
                # The process that solves the live body is gone, so nothing can
                # arrive on this channel. Say so instead of waiting forever, and
                # leave Find person able to start the character again.
                timer, self.live_poll_timer = self.live_poll_timer, None
                if timer is not None:
                    timer.stop()
                self.live_channel = None
                self.live_worker = None
                self.filename.setText('The live character stopped. Press Find person to start it again.')
            return
        if not getattr(self, '_live_person_row', False):
            # The performer being followed is named in People, so the row and
            # the 3D view agree on who is on screen.
            self._live_person_row = True
            box = channel.box()
            self.show_live_person(self.live_person_icon(box))
        if getattr(self.motion_view, 'live_message', '') or getattr(self.motion_view, 'live_progress', None):
            self.motion_view.live_message = ''
            self.motion_view.live_progress = None
        if count == getattr(self, 'live_shown', 0):
            return
        from .preview_data import motion_summary
        # The newest frame only: live is a mirror of the performer, not a take.
        # Filling the timeline made the preview look (and cost) like a recording,
        # and nothing is captured until Record window.
        joints = channel.joints_latest(count)
        # The floor and framing come from the recent window, so they do not bob
        # with every solved frame.
        recent = channel.joints_view(count)
        self.motion_view.live_body_mode = True
        self.motion_view.set_motion(joints, motion_summary(recent[-30:]), None)
        # The follower wears the BASE CHARACTER: the take's own SMPL-X mesh,
        # driven by the same packed frames a Prepare Mesh result uses.
        if self._live_mesh_asset is None:
            assets = sorted(ROOT.glob('models/body_models/smplx/.mesh-lbs-v*.npz'))
            if assets:
                with np.load(assets[-1], allow_pickle=False) as data:
                    self._live_mesh_asset = {key: data[key] for key in
                                             ('packed', 'faces', 'offsets', 'neighbors', 'vertex_count')}
                self._live_mesh_asset['key'] = ('live-mesh', assets[-1].name)
        if self._live_mesh_binding is None:
            # The follower's base character is skinned from this binding; the
            # worker writes it before the first inference.
            package = ROOT / 'cache' / 'live-binding.npz'
            if package.is_file():
                try:
                    with np.load(package, allow_pickle=False) as data:
                        self._live_mesh_binding = {key: data[key] for key in
                                                   ('vertices', 'frames', 'local', 'rest',
                                                    'parents', 'betas')}
                    self._live_mesh_binding['key'] = ('live-binding', package.stat().st_mtime_ns)
                except (OSError, ValueError, KeyError):
                    self._live_mesh_binding = None
        # An imported character is the model of the session: it takes the base
        # body's place while it is visible, and the base body comes back when
        # every imported character is hidden (or none has a live rig yet).
        poses = self.update_live_characters(channel.rig_latest())
        # Display > Characters Only hides the base body in the live view too.
        hide_body = bool(poses) and self.motion_view.characters_only_active()
        if poses and not getattr(self, '_live_character_ready', False):
            self._live_character_ready = True
            names = ', '.join(c['name'] for c in self.characters if c['id'] in poses)
            self.status.setText(f'{names} follows the video.')
        if hide_body:
            self.motion_view.mesh_frames = None
            self.motion_view.mesh_asset = None
            self.motion_view.mesh_binding = None
        body_shown = (not hide_body
                      and self._live_mesh_asset is not None
                      and self._live_mesh_binding is not None
                      and channel.mesh_count() > 0)
        if body_shown:
            self.motion_view.mesh_asset = self._live_mesh_asset
            self.motion_view.mesh_frames = channel.mesh_latest
            self.motion_view.mesh_binding = self._live_mesh_binding
            if not getattr(self, '_live_character_ready', False):
                self._live_character_ready = True
                self.status.setText('Live character is following the video.')
        # Bones above the skinned character: the view's own display modes, as
        # they are for a prepared result.
        self.motion_view.index = 0
        self.motion_view.frame_stamp = count
        self.motion_view.update()
        self.live_shown = count
        label = getattr(getattr(self, 'camera_session', None), 'label', 'Live camera')
        self.fps_label.setText(f'{label} · solver {channel.worker_fps():.1f} fps')
        self.filename.setText(f'{label} · the character follows the video in the 3D view'
                              if label.startswith('Live link')
                              else 'Live camera · the character follows you in the 3D view')

    def restart_live_character(self):
        """Bring the live character back with the model that is now selected.

        Find person starts the character once and then leaves it running, so a
        model switch would keep the live view following the previous model's
        rig. Restarting it is what makes the live view present the selected
        model's own character.
        """
        if getattr(self, 'live_channel', None) is None:
            return
        self.stop_live_body()
        session = getattr(self, 'camera_session', None)
        if session is None or not session.running:
            return
        try:
            self.start_live_body()
        except Exception as error:          # noqa: BLE001 - reported, not fatal
            self.status.setText('Live character failed: ' + str(error)[:160])

    def stop_live_body(self):
        channel = getattr(self, 'live_channel', None)
        if channel is None:
            return
        from .live_channel import STOP
        channel.set_state(STOP)
        worker = getattr(self, 'live_worker', None)
        if worker is not None:
            # A worker that is still loading its model ignores the stop state for
            # a while; timing out is the normal case and must not abort the stop.
            try:
                worker.wait(10.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if worker.poll() is None:
                worker.kill()
            try:
                worker.wait(5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.live_worker = None
        log = getattr(self, 'live_worker_log', None)
        if log is not None:
            log.close()
            self.live_worker_log = None
        timer, self.live_poll_timer = getattr(self, 'live_poll_timer', None), None
        if timer is not None:
            timer.stop()
        self.live_channel = None
        self.clear_live_characters()
        self.live_shown = 0
        self.live_warmup = False
        live_marker(ROOT).unlink(missing_ok=True)
        self.motion_view.live_message = ''
        self.motion_view.live_progress = None
        self._live_progress_started = None
        self._live_progress_file = None
        self.motion_view.live_body_mode = False
        self.motion_view.set_motion(None, None, None)
        self.motion_view.target_frames = None
        self.motion_view.target_asset = None
        self.motion_view.frame_stamp = None
        self.motion_view.show_bones = getattr(self, '_live_bones_visible', True)
        self.motion_view.mesh_frames = None
        self.motion_view.mesh_asset = None
        self.motion_view.mesh_binding = None
        self._live_mesh_binding = None

    def stop_live_camera(self):
        """Stop the live camera and its capture."""
        self.source_audio.stop()
        self.stop_live_body()
        self.stop_live_box()
        self.stop_live_feed()
        session = getattr(self, 'camera_session', None)
        if session is not None:
            session.stop()
        self.camera_session = None
        self.video_view.close_camera()
        self.show_link_playback(None)
        if getattr(self, '_live_person_row', False):
            # Nothing was tracked or saved, so the live row leaves with it.
            self._live_person_row = False
            self.reset_people()
        self.live_record_button.setVisible(False)
        self.live_stop_button.setVisible(False)
        self.fps_label.setText('No clip loaded')
        self.filename.setText('Live camera stopped')

        # Keep the models warm for the next live source.
        QTimer.singleShot(2500, self.preload_live_solver)
        self.update_people_readiness()

    def record_live_window(self):
        """Record one window from the live camera, load it, and resume live.

        The window is a capture of its own so its first and last frame are
        exactly the user's press: a cut out of the running live capture cannot
        do that, because this camera encodes slower than it films and the file
        falls behind the picture (measured: 12-21 fps written at 1080p). The
        live character keeps following while the window records - the recorder
        writes the same frames to the live solver.
        """
        session = getattr(self, 'camera_session', None)
        if session is None or self.busy:
            return
        # The live session is running by definition - recording REPLACES it:
        # the camera is freed, one window is captured, and live resumes.
        from . import camera_capture
        device = session.device
        width, height = session.capture_width, session.capture_height
        fps = session.fps
        preview_width = session.preview_width
        rotation = session.rotation
        session.stop()
        self.video_view.close_camera()
        self.camera_session = None
        record_session = None
        for _attempt in range(6):
            record_session = camera_capture.Session(
                camera_capture.default_path(ROOT),
                device, width, height, fps, preview_width=preview_width,
                rotation=rotation)
            record_session.start()
            time.sleep(2.5)
            if record_session.running:
                break
            record_session.stop()
            record_session = None
            time.sleep(1.0)
        if record_session is None or not record_session.running:
            self.filename.setText('The camera would not reopen for recording.')
            return
        self.camera_session = record_session
        self.video_view.open_camera(record_session)
        self.live_stop_button.setVisible(True)
        self.live_record_button.setVisible(False)
        self.filename.setText(f'Recording a {self.live_window_seconds}s window...')

        def finish():
            report = record_session.stop()
            self.camera_session = None
            path = report.get('path')
            if path and Path(path).is_file():
                # The live presenter would keep repainting the capture's last
                # frame over the freshly decoded window.
                self.video_view.close_camera()
                self.load_video(Path(path))
                # The file decoder's metadata has to land before the camera
                # replaces it: open_camera shuts the decoder down, and a
                # decoder killed before its metadata leaves the load
                # unreadable forever. Wait for the readable flag.
                waited = 0.0
                while not self.video_readable and waited < 15.0:
                    QApplication.processEvents()
                    time.sleep(0.1)
                    waited += 0.1
            # resume live: reopen the camera and keep the loop. The dshow
            # release after a capture takes a few seconds - the retries have
            # to outlast it.
            resume = camera_capture.Session(
                camera_capture.default_path(ROOT),
                device, width, height, fps, preview_width=preview_width,
                rotation=rotation)
            for _attempt in range(8):
                resume.start_preview()
                time.sleep(2.5)
                if resume.running:
                    break
                resume.stop()
                resume = None
                time.sleep(1.0)
            if resume is not None and resume.running:
                self.start_live_camera(resume)
                self.filename.setText('Live camera resumed · the recorded window is loading')
            # One press gives the take and its motion: the window that was just
            # kept runs through the normal reconstruction right away.
            if path and Path(path).is_file():
                self.status.setText('Window kept · reconstructing its motion...')
                self.run_full()

        QTimer.singleShot(int(self.live_window_seconds * 1000), finish)

    def choose_video_url(self, url=''):
        """Preview one video link as the live source without saving it.

        Full reconstruction re-reads a clip, so it needs the file: the
        Download button beside the video title saves this link once.
        """
        from .video_url import VideoUrlDialog
        if self.busy:
            return
        dialog = VideoUrlDialog(self, url)
        try:
            if dialog.exec() != QDialog.Accepted:
                return
            if dialog.path is not None:
                self.load_video(Path(dialog.path))
            elif dialog.url:
                self.live_from_url(dialog.url)
        finally:
            dialog.deleteLater()

    def download_source(self):
        """Keep the previewed link on disk and open it as a normal video.

        The live preview streams a link frame by frame, which is all the
        character needs. Every reconstruction stage reads the clip several
        times, so that path still needs one download - on request, here.
        """
        url = getattr(self, 'source_url', '')
        if not url:
            self.status.setText('Preview a link first (File > Open Video from URL).')
            return
        if self.busy or getattr(self, 'source_fetch', None) is not None:
            return
        from .paths import DOWNLOADS
        from .video_url import FetchVideo
        fetch = FetchVideo(url, DOWNLOADS, self)
        self.source_fetch = fetch
        self.source_download.setEnabled(False)
        fetch.progress.connect(self.source_download_progress)
        fetch.fetched.connect(self.source_downloaded)
        fetch.failed.connect(self.source_download_failed)
        fetch.start()
        self.status.setText('Downloading the link into the downloads folder...')

    def source_download_progress(self, done, total):
        if total > 0:
            self.status.setText(f'Downloading the link {done / 1024 ** 2:.1f} of '
                                f'{total / 1024 ** 2:.1f} MB')
        else:
            self.status.setText('Downloading the link...')

    def source_downloaded(self, path):
        self.close_source_fetch()
        self.status.setText('Link saved. Opening it as a video...')
        self.load_video(Path(path))

    def source_download_failed(self, message):
        self.close_source_fetch()
        self.status.setText('The link could not be downloaded: ' + str(message)[:200])

    def close_source_fetch(self):
        fetch, self.source_fetch = getattr(self, 'source_fetch', None), None
        if fetch is not None:
            try:
                fetch.wait()
            except (OSError, RuntimeError):
                pass
            fetch.deleteLater()
        self.source_download.setEnabled(bool(getattr(self, 'source_url', '')))

    def load_video(self, path):
        if self._pear_active:return self.load_pear_videos([path])
        if self.model_choice.currentData()=='mamma':
            return self.load_mamma_videos([path])
        import traceback as _tb
        try:
            # Opening another video replaces the live view. A source left
            # running behind it would keep pulling camera frames or network
            # video for nobody, so it is stopped first.
            if getattr(self, 'camera_session', None) is not None:
                self.stop_live_camera()
            if self.workspace_root is not None:self.remember_workspace_model()
            self.clear_mamma_inputs()
            self.workspace_root=None;self._workspace_video_reference=False;self._workspace_camera=None
            self.stop_playback()
            self.clear_preview()
            self.reset_people()
            self.motion_view.auto_fit = True
            self.video_path, self.project, self.info = path, None, None
            self.setWindowTitle('RGB MoCap | Motion Workspace')
            self.flow_sampling.setCurrentIndex(self.flow_sampling.findData(0))
            self.fast_sam.setChecked(True)
            self.flow_batch4.setChecked(True)
            self.flow_gpu_crop.setChecked(True)
            self.flow_hand_stride4.setChecked(True)
            self.flow_hamer.setChecked(False)
            self.exmr_camera.setCurrentIndex(0)
            self.exmr_smoothing.setChecked(True)
            self.camera_tools.restore_background({})
            self.handflow_refine.blockSignals(True)
            self.handflow_refine.setChecked(self.model_choice.currentData()=='gvhmr')
            self.handflow_refine.blockSignals(False)
            self._tool_refinement={'gvhmr':(True,True,True,True)}
            for option in (self.htd_refine, self.footmr_refine, self.motion_correction):
                option.blockSignals(True)
                option.setChecked(True)
                option.blockSignals(False)
            self.active_result = self.selected_result()
            self.restore_correction_options({})
            self.source_offset = self.source_end = 0.0
            self.video_readable = False
            self.slider.setRange(0, 0)
            self.filename.setText(f"{path.name} · Opening video...")
            self.video_view.open(path)
            self.update_readiness()
        except Exception as error:
            print('LOAD: ERROR', _tb.format_exc()[-600:], flush=True)
            QMessageBox.warning(self, "Video Error", str(error))

    def video_ready(self, info):
        if self.pear_restore_media_ready():return
        if self.pear_video_ready(info):return
        if self.mamma_video_ready(info):return
        self.video_readable = True
        self.slider.set_media(self.info["fps"] if self.info else info["fps"])
        if self.project is None and not self._workspace_video_reference:
            self.info = info
            self.slider.setRange(0, max(0, info["frames"] - 1))
            self.source_offset, self.source_end = 0.0, info["duration"]
            self.slider.reset_trim()
            self.clip_changed(self.slider.clip_in, self.slider.clip_out)
            self.filename.setText(f"{self.video_path.name} · {info['width']}×{info['height']} · {info['fps']:.2f}fps · {info['duration']:.2f} s")
            self.fps_label.setText(f"Source {info['fps']:.2f} fps")
            self.seek(0)
            if self.workspace_root is not None:
                from .project_workspace import read
                self.source_offset=float(read(self.workspace_root).get('time_offset',0.))
                self.source_end=self.source_offset+info['duration']
                self.restore_workspace_view(getattr(self,'_workspace_pending_view',{}))
        self.update_readiness()

    def video_failed(self, message):
        self.video_readable = False
        self.stop_playback()
        self.status.setText("Video preview: " + message)
        self.update_readiness()

    def choose_project(self):
        name, _ = QFileDialog.getOpenFileName(self, "Open Project", str(PROJECTS), "Project (project.json)")
        if name:
            from .project_workspace import root_for
            try:
                project=Path(name).parent
                if root_for(project) is None:
                    raise ValueError('This project uses the previous folder format. Open its video to create a new model-folder project.')
                self.load_project(project)
            except (OSError,ValueError,KeyError) as error:self.status.setText('Cannot open project: '+str(error))

    @batch_readiness
    def load_project(self, project):
        if self._pear_active:
            self.pear_leave()
            self.model_choice.blockSignals(True);self.model_choice.setCurrentIndex(self.model_choice.findData(self._selected_tool));self.model_choice.blockSignals(False)
        from .project_workspace import is_workspace
        if is_workspace(project):return self.open_workspace(project)
        from .project_storage import project_to_open, source_video
        try:
            project, workspace = project_to_open(project)
        except (OSError, ValueError) as error:
            self.status.setText('Cannot open project: '+str(error));return False
        from .motion_tools import require_supported_project, UnsupportedMotionTool
        try:
            require_supported_project(project)
        except UnsupportedMotionTool as error:
            self.status.setText(str(error))
            return False
        manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
        if manifest.get('pipeline',{}).get('model')=='prompthmr':
            group=(project/manifest.get('people_root','.')).resolve()
            if (group/'prompthmr_pending.json').exists():
                self.status.setText('PromptHMR publication is incomplete. Reconstruct again; previous projects were preserved.');return False
            self.restore_prompthmr_options(manifest.get('prompthmr_options',{}))
        if manifest.get('pipeline',{}).get('model')=='mamma':
            group=(project/manifest.get('people_root','.')).resolve()
            if (group/'mamma_pending.json').exists():
                self.status.setText('MAMMA publication is incomplete. Reconstruct again to finish; previous projects were preserved.')
                return False
        self.stop_playback()
        self.clear_preview()
        self.motion_view.auto_fit = True
        self.project = project
        self.camera_tools.restore_background({})
        self.workspace_project_loaded(project,manifest)
        from .project_storage import storage_root
        saved_root=storage_root(project,manifest)
        self.setWindowTitle('RGB MoCap | '+(saved_root.name if saved_root else 'Motion Workspace'))
        pipeline = manifest.get("pipeline", {})
        from .flow_camera import saved_interval
        from .flow_sam import saved_options
        flow=saved_options(project,manifest)
        self.flow_hamer.setChecked(False)
        self.fast_sam.setChecked(flow['fast_sam'])
        self.flow_batch4.setChecked(flow['sam_batch4'])
        self.flow_gpu_crop.setChecked(flow['sam_gpu_crop'])
        self.flow_hand_stride4.setChecked(flow['sam_hand_stride4'])
        self.flow_hamer.setChecked(flow['hamer'])
        from .exmr_options import saved as saved_exmr
        exmr=saved_exmr(project,manifest)
        self.exmr_camera.setCurrentIndex(self.exmr_camera.findData(exmr['static_camera']))
        self.exmr_smoothing.setChecked(exmr['smoothing']=='global')
        self.flow_sampling.setCurrentIndex(max(0,self.flow_sampling.findData(saved_interval(project,manifest))))
        self.model_choice.blockSignals(True)
        self.model_choice.setCurrentIndex(self.model_choice.findData(pipeline.get('model','gvhmr')))
        self.model_choice.blockSignals(False)
        opening=self._workspace_loading;self._workspace_loading=True
        try:self.select_model()
        finally:self._workspace_loading=opening
        has_refined = any((project / name).exists() for name in ("opt_results.pt", "refined_joints.npz", "refined_mesh.json"))
        has_motion = has_refined or any((project / name).exists() for name in ("hmr4d_results.pt", "initial_joints.npz"))
        result = pipeline.get("result", "refined" if has_refined else "initial")
        saved_selection=bool(pipeline.get('result'))
        self.handflow_refine.blockSignals(True)
        self.handflow_refine.setChecked(has_handflow(result) if has_motion or saved_selection else pipeline.get('model','gvhmr')=='gvhmr')
        self.handflow_refine.blockSignals(False)
        self.htd_refine.blockSignals(True)
        self.htd_refine.setChecked(result_options(result)[0] if has_motion or saved_selection else True)
        self.htd_refine.blockSignals(False)
        self.footmr_refine.blockSignals(True)
        # Existing projects keep their saved selection; new reconstructions default all on.
        self.footmr_refine.setChecked(result_options(result)[1] if has_motion or saved_selection else True)
        self.footmr_refine.blockSignals(False)
        self.motion_correction.blockSignals(True)
        self.motion_correction.setChecked(result.startswith('corrected_') if has_motion or saved_selection else True)
        self.motion_correction.blockSignals(False)
        self.restore_correction_options(manifest.get('correction',{}))
        self.active_result = self.selected_result()
        self.video_path = source_video(project, manifest)
        self.video_readable = False
        fps = manifest.get("fps", 30)
        self.info = {"fps": fps, "frames": manifest["frames"], "duration": manifest["frames"] / fps}
        self.source_offset, self.source_end = manifest["start"], manifest["end"]
        self.slider.set_media(fps)
        # MAMMA restores its selected camera below. Opening the actor's default
        # clip first discards that decoder and its cache on every row selection.
        if not (pipeline.get('model')=='mamma' and self.mamma_inputs and self.mamma_thumbnail_buttons):
            self.video_view.open(select_preview(project), (manifest["width"], manifest["height"]))
        self.slider.setRange(0, manifest["frames"] - 1)
        self.filename.setText(f"{self.video_path.name}  ·  Project loaded  ·  {manifest['frames']} frames")
        self.slider.reset_trim()
        self.clip_changed(self.slider.clip_in, self.slider.clip_out)
        self.refresh_results()
        self.seek(0)
        self.fps_label.setText(f"Source {fps:.2f} fps")
        if (project / "motion.bvh").exists():
            self.status.setText("Motion loaded. Play the animation or adjust the video clip range.")
        if (project / "keypoints.npz").exists() and not (project / "hmr4d_results.pt").exists():
            self.status.setText("2D observations loaded. Prepare the body model to reconstruct full-body motion.")
        self.sync_people(project,manifest)
        if workspace:
            trim=workspace.get('trim')
            if trim and 0<=trim[0]<trim[1]<=manifest['frames']:
                self.slider.clip_in,self.slider.clip_out=trim
                self.clip_changed(*trim)
            self.slider.setValue(max(0,min(int(workspace.get('frame',0)),manifest['frames']-1)))
            self.seek(self.slider.value())
        self.update_readiness()
        self.mamma_project_loaded(project,manifest)
        return True

    def seek(self, index):
        self.pear_seek(index)
        self.frame_number.blockSignals(True)
        self.frame_number.setValue(index + 1)
        self.frame_number.blockSignals(False)
        if self.timer.isActive() and not self.playback_tick:
            self.reset_play_clock()
        self.video_view.seek(index)
        fps = self.info["fps"] if self.info else 30
        total = self.info["duration"] if self.info else 0
        self.time_label.setText(f"{index/fps:05.2f} / {total:05.2f} s")

    def processing_range(self):
        """Map preview-local trim edges to the unchanged original source."""
        fps = self.info["fps"]
        first = self.source_offset + self.slider.clip_in / fps
        end = min(self.source_end, self.source_offset + self.slider.clip_out / fps)
        return first, end

    def clip_changed(self, first, end):
        if hasattr(self, "timer"):
            self.stop_playback()
        if self.info:
            start_seconds, end_seconds = self.processing_range()
            text = f"Clip: {first + 1}–{end}  |  {end_seconds-start_seconds:.2f} s"
            self.clip_label.setText(text)
            self.clip_label.setToolTip(tooltip_text(f"Video range: {start_seconds:.3f}–{end_seconds:.3f} s\nDrag the video track edges to trim playback and reconstruction. Existing motion files stay unchanged until you reconstruct again."))
        else:
            self.clip_label.setText("Clip: no video")

    def present_frame(self, index):
        if self._pear_active:
            self.pear_submit();return
        # 3D and overlays advance only when the corresponding decoded image is ready.
        self.motion_view.index = index
        self.motion_view.scene_time=self.source_offset+index/(self.info['fps'] if self.info else 30)
        self.motion_view.update()
        self.publish_live_frame(index)
        if self.timer.isActive():
            now = time.perf_counter()
            self.presented_times.append(now)
            while self.presented_times and now - self.presented_times[0] > 2:
                self.presented_times.popleft()
            if len(self.presented_times) > 2 and now - self.presented_times[0] >= 0.5:
                actual = (len(self.presented_times) - 1) / (now - self.presented_times[0])
                self.fps_label.setText(f"Source {self.info['fps']:.2f} | Play {actual:.1f} fps")

    def live_pose(self, index):
        """Canonical 22-joint pose of what is on screen, in app world space."""
        view = self.motion_view
        if view.people_mode and view.people_scene is not None:
            actors = view.scene_actors()
            chosen = next(((take, frame) for take, frame in actors
                           if take.ident == view.scene_selected), None)
            if chosen is None:
                chosen = actors[0] if actors else None
            return view.scene_joint_points(*chosen) if chosen else None
        if view.joints is None:
            return None
        return view.joints[max(0, min(int(index), len(view.joints) - 1))]

    def live_body_frame(self, index):
        """The take on screen whose base body can be streamed, with its frame."""
        from .live_body import for_project
        view = self.motion_view
        if getattr(view, 'people_mode', False) and view.people_scene is not None:
            actors = view.scene_actors()
            chosen = next(((take, frame) for take, frame in actors
                           if take.ident == view.scene_selected), None)
            if chosen is None:
                chosen = actors[0] if actors else None
            if chosen is None:
                return None
            take, frame = chosen
            body = for_project(take.project)
            return (body, int(frame), getattr(take, 'transform', None)) if body else None
        if self.project is None:
            return None
        body = for_project(self.project)
        return (body, int(index), None) if body else None

    # -- Face capture (ARKit) from the take's own video ------------------------
    def face_takes(self):
        """Take folders a face can be read from: input.mp4 beside tracking.npz."""
        projects = [self.project] if self.project is not None else []
        if self.people_state:
            from .people_store import safe_child
            projects += [safe_child(self.people_root, take['project'])
                         for person in self.people_state['people']
                         for take in person.get('takes', [])]
        return [Path(p) for p in dict.fromkeys(projects)
                if p is not None and (Path(p) / 'input.mp4').is_file()
                and (Path(p) / 'tracking.npz').is_file()]

    def extract_face(self):
        """Read ARKit face values for every take of this project (face.npz)."""
        if getattr(self, 'face_process', None) is not None:
            self.status.setText('Face capture is already running.');return
        takes = self.face_takes()
        if not takes:
            self.status.setText('Open a reconstructed project first: face capture reads the take videos.');return
        args = []
        for take in takes:
            args += ['--take', str(take)]
        try:
            cmd = worker_command('face', *args)
        except RuntimeError as error:
            self.status.setText(str(error).replace('Run Setup to repair this installation.',
                                                   'Run Setup_Face.cmd to install face capture.'))
            return
        process = QProcess(self);self.face_process = process
        env = QProcessEnvironment.systemEnvironment();env.insert('PYTHONUTF8', '1');env.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(env);process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        self._face_buffer = ''
        process.readyReadStandardOutput.connect(self.face_output)
        process.finished.connect(self.face_finished)
        self.status.setText('Face capture: starting (%d take(s))...' % len(takes))
        process.start(cmd[0], cmd[1:])

    def face_output(self):
        process = getattr(self, 'face_process', None)
        if process is None:return
        self._face_buffer += bytes(process.readAllStandardOutput()).decode('utf-8', 'replace')
        lines = self._face_buffer.split('\n');self._face_buffer = lines.pop()
        for line in lines:
            if not line.startswith('@@MOCAP@@'):continue
            try:event = json.loads(line[9:])
            except ValueError:continue
            if event.get('type') == 'progress':
                self.status.setText('%s: %d%%' % (event.get('message', 'Face capture'),
                                                  100 * event.get('current', 0) // max(event.get('total', 1), 1)))
            elif event.get('type') == 'done':
                rows = event.get('takes', [])
                found = sum(r.get('found', 0) for r in rows);frames = sum(r.get('frames', 0) for r in rows)
                errors = [r['error'] for r in rows if 'error' in r]
                self.status.setText(('Face capture: face found in %d of %d frames.' % (found, frames))
                                    + (' ' + errors[0] if errors else ''))
            elif event.get('type') == 'failed':
                self.status.setText('Face capture: ' + event.get('message', 'failed'))

    def face_finished(self, *_):
        self.face_output()
        self.face_process = None
        self._face_cache = {}
        self.log.append('Face capture: ' + self.status.text())

    def live_face_values(self, index):
        """ARKit values for the take on screen at this frame, or None."""
        found = self.live_body_frame(index)
        if found is None:
            return None
        body, frame, _transform = found
        path = Path(body.project) / 'face.npz'
        cache = self.__dict__.setdefault('_face_cache', {})
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        cached = cache.get(path)
        if cached is None or cached[0] != stamp:
            import numpy as np
            with np.load(path, allow_pickle=False) as data:
                cached = cache[path] = (stamp, np.asarray(data['values'], np.float32))
        values = cached[1]
        if not len(values):
            return None
        return values[max(0, min(int(frame), len(values) - 1))]

    def live_people(self, index):
        """Every person on screen except the one sent as the main character.

        The live link used to send only the selected person; the others now go
        with it, each as its own base body, so Blender shows the whole scene.
        """
        from .live_body import for_project
        view = self.motion_view
        if not (getattr(view, 'people_mode', False) and view.people_scene is not None):
            return []
        actors = view.scene_actors()
        if len(actors) < 2:
            return []
        chosen = next(((take, frame) for take, frame in actors if take.ident == view.scene_selected), None)
        if chosen is None:
            chosen = actors[0]
        names = getattr(view, 'scene_names', {}) or {}
        rows = []
        for take, frame in actors:
            if take is chosen[0]:
                continue
            body = for_project(take.project)
            if body is None:
                continue
            label = str(names.get(take.ident) or 'Person %s' % take.ident)
            rows.append((dict(ident=str(take.ident), label=label, blend=str(body.rest_file()),
                              palette=list(body.names), armature='RGBMoCap ' + label,
                              transform=getattr(take, 'transform', None), height=body.height,
                              rest_format='npz', frame=int(frame), fps=float(body.fps)),
                         body.matrices(frame)))
        return rows

    def publish_live_frame(self, index):
        # Cascadeur is a second, independent receiver: it is updated before the
        # Blender branch below, which returns early whenever it has something
        # to send.
        self.publish_cascadeur_frame(index)
        link = getattr(self, 'live_link', None)
        if link is None or not link.active:
            return
        try:
            # The take's face rides on the same packet as its body, so Blender
            # gets both for the same frame.
            try:
                link.set_face(self.live_face_values(index))
            except (OSError, ValueError, KeyError):
                link.set_face(None)
            try:
                link.set_people(self.live_people(index))
            except (OSError, ValueError, KeyError) as error:
                link.set_people(None)
                self.log.append('Live link: other people: ' + str(error))
            fps = self.info['fps'] if self.info else 30
            frames = self.info['frames'] if self.info else None
            # A character on screen is streamed as itself: its own rest file,
            # palette and bone matrices, so Blender shows the same rig.
            take = self.character_live_take() if getattr(self, 'characters', None) else None
            if take is not None:
                char, path, frame, transform, matrix = take
                info = self.character_live_info(path)
                if info is not None:
                    link.publish_character(info['blend'], info['palette'], info['armature'],
                                           matrix, frame, fps, frames, char.get('name', ''),
                                           transform, info.get('height'))
                    return
            # The app's own base body: its SMPL surface skinned to the export
            # skeleton, so Blender shows what is on screen without any import.
            body_frame = self.live_body_frame(index)
            if body_frame is not None:
                body, body_index, body_transform = body_frame
                link.publish_character(body.rest_file(), body.names, 'RGBMoCap Body',
                                       body.matrices(body_index), body_index, body.fps,
                                       body.count, getattr(self.motion_view, 'actor_name', ''),
                                       body_transform, body.height, rest_format='npz')
                return
            joints = self.live_pose(index)
            if joints is None:
                return
            link.publish_joints(joints, index, fps, frames,
                                getattr(self.motion_view, 'actor_name', ''))
        except Exception as error:
            self.log.append('Live link: ' + str(error))

    def publish_live_character(self):
        """Send the character on screen first, so Blender imports it before poses."""
        link = getattr(self, 'live_link', None)
        if link is None or not link.active or not getattr(self, 'characters', None):
            return
        try:
            take = self.character_live_take()
            if take is None:
                # The take may not be playing or loaded yet; its rest file is
                # still the right thing to import first.
                char = next((c for c in self.characters if c.get('visible', True)), None)
                if char is None:
                    return
                takes = char.get('takes') or []
                folder = char.get('animation', '') + '/' + takes[-1]['folder'] if takes else ''
                take = (char, self.character_path(folder or char['asset']), 0,
                        takes[-1].get('transform') if takes else None, None)
            char, path, frame, transform, _matrix = take
            info = self.character_live_info(path)
            if info is None:
                return
            link.publish_character(info['blend'], info['palette'], info['armature'], None,
                                   frame, self.info['fps'] if self.info else 30,
                                   self.info['frames'] if self.info else None,
                                   char.get('name', ''), transform, info.get('height'))
        except Exception as error:
            self.log.append('Live link: ' + str(error))

    def toggle_live_link(self, enabled):
        if not enabled:
            link = getattr(self, 'live_link', None)
            if link is not None:
                link.stop()
            self.status.setText('Live link stopped.')
            return
        from .live_link import LiveLink, default_port
        link = getattr(self, 'live_link', None)
        if link is None:
            self.live_link = link = LiveLink(port=default_port())
        ok, message = link.start()
        self.log.append('Live link: ' + message)
        if not ok:
            self.status.setText(message)
            action = getattr(self, 'live_link_action', None)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
            return
        self.status.setText('Live link listening on %s. In Blender: RGB MoCap > Start Live Link.'
                            % link.status()['address'])
        # Send the character itself first, then its current pose: Blender
        # appends the app's own rest file before any bone transform arrives.
        self.publish_live_character()
        self.publish_live_frame(getattr(self.motion_view, 'index', 0) or 0)

    def cascadeur_poll_timer(self):
        """The timer that watches a connection attempt finishing.

        Connecting posts a script to Cascadeur and waits for its receiver to
        answer, which is seconds not milliseconds. Doing that from the menu
        would freeze the window, and updating it from the worker thread is not
        allowed, so the worker leaves its answer here and this picks it up.
        """
        timer = getattr(self, 'cascadeur_poll', None)
        if timer is None:
            self.cascadeur_poll = timer = QTimer(self)
            timer.setInterval(120)
            timer.timeout.connect(self.poll_cascadeur_link)
        return timer

    def send_base_character(self):
        """Import the T-pose base body into a new Cascadeur scene and check it."""
        from .cascadeur_link import CascadeurLink, base_character_path
        fbx = base_character_path(Path(__file__).resolve().parents[1])
        if not fbx.is_file():
            message = ('the base character is missing at %s; export it once with '
                       'tools/send-base-body-to-cascadeur.py' % fbx)
            self.log.append('Cascadeur: ' + message)
            self.status.setText('Cascadeur: ' + message)
            return
        self.status.setText('Cascadeur: sending the base character...')
        outcome = {}

        def work():
            try:
                outcome['value'] = CascadeurLink().send_character(fbx)
            except Exception as error:      # noqa: BLE001 - reported in the window
                outcome['value'] = (False, str(error), {})

        threading.Thread(target=work, name='rgbmocap-cascadeur-character',
                         daemon=True).start()

        def poll():
            if 'value' not in outcome:
                QTimer.singleShot(200, poll)
                return
            ok, message, checks = outcome['value']
            self.log.append('Cascadeur: ' + message)
            self.status.setText('Cascadeur: ' + message)
            if ok:
                self.log.append('Cascadeur: base character ready. Now check '
                                'File > Live Link to Cascadeur.')

        QTimer.singleShot(200, poll)

    def toggle_cascadeur_link(self, enabled):
        from .cascadeur_link import CascadeurLink
        link = getattr(self, 'cascadeur_link', None)
        if not enabled:
            self.cascadeur_poll_timer().stop()
            if link is not None:
                link.stop()
                try:
                    link.disconnect()
                except Exception as error:      # noqa: BLE001 - logged, not raised
                    self.log.append('Cascadeur: ' + str(error))
            self.status.setText('Cascadeur live link stopped.')
            return
        if link is None:
            self.cascadeur_link = link = CascadeurLink()
        self.status.setText('Connecting to Cascadeur...')
        self.cascadeur_outcome = None

        def work():
            try:
                self.cascadeur_outcome = link.connect()
            except Exception as error:          # noqa: BLE001 - shown, not raised
                self.cascadeur_outcome = (False, str(error))

        threading.Thread(target=work, name='rgbmocap-cascadeur-connect', daemon=True).start()
        self.cascadeur_poll_timer().start()

    def poll_cascadeur_link(self):
        outcome = getattr(self, 'cascadeur_outcome', None)
        if outcome is None:
            return
        action = getattr(self, 'cascadeur_link_action', None)
        if action is not None and not action.isChecked():
            # Stopped while the connection was still being made.
            self.cascadeur_outcome = None
            self.cascadeur_poll_timer().stop()
            return
        self.cascadeur_outcome = None
        self.cascadeur_poll_timer().stop()
        ok, message = outcome
        self.log.append('Cascadeur: ' + message)
        if not ok:
            self.cascadeur_link_failed(message)
            return
        frame = getattr(self.motion_view, 'index', 0) or 0
        found = self.live_body_frame(frame)
        if found is None:
            self.status.setText('Cascadeur linked. Play a reconstructed take to stream it.')
            return
        body, index, _transform = found
        ok, message = self.cascadeur_link.start(body, index)
        self.log.append('Cascadeur: ' + message)
        if not ok:
            self.cascadeur_link_failed(message)
            return
        self.status.setText(message)

    def cascadeur_link_failed(self, message):
        self.status.setText('Cascadeur: ' + message)
        action = getattr(self, 'cascadeur_link_action', None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)

    def publish_cascadeur_frame(self, index):
        link = getattr(self, 'cascadeur_link', None)
        if link is None or not link.active:
            return
        try:
            found = self.live_body_frame(index)
            if found is None:
                return
            body, body_index, _transform = found
            if body is not link.body:
                # A different take or person came on screen: rebind and rewind
                # rather than writing one take's frames into another's timeline.
                ok, message = link.start(body, body_index)
                self.log.append('Cascadeur: ' + message)
                if not ok:
                    self.cascadeur_link_failed(message)
                    return
            link.publish(body_index)
        except Exception as error:              # noqa: BLE001 - logged, not raised
            self.log.append('Cascadeur live: ' + str(error))

    def reset_play_clock(self, *_):
        self.play_anchor = self.slider.value()
        self.play_clock.start()
        self.presented_times.clear()
        if self.timer.isActive() and self.info:
            self.timer.start(max(1, math.ceil(1000 / (self.info['fps'] * self.play_speed.currentData()))))
            # The playhead or the speed moved: the sound restarts from there.
            self.start_playback_audio()

    def stop_playback(self):
        self.timer.stop()
        self.timer_resolution.release()
        self.source_audio.stop()
        self.set_playback_indicator(False)
        self.pear_playback_stopped()

    def set_playback_indicator(self, playing):
        label = "Pause" if playing else "Play"
        self.play_button.setIcon(self.pause_icon if playing else self.play_icon)
        self.play_button.setToolTip(tooltip_text(label + " (Space)"))
        self.play_button.setAccessibleName(label)

    def step_frames(self, amount):
        if self._pear_active and self.pear_using_webcams():return
        self.stop_playback()
        self.slider.setValue(max(0, min(self.slider.maximum(), self.slider.value() + amount)))

    def toggle_play(self):
        if self._pear_active and self.pear_using_webcams():return
        if self.timer.isActive():
            self.stop_playback()
        elif self.video_path and self.info and self.video_readable:
            if not self.slider.clip_in <= self.slider.value() < self.slider.clip_out:
                self.slider.setValue(self.slider.clip_in)
            self.timer_resolution.acquire()
            self.reset_play_clock()
            self.timer.start(max(1, math.ceil(1000 / (self.info['fps'] * self.play_speed.currentData()))))
            self.start_playback_audio()
            self.set_playback_indicator(True)

    def audio_source(self):
        """Where the preview's sound comes from, or None when it has none.

        A link preview already holds the stream address and the headers that
        address needs; a clip preview reads the same file the picture comes
        from. A live camera has no soundtrack to play.
        """
        session = getattr(self, 'camera_session', None)
        if session is not None:
            address = getattr(session, 'audio_address', None) or getattr(session, 'address', None)
            if not address:
                return None
            headers = getattr(session, 'audio_headers', None) or getattr(session, 'headers', {})
            return dict(address=address, headers=dict(headers or {}))
        path = getattr(self, 'video_path', None)
        return path if path is not None else None

    def start_playback_audio(self):
        """Sound follows the playhead and the preview speed."""
        source = self.audio_source()
        fps = float((self.info or {}).get('fps') or 0.)
        if source is None or fps <= 0:
            return
        self.source_audio.play(source, start=max(0., self.slider.value() / fps),
                               speed=self.play_speed.currentData() or 1.)

    def set_preview_volume(self, percent):
        """Preview volume, 0-100: the source keeps playing, only quieter."""
        self.preview_volume = max(0, min(100, int(percent))) / 100.
        self.source_audio.set_volume(self.preview_volume)
        mute = getattr(self, 'mute_button', None)
        if mute is not None:
            mute.blockSignals(True)
            mute.setChecked(self.preview_volume <= 0)
            mute.blockSignals(False)

    def toggle_preview_mute(self, muted):
        """Mute without losing the level the user chose."""
        if muted:
            self._volume_before_mute = self.volume.value() or 100
            self.volume.setValue(0)
        else:
            self.volume.setValue(getattr(self, '_volume_before_mute', 100) or 100)

    def start_link_audio(self, session):
        """A link preview plays its clip again, so its sound repeats as well."""
        address = getattr(session, 'audio_address', None) or getattr(session, 'address', None)
        if not address:
            return
        headers = getattr(session, 'audio_headers', None)
        if not headers:
            headers = getattr(session, 'headers', {})
        self.source_audio.play(dict(address=address, headers=dict(headers or {})), loop=True)

    def next_frame(self):
        if not self.timer.isActive() or self.info is None or not self.video_readable:
            return
        elapsed = self.play_clock.nsecsElapsed() / 1e9
        rate = self.info["fps"] * self.play_speed.currentData()
        target = self.slider.clip_in + playback_frame(self.play_anchor - self.slider.clip_in,
                                elapsed, self.info["fps"], self.slider.clip_out - self.slider.clip_in,
                                self.play_speed.currentData())
        wrapped = target < self.slider.value()
        self.playback_tick = True
        try:
            self.slider.setValue(target)
        finally:
            self.playback_tick = False
        if wrapped:
            # The loop came back round to the clip start. The sound played out
            # once and stopped there, so the second pass was silent; restarting
            # the clock here also restarts the sound from the new playhead.
            self.reset_play_clock()
            return
        next_deadline = (math.floor(elapsed * rate + 1e-8) + 1) / rate
        remaining = next_deadline - self.play_clock.nsecsElapsed() / 1e9
        self.timer.start(max(1, math.ceil(remaining * 1000)))

    def shutdown_preview(self):
        if hasattr(self,'camera_tools'):self.camera_tools.close()
        self.shutdown_mamma()
        self.stop_playback()
        if hasattr(self,'character_loader'):
            self.character_loader.close();self.character_assets.clear()
        self.scene_loader.stop()
        self.scene_loader.wait()
        self.clear_people_scene()
        self.clear_preview()
        self.data_loader.stop()
        self.data_loader.wait()
        self.contact_writer.stop()
        self.contact_writer.wait()
        self.video_view.shutdown()
        self.motion_view.cleanup_gl()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange and self.isMinimized() and hasattr(self, "timer"):
            self.stop_playback()
        super().changeEvent(event)

    def hideEvent(self, event):
        if hasattr(self, "timer"):
            self.stop_playback()
        super().hideEvent(event)

    def input_jobs(self, stages, result=None):
        if MOTION_TOOLS[self.model_choice.currentData()].input_setup_only:
            self.status.setText('MAMMA camera input setup is available; reconstruction is not integrated yet.')
            return
        if self.video_path is None:
            self.choose_video()
            return
        if self.info is None or not self.video_readable:
            self.status.setText("Wait for the video preview to finish opening.")
            return
        if 'people' not in stages and self._single_actor and not self._single_actor['checked']:
            self.status.setText('Check Person 1 in the People list to reconstruct it.')
            return
        if 'people' not in stages and self.input_person_jobs(stages,result):
            return
        unavailable = [model for stage in stages for model in ([] if self.model_choice.currentData()=='flowhmr' and stage=='flow_camera' and self.flow_sampling.currentData()==-1 else missing('flow_export' if self.model_choice.currentData()=='flowhmr' and stage in ('correct','export') else stage))]
        if self.model_choice.currentData()=='flowhmr' and self.flow_hamer.isChecked():
            from .flow_hamer import MODEL_KEYS
            from .models import BY_KEY
            unavailable += [BY_KEY[k] for k in MODEL_KEYS if not ready(BY_KEY[k])]
        if unavailable:
            names = ", ".join(dict.fromkeys(model.name for model in unavailable))
            self.status.setText(f"Prepare these models first: {names}")
            self.open_model_manager()
            return
        first, end = self.processing_range()
        prepare_args = ["--video", str(self.video_path), "--start", str(first), "--end", str(end)]
        if self.workspace_root is not None:
            prepare_args += ['--project',str(self.workspace_root)]
        elif self.project is not None:
            from .project_storage import storage_root
            if storage_root(self.project) is not None:
                prepare_args += ['--project', str(self.project)]
        if result or self.model_choice.currentData()!='gvhmr':
            prepare_args+=['--motion-tool',self.model_choice.currentData()]
        if result:
            prepare_args += ["--result", result]
            if result.startswith('corrected_') and not self.auto_floor.isChecked():
                prepare_args += ['--floor',str(self.floor_height.value())]
            if result.startswith('corrected_'):
                prepare_args += ['--correction-preset',self.correction_preset.currentData(),
                                 '--correction-mode',self.correction_mode.currentData(),
                                 '--foot-lock',str(int(self.foot_lock.isChecked())),'--hand-lock',str(int(self.hand_lock.isChecked())),
                                 '--occlusion-fill',str(int(self.occlusion_fill.isChecked())),
                                 '--pelvis-offset',str(self.pelvis_offset.value())]
        if self.people_root is None and 'correct' in stages and RECONSTRUCTION_STAGES & set(stages):
            self._single_result_kind = result or self.active_result
            self._ground_after_single_batch = True
        self.start_jobs([("prepare", prepare_args)]
                        + [(stage, ["--result", result] if stage in {"export", "flow_export", "footmr", "handflow", "correct"} and result else None) for stage in stages])

    def ground_lone_actor_after_reconstruction(self):
        """Ensure a lone actor's feet land on its flat floor after reconstruction.

        The corrected pass normally does this, but a cancelled, skipped or
        settings-changed stage would leave the preview floating until a manual
        correction. Only a missing/stale corrected result is recomputed.
        """
        if self.busy or self.project is None or self.people_root is not None:return
        if not self.motion_correction.isChecked():return
        kind=getattr(self,'_single_result_kind',None) or 'corrected_initial'
        if not kind.startswith('corrected_'):kind='corrected_'+base_result(kind)
        from .motion_tools import result_sources
        bases=result_sources(base_result(kind))
        if not all((self.project/RESULTS[base][0]).is_file() and result_is_current(self.project,base) for base in bases):return
        if result_is_current(self.project,kind):return
        self.status.setText('Grounding feet on the floor...')
        self.start_jobs([(stage,['--project',str(self.project),'--result',kind])
                         for stage in ('correct','export')],refresh_characters=True)

    def run_analysis(self):
        self.input_jobs(["track", "pva"])

    def run_full(self):
        tool = MOTION_TOOLS[self.model_choice.currentData()]
        if tool.key=='mamma':return self.run_mamma()
        if tool.key=='prompthmr':return self.run_prompthmr()
        if tool.key=='gemx':return self.run_gemx()
        if tool.key=='exmr' and self.people_root is not None:
            from .storage import atomic_json
            path=self.people_root/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['exmr_options']={'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'}
            atomic_json(path,state)
        if tool.key=='flowhmr' and self.people_root is not None:
            from .storage import atomic_json
            path=self.people_root/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['flow_options']={**state.get('flow_options',{}),**self.selected_flow_options()}
            atomic_json(path,state)
        refine = bool(tool.refinement) and self.htd_refine.isChecked()
        foot = tool.key=='gvhmr' and self.footmr_refine.isChecked()
        correction=self.motion_correction.isChecked()
        if correction:
            # The automatic pass grounds a lone actor on one flat floor; Multi
            # Level stays available for a manual Apply Correction.
            if self.people_root is None:self.use_flat_ground()
            self.save_correction_options()
        hand=tool.key=='gvhmr' and self.handflow_refine.isChecked()
        self.input_jobs(tool.stages(refine, foot,correction,handflow=hand), result=result_kind(refine, foot,correction,handflow=hand))

    def use_flat_ground(self):
        """Select the shared flat floor without firing change handlers."""
        if self.correction_mode.currentData()=='flat':return
        self.correction_mode.blockSignals(True)
        try:self.correction_mode.setCurrentIndex(self.correction_mode.findData('flat'))
        finally:self.correction_mode.blockSignals(False)
        self.update_correction_description()

    def save_correction_options(self):
        if self.project is not None:
            if self.people_root is not None:
                from .scene_floor import save_config
                save_config(self.people_root,self.selected_correction_options())
            from .storage import atomic_json
            path=self.project/'project.json';state=json.loads(path.read_text(encoding='utf-8'))
            state['correction']=self.selected_correction_options()
            atomic_json(path,state)

    def selected_correction_options(self):
        from .correction_settings import correction_options
        return correction_options({'auto_floor':self.auto_floor.isChecked(),'floor':self.floor_height.value(),
                                   'mode':self.correction_mode.currentData(),'foot_lock':self.foot_lock.isChecked(),'hand_lock':self.hand_lock.isChecked(),
                                   'occlusion_fill':self.occlusion_fill.isChecked(),
                                   'floor_board':self.floor_board.isChecked(),
                                   'preset':self.correction_preset.currentData(),'pelvis_offset':self.pelvis_offset.value()})

    def update_correction_description(self):
        descriptions = {
            'weak': 'Gentler correction, with some sliding allowed to preserve more of the original movement.',
            'normal': 'Balanced contact correction. Holds stable contacts while keeping transitions smooth.',
            'strong': 'Firmer holds, including less certain contacts. May resist intentional slides; review the result.',
        }
        self.preset_description.setText(descriptions[self.correction_preset.currentData()])
        if self.correction_mode.currentData()=='multi_level':
            self.floor_description.setText('Multi Level uses the inferred height of each foot contact. This floor value is unused; switch to Flat Ground to edit it.')
        else:
            self.floor_description.setText('Higher values raise the correction floor. Use the Correction Floor checkbox to preview it; Apply Correction updates the motion.')

    def restore_correction_options(self, options):
        from .correction_settings import correction_options
        options=correction_options(options)
        controls=(self.correction_mode,self.correction_preset,self.auto_floor,self.floor_height,self.pelvis_offset,self.foot_lock,self.hand_lock,self.occlusion_fill,self.floor_board)
        for control in controls:control.blockSignals(True)
        try:
            # An empty options dict (a fresh load) keeps the current state:
            # the .get defaults are each control's own value.
            def _opt(key, widget):
                return options[key] if key in options else (widget.isChecked() if hasattr(widget, 'isChecked') else widget.value())
            self.correction_preset.setCurrentIndex(self.correction_preset.findData(_opt('preset', self.correction_preset)))
            self.correction_mode.setCurrentIndex(self.correction_mode.findData(_opt('mode', self.correction_mode)))
            self.foot_lock.setChecked(_opt('foot_lock', self.foot_lock));self.hand_lock.setChecked(_opt('hand_lock', self.hand_lock))
            self.occlusion_fill.setChecked(_opt('occlusion_fill', self.occlusion_fill))
            self.floor_board.setChecked(bool(options.get('floor_board')))
            self.auto_floor.setChecked(_opt('auto_floor', self.auto_floor));self.floor_height.setValue(_opt('floor', self.floor_height))
            self.pelvis_offset.setValue(_opt('pelvis_offset', self.pelvis_offset))
        finally:
            for control in controls:control.blockSignals(False)
        self.update_correction_description()

    def correction_settings_changed(self, *_):
        if self.sender() is self.floor_height and self.auto_floor.isChecked():
            # A person-entered height replaces the automatic estimate.
            self.auto_floor.blockSignals(True);self.auto_floor.setChecked(False);self.auto_floor.blockSignals(False)
        self.update_correction_description()
        self.save_correction_options()
        if self.sender() is self.correction_preset:
            self.request_people_scene(force=True)
            if self.preview_data is not None:self.apply_preview()
            self.update_readiness()
            return
        # Keep the last committed pose visible while editing the reference
        # plane. Only Apply Correction changes motion; stale export stays blocked.
        if self.preview_data is not None:
            self.correction_note.setText('Correction settings changed. Preview shows the last applied motion.\nApply Correction to update the pelvis, contacts and FBX.')
            self.correction_note.show()
        self.update_readiness()

    def apply_correction(self):
        if self.project is None or self.busy or self.contact_saving:return
        info=self.contact_tracks.get(base_result(self.active_result))
        if info and info.get('unsaved'):
            self.status.setText('Contact-lock timing was not saved. Edit or reset the tracks to retry saving.');return
        self.save_correction_options()
        kind=self.selected_result(True)
        if self.model_choice.currentData()=='gemx':
            self.start_jobs([('gemx_correct',['--project',str(self.project),'--result',kind])], refresh_characters=True)
            return
        self.set_active_result(kind)
        if self.people_root is not None:
            self.apply_scene_correction(kind);return
        self.start_jobs([('correct',['--result',kind]),('export',['--result',kind])], refresh_characters=True)

    def update_foot_contacts(self):
        base=base_result(self.active_result)
        info=self.contact_tracks.get(base)
        if info and self.info and (info['context']['frames']!=self.info['frames'] or info['context']['fps']!=self.info['fps']):info=None
        self.slider.set_contacts(info['ranges'] if info else None,info['automatic'] if info else None,
            editable=not self.busy and not self.contact_saving and self.motion_correction.isChecked(),
            identity=(str(self.project),base),token=id(info))
        from .foot_contacts import contact_levels
        self.motion_view.foot_lock_levels=lambda frame:contact_levels(info,frame)

    def foot_contacts_changed(self,ranges):
        self.save_foot_contacts(ranges)

    def reset_foot_contacts(self):
        self.save_foot_contacts(None)

    def save_foot_contacts(self,ranges):
        if self.project is None or self.busy or self.contact_saving:return
        base=base_result(self.active_result);info=self.contact_tracks.get(base)
        if info is None:return
        self.stop_playback();self.contact_saving=True
        try:self.contact_save_token=self.contact_writer.request(self.project,base,ranges,info['context'])
        except Exception as error:
            self.contact_saving=False;self.status.setText('Cannot save contact-lock timing: '+str(error));return
        # Keep immutable metadata while the writer commits off the GUI thread.
        self.contact_tracks[base]={**info,'ranges':ranges if ranges is not None else info['automatic'],'manual':ranges is not None,'unsaved':True}
        self.status.setText('Saving contact-lock timing...')
        self.update_readiness()

    def foot_contacts_saved(self,sequence,record):
        if self.project is None or str(self.project.resolve())!=record['project']:return
        info=self.contact_tracks.get(record['base'])
        if info is None:return
        self.contact_tracks[record['base']]={**info,'revision':record['state']['revision'],'unsaved':False}
        for key,scene in self.scene_cache.items():
            if base_result(key[1])==record['base']:
                take=scene.by_project.get(self.project)
                if take:
                    take.set_contacts(info['ranges'])
                    take.contact_revision=record['state']['revision']
        if sequence==self.contact_save_token:
            if self.preview_data is not None:self.apply_preview()
            self.status.setText('Contact-lock timing saved. Apply Correction to update the motion and FBX.')

    def foot_contacts_failed(self,sequence,message):
        if sequence==self.contact_save_token:self.status.setText('Cannot save contact-lock timing: '+message)

    def foot_contacts_settled(self,sequence):
        if sequence==self.contact_save_token:
            self.contact_saving=False;self.update_readiness()

    def prepare_mesh(self):
        if self.project is not None and not self.busy:
            self.prepare_target_preview()

    def resource_safety_changed(self,value):
        from .resource_safety import ENV,save,message
        try:save(value)
        except (OSError,ValueError) as error:
            self.resource_safety.blockSignals(True);self.resource_safety.setChecked(not value);self.resource_safety.blockSignals(False)
            self.log.append('Cannot save Resource Safety: '+str(error));return
        os.environ[ENV]='1' if value else '0'
        self.log.append(message(value))

    def start_jobs(self, jobs, refresh_characters=False):
        if self.busy:
            return
        # A reconstruction needs the whole GPU; the waiting live worker gives
        # it back and is preloaded again when a live source next opens.
        self.stop_preloaded_worker()
        jobs=list(jobs)
        from .resource_safety import message
        self._job_resource_safety=self.resource_safety.isChecked()
        self.log.append(message(self._job_resource_safety))
        self.character_refresh_queue=[]
        if refresh_characters:
            try:
                self.prepare_character_refresh(jobs)
            except (OSError, ValueError, KeyError) as exc:
                self.character_refresh_queue=[]
                self.status.setText(f'Cannot update linked characters: {exc}')
                self.update_readiness()
                return
        self._batch_started_at = time.monotonic()
        self._batch_has_motion = False
        self.task_state.reset(self._batch_started_at)
        self._task_manifests={}
        self.jobs, self.busy, self.cancelled, self.failed = list(jobs), True, False, False
        self._task_preparing=any(stage=='prepare' for stage,_ in self.jobs)
        self.update_readiness()
        self.next_job()

    def task_descriptor(self,stage,arguments):
        """Small metadata once per transition; no filesystem access on ticks."""
        arguments=arguments or []
        project=(Path(arguments[arguments.index('--project')+1]) if '--project' in arguments else
                 None if getattr(self,'_task_preparing',False) else self.project)
        tool=self.model_choice.currentData();state={};estimate=None
        if project is not None:
            project=Path(project)
            if project not in self._task_manifests:
                try:
                    file=project/'project.json'
                    state=json.loads(file.read_text(encoding='utf-8')) if file.stat().st_size<=1024**2 else {}
                except (OSError,ValueError):state={}
                if len(self._task_manifests)>=32:self._task_manifests.pop(next(iter(self._task_manifests)))
                self._task_manifests[project]=state
            state=self._task_manifests[project];tool=state.get('pipeline',{}).get('model',tool)
            estimate=state.get('stages',{}).get(stage,{}).get('elapsed')
            if stage.startswith('exmr'):
                try:
                    file=project/(stage+'_receipt.json')
                    if file.stat().st_size<=65536:estimate=json.loads(file.read_text(encoding='utf-8')).get('elapsed')
                except (OSError,ValueError):pass
        frames=state.get('frames',self.info.get('frames',0) if self.info else 0)
        if '--first-frame' in arguments and '--end-frame' in arguments:
            frames=int(arguments[arguments.index('--end-frame')+1])-int(arguments[arguments.index('--first-frame')+1])
        model=arguments[arguments.index('--model')+1] if '--model' in arguments else ''
        result=arguments[arguments.index('--result')+1] if '--result' in arguments else self.active_result
        key=(stage,tool,frames,model,result,self.compute_mode.currentData(),self.fast_sam.isChecked(),self.flow_batch4.isChecked(),self.flow_gpu_crop.isChecked(),self.flow_hand_stride4.isChecked(),self.flow_hamer.isChecked(),self.flow_sampling.currentData(),self.exmr_camera.currentData())
        if not isinstance(estimate,(int,float)) or not math.isfinite(estimate) or not 1<=estimate<=7*86400:estimate=None
        if stage in ('flow_batch','exmr_batch','gvhmr_batch'):
            from .queue_groups import argument
            items=json.loads(argument(arguments,'--batch-items'))
            key=(*key,argument(arguments,'--batch-phase'),tuple((item['project'],item['result']) for item in items),len(items))
            estimate=None
        estimate=self.task_state.history.get(key,estimate)
        return key,estimate

    def update_task_progress(self):
        if not self.busy or self.failed or self.cancelled or not self.task_state.active:return
        text,percent=self.task_state.status(time.monotonic(),self._queued_task_estimates)
        self.task_info.setText(text)
        # A real 0/N observation is loading, not a stationary empty bar.
        if percent is None:self.progress.setRange(0,0)
        else:
            total=self.task_state.total;current=self.task_state.current
            if total<=2147483647 and total.is_integer():
                self.progress.setRange(0,int(total));self.progress.setValue(int(current))
            else:
                self.progress.setRange(0,1000);self.progress.setValue(round(1000*current/total))

    def cleanup_batch_request(self):
        from .queue_groups import remove_request
        path=getattr(self,'_batch_request',None);self._batch_request=None
        try:remove_request(path)
        except OSError as error:self.log.append('Could not remove worker request: '+str(error))

    def next_job(self):
        self.cleanup_batch_request()
        self.guard_timer.stop()
        self.processing_guard = None
        if not self.jobs or self.failed or self.cancelled:
            self.task_timer.stop()
            elapsed = (max(0., time.monotonic() - self._batch_started_at)
                       if self._batch_started_at is not None else None)
            self._batch_started_at = None
            self.busy = False
            self.finish_single_tracking()
            self.task_state.active=False
            from .task_progress import duration
            self.task_info.setText(('Stopped' if self.failed or self.cancelled else 'Done')+
                                   (f' · Elapsed {duration(elapsed)}' if elapsed is not None else ''))
            self.progress.setRange(0, 100)
            self.progress.setValue(0 if self.failed or self.cancelled else 100)
            if self.current_stage in ('save_fbx','copy_bvh','save_blend','save_blend_scene'):
                # Saving a derived file does not replace scene/preview data.
                # Reloading the people scene here moves a following camera.
                if not self.failed and not self.cancelled:
                    paths=getattr(self,'_motion_export_targets',[])
                    self.status.setText('Motion export complete.'+(' '+paths[0] if len(paths)==1 else f' {len(paths)} files.' if paths else ''))
                    for path in paths:self.log.append('Exported: '+path)
                self.update_readiness()
                return
            if self.current_stage=='save_project':
                self.finish_project_save()
                return
            person_batch=self._people_batch
            completion_message=self.status.text()
            self.finish_people_jobs()
            # A finished reconstruction of a lone actor gets one grounding check.
            if getattr(self,'_ground_after_single_batch',False):
                self._ground_after_single_batch=False
                if not self.failed and not self.cancelled:
                    QTimer.singleShot(0,self.ground_lone_actor_after_reconstruction)
            if self.failed or self.cancelled:self.status.setText(completion_message)
            if not self.failed and not self.cancelled:
                if self.current_stage in ("export",'flow_export') and self.project is not None and not person_batch:
                    metadata = json.loads((self.project / "export.json").read_text(encoding="utf-8"))
                    self.set_active_result(metadata.get("result_kind", "refined"))
                if self.current_stage == "diagnose" and self.hardware_summary:
                    self.status.setText(self.hardware_summary)
                elif self.current_stage == "pva":
                    self.status.setText("2D analysis complete. View the estimated joints over the video.")
                elif self.current_stage == "download":
                    self.status.setText("Downloads complete. Check Model Manager for any required imports.")
                elif self.current_stage == "import_model":
                    self.status.setText("Model setup complete.")
                elif self.current_stage not in ('people','person'):
                    self.status.setText("Complete. Play the result or open the project folder.")
                if self._batch_has_motion and elapsed is not None:
                    seconds = round(elapsed)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    duration = (f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else
                                f'{minutes}m {seconds:02d}s' if minutes else f'{elapsed:.1f} s')
                    self.status.setText(f'Complete. Total time: {duration}. Play the result or open the project folder.')
                    self.log.append(f'Total motion processing time: {elapsed:.2f} s')
            self.refresh_results()
            self.update_readiness()
            if getattr(self, '_auto_play', False):
                self._auto_play = False
                if not self.failed and not self.cancelled:
                    self.slider.setValue(self.slider.clip_in)
                    self.toggle_play()
                    resume = getattr(self, '_live_resume_info', None)
                    if resume is not None:
                        self._live_resume_info = None
                        device, cw, ch, fps, preview_width, rotation = resume
                        from . import camera_capture
                        session = camera_capture.Session(
                            camera_capture.default_path(ROOT), device, cw, ch, fps,
                            preview_width=preview_width, rotation=rotation)
                        session.start_preview()
                        self.start_live_camera(session)
            if self.failed or self.cancelled:
                self.character_refresh_queue=[]
            else:
                self.next_character_refresh()
            return
        from .queue_groups import join_single_pipeline
        self.jobs = join_single_pipeline(self.jobs, self.project)
        stage, arguments = self.jobs.pop(0)
        if stage in {'gvhmr', 'refine', 'footmr', 'handflow', 'correct',
                     'flow_camera', 'flow_features', 'flowhmr',
                     'exmr_preprocess', 'exmr_features', 'exmr'}:
            self._batch_has_motion = True
        if stage.startswith(('mamma_','prompthmr_')) or stage in ('flow_batch','exmr_batch','gvhmr_batch','gemx_reconstruct'):self._batch_has_motion=True
        self.current_stage = stage
        self._batch_inner_stage=stage
        if arguments is None:
            arguments = ["--project", str(self.project)]
        elif stage in {"export", "footmr", "handflow", "correct", 'flow_export'} and "--project" not in arguments:
            arguments = ["--project", str(self.project), *arguments]
        self.job_project = Path(arguments[arguments.index('--project')+1]) if '--project' in arguments else None
        if stage=='track':self.begin_single_tracking(self.job_project)
        if stage in {"download", "import_model"} and "--model" in arguments:
            from .models import BY_KEY
            model_key = arguments[arguments.index("--model") + 1]
            self.status.setText(("Downloading " if stage == "download" else "Importing ") + BY_KEY[model_key].name + "...")
        else:
            self.status.setText(STAGES[stage] + "...")
        key,estimate=self.task_descriptor(stage,arguments)
        self.task_state.begin(time.monotonic(),key,self.status.text(),estimate)
        actor=next((part for part in self.job_project.parts if part.startswith('person_')),'') if self.job_project else ''
        self._task_actor=actor.replace('_',' ').title().replace('000','')
        self._queued_task_estimates=[self.task_descriptor(s,a) for s,a in self.jobs]
        self.task_timer.start();self.update_task_progress()
        self.progress.setRange(0, 0)
        self.buffer = ""
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_buffer = ""
        if stage in ('flow_batch','exmr_batch','gvhmr_batch'):
            from .queue_groups import write_request
            try:
                self._batch_request=write_request(stage,arguments)
            except (OSError,ValueError) as error:
                self.failed=True;self.log.append(str(error));self.status.setText('Cannot prepare people processing: '+str(error))
                self.next_job();return
            arguments=['--request',str(self._batch_request),'--project',str(self.job_project)]
        if stage=='gemx_reconstruct':
            self._batch_request=Path(arguments[arguments.index('--request')+1])
        try:
            command = worker_command(stage, *arguments)
        except (OSError, ValueError, RuntimeError) as error:
            self.failed=True
            self.log.append(str(error))
            self.status.setText(str(error))
            self.next_job()
            return
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("RGB_MOCAP_COMPUTE_MODE", self.compute_mode.currentData())
        environment.insert('RGB_MOCAP_RESOURCE_SAFETY','1' if self._job_resource_safety else '0')
        environment.insert('RGB_MOCAP_FLOW_INTERVAL',str(self.flow_sampling.currentData()))
        environment.insert('RGB_MOCAP_FLOW_FAST_SAM','1' if self.fast_sam.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_BATCH4','1' if self.flow_batch4.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_GPU_CROP','1' if self.flow_gpu_crop.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_HAND_STRIDE4','1' if self.flow_hand_stride4.isChecked() else '0')
        environment.insert('RGB_MOCAP_FLOW_HAMER','1' if self.flow_hamer.isChecked() else '0')
        environment.insert('RGB_MOCAP_EXMR_OPTIONS',json.dumps({'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'}))
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.started.connect(self.start_processing_guard)
        self.process.start(command[0], command[1:])

    def start_processing_guard(self):
        from .processing_guard import ProcessingGuard
        try:
            self.processing_guard = ProcessingGuard(self.process.processId(),resource_safety=self._job_resource_safety)
            self.guard_timer.start()
        except Exception as error:
            self.log.append('Resource monitor unavailable: ' + str(error))

    def check_processing_resources(self):
        if not self.processing_guard or not self.process:return
        message = self.processing_guard.check()
        if message:
            self.failed = True
            self.cancel_job()
            self.status.setText(message)
            self.log.append(message)
            self.guard_timer.stop()
            if getattr(self,'job_project',None) is not None:
                from .storage import atomic_json
                path=self.job_project/'project.json'
                try:
                    state=json.loads(path.read_text(encoding='utf-8'))
                    state.setdefault('stages',{})[self._batch_inner_stage]={'status':'failed','error':message}
                    atomic_json(path,state)
                except (OSError,ValueError) as error:self.log.append('Could not save stopped stage: '+str(error))

    def read_stdout(self, final=False):
        if self.processing_guard:self.processing_guard.output()
        self.buffer += self._stdout_decoder.decode(bytes(self.process.readAllStandardOutput()), final=final)
        if final and self.buffer and not self.buffer.endswith("\n"):self.buffer += "\n"
        if len(self.buffer) > 65536 and "\n" not in self.buffer:
            self.buffer = self.buffer[-65536:]
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.startswith("@@MOCAP@@"):
                try:
                    self.handle_event(json.loads(line[len("@@MOCAP@@"):]))
                except (ValueError, KeyError) as error:
                    self.log.append(str(error))
            elif line.strip():
                self.log.append(line.rstrip())

    def read_stderr(self, final=False):
        if self.processing_guard:self.processing_guard.output()
        text = self._stderr_decoder.decode(bytes(self.process.readAllStandardError()), final=final)
        self._stderr_buffer = (self._stderr_buffer + text.replace("\r", "\n"))[-65536:]
        lines = self._stderr_buffer.split("\n");self._stderr_buffer = lines.pop()
        if final and self._stderr_buffer:lines.append(self._stderr_buffer);self._stderr_buffer = ""
        for line in lines:
            if line.strip():self.log.append(line[-8192:])

    def handle_event(self, event):
        kind = event["type"]
        self.single_tracking_event(event)
        if self.people_event(event):
            self._queued_task_estimates=[self.task_descriptor(s,a) for s,a in self.jobs]
            self.update_task_progress();return
        if kind=='batch_item_start':
            self.job_project=Path(event['project']);self._batch_inner_stage=event['stage']
            self._task_actor=f"Actor/take {event['index']}/{event['count']}"
            self.task_state.update(dict(current=0,total=0,phase=f"batch:{event['phase']}:{event['index']}",message='Loading actor data'),time.monotonic())
            self.status.setText(self._task_actor+' · '+STAGES.get(event['stage'],event['stage']))
            self.update_task_progress()
        elif kind == 'gemx_corrected':
            if self.project == Path(event['project']):
                self.set_active_result('corrected_initial')
        elif kind in ('mamma_ready','prompthmr_ready','gemx_ready'):
            self.workspace_root=Path(event['root'])
            self.load_project(Path(event['project']))
            self.set_active_result(event['result'])
        elif kind == "prepared":
            self._task_preparing=False
            self._task_manifests={}
            self.project = Path(event["project"])
            from .project_storage import storage_root
            saved_root=storage_root(self.project)
            from .project_workspace import is_workspace
            if saved_root is not None and not is_workspace(saved_root):
                from .storage import atomic_json
                atomic_json(saved_root/'workspace_state.json',dict(version=1,name=saved_root.name,
                    active=self.project.resolve().relative_to(saved_root).as_posix(),frame=0,trim=None))
            self.load_project(self.project)
            self.trim_live_snapshot_to_person()
        elif kind == 'project_saved':
            self._project_saved_root=Path(event['root'])
        elif kind == 'save_cancelled':
            self.cancelled=True
            self.status.setText(event['message'])
        elif kind in ('resource_mode', 'info'):
            self.log.append(event['message'])
        elif kind == "progress":
            message=event.get('message','Processing...')
            self.log.status_terminal(message)
            self.status.setText((self._task_actor+' · ' if getattr(self,'_task_actor','') else '')+message)
            self.task_state.update(event,time.monotonic())
            self.update_task_progress()
        elif kind == "diagnostics":
            data = event["result"]
            self.log.append(json.dumps(data, ensure_ascii=False, indent=2))
            support = data.get('support') or {}
            verdict = {'unsupported': ' · GPU not supported, see Console',
                       'warning': ' · GPU limited, see Console'}.get(support.get('status'), '')
            self.hardware_summary = f"{data.get('gpu', 'GPU unavailable')} · RAM {data.get('ram_gib', '?')}GB · CUDA check {'Passed' if data.get('cuda_matmul_verified') else 'Unavailable'}" + verdict
            self.status.setText(self.hardware_summary)
        elif kind in {"failed", "blocked"}:
            self.failed = True
            self.task_timer.stop();self.task_info.setText('Failed')
            self.status.setText("Processing failed. " + event.get("message", "See the console for details."))
            self.log.append(self.status.text())
        elif kind == "complete":
            if not event.get('batch_inner'):self.task_state.cached|=bool(event.get('cached',False))
            self.log.append(f"{STAGES.get(event['stage'])}: {event.get('elapsed', 0):.2f} s · {event.get('output')}")
        self.log.ensureCursorVisible()

    def process_finished(self, code, _):
        self.read_stdout(final=True)
        self.read_stderr(final=True)
        if code != 0 and not self.cancelled:
            self.failed = True
            if "failed" not in self.status.text():
                self.status.setText("Processing stopped. See the console for details.")
        self.process.deleteLater()
        self.process = None
        self.task_state.finish(time.monotonic(),not self.failed and not self.cancelled and code==0)
        self.cleanup_fbx_export()
        self.next_job()

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.character_refresh_queue=[]
            self.task_timer.stop();self.task_state.active=False;self.task_info.setText('Failed to start')
            self.guard_timer.stop()
            self.processing_guard = None
            if self.process is not None:
                self.process.deleteLater()
                self.process = None
            self.jobs.clear()
            self.cleanup_fbx_export()
            self.cleanup_batch_request()
            self.cleanup_project_save()
            self.failed = True
            self.busy = False
            self.finish_single_tracking()
            self.finish_people_jobs()
            self.status.setText("Cannot start the processing engine. Check the app installation.")
            self.update_readiness()

    def cancel_job(self):
        self.pear_leave()
        self.pear_stop_scan()
        if self.character_process is not None:
            self.cancel_character();return
        if self.current_stage=='save_project' and getattr(self,'_project_save_request',None):
            self._project_save_request.with_suffix('.cancel').touch()
            self.task_info.setText('Stopping...')
            self.status.setText('Cancelling project save...')
            self.cancel_button.setEnabled(False)
            return
        from shiboken6 import isValid
        pending=getattr(self,'cancel_process',None)
        if pending is not None and isValid(pending) and pending.state()!=QProcess.NotRunning:
            return
        self.task_timer.stop();self.task_state.active=False;self.task_info.setText('Stopping...')
        self.guard_timer.stop()
        self.cancelled = True
        self.jobs.clear()
        worker = self.process
        if worker and isValid(worker) and worker.processId():
            if os.name == "nt":
                # Stop FFmpeg/other descendants before losing their parent PID.
                helper = QProcess(self)
                self.cancel_process = helper
                def fallback(error):
                    if error == QProcess.FailedToStart and self.process is worker and isValid(worker):
                        worker.kill()
                    if error == QProcess.FailedToStart:
                        if self.cancel_process is helper:self.cancel_process=None
                        if isValid(helper):helper.deleteLater()
                helper.errorOccurred.connect(fallback)
                def cancelled(code, _):
                    if code and self.process is worker and isValid(worker):
                        worker.kill()
                    if self.cancel_process is helper:self.cancel_process=None
                    if isValid(helper):helper.deleteLater()
                helper.finished.connect(cancelled)
                helper.start("taskkill.exe", ["/PID", str(worker.processId()), "/T", "/F"])
            else:
                worker.kill()
        self.status.setText("Cancelled. Completed stages will be reused next time.")

    def trim_live_snapshot_to_person(self):
        """A live snapshot holds the waiting time before the person walked in,
        and a body model invents motion for every frame it is given: clip the
        working range to where the tracker actually saw the person, so the
        reconstruction covers only real footage."""
        snapshot = getattr(self, '_live_snapshot', None)
        self._live_snapshot = None
        if snapshot is None or self.project is None:
            return
        import numpy as _np
        state = None
        try:
            from .people_store import load_people, root_for
            manifest = json.loads((self.project / 'project.json').read_text(encoding='utf-8'))
            root = root_for(self.project, manifest)
            state = load_people(root)
        except (OSError, ValueError, KeyError):
            return
        if not state or not state.get('people'):
            return
        person = next((p for p in state['people'] if p['id'] == state.get('active')), state['people'][0])
        track = root / person['preview'] / 'tracking.npz'
        if not track.is_file():
            return
        try:
            with _np.load(track, allow_pickle=False) as data:
                observed = data['observed']
        except (OSError, ValueError, KeyError):
            return
        if not observed.any():
            return
        first = int(observed.argmax())
        end = min(len(observed), int(len(observed) - observed[::-1].argmax()) + 1)
        if end - first < 4:
            return
        self.slider.set_clip(first, end)
        self.clip_changed(self.slider.clip_in, self.slider.clip_out)
        self.status.setText(f'Person found in frames {first + 1}-{end}. '
                            f'The range is clipped to the footage with the person in it.')

    def refresh_results(self):
        if self.project:
            signature = project_signature(self.project)
            if signature != self.preview_signature:
                self.preview_signature = signature
                self.data_loader.request(self.project)
            elif self.preview_data is not None:
                self.apply_preview()

    def clear_preview(self):
        self.preview_help.setText('')
        self.pelvis_auto_note.setText('Automatic lowering varies by frame.')
        self.data_loader.cancel()
        self.preview_signature = None
        self.slider.has_motion = False
        self.slider.set_observed(None)
        self.motion_view.observed_spans = None
        self.slider.update()
        self.video_view.set_observations({})
        self.motion_view.set_motion(None)
        self.refine_preview_note.hide()
        self.correction_note.hide()
        self.contact_tracks.clear()
        self.slider.set_contacts(None,identity=None,token=None)
        if self.preview_data is not None:
            self.preview_data.close()
            self.preview_data = None

    @batch_readiness
    def preview_ready(self, generation, result):
        if generation != self.data_loader.generation:
            result.close()
            return
        if result.signature != project_signature(result.project):
            # The worker committed another generation during loading. Never
            # pair these arrays with the new on-disk receipts.
            result.close()
            self.preview_signature=None
            self.refresh_results()
            return
        # Keep the current overlay while the same project reloads (a finished
        # stage, a correction) so the box and its points never blink out; a
        # different project must not inherit them.
        if self.preview_data is None or self.preview_data.project != result.project:
            self.video_view.set_observations({})
        previous=self.preview_data
        preserve_view=previous is not None and previous.project==result.project
        auto_fit=self.motion_view.auto_fit
        if preserve_view:self.motion_view.auto_fit=False
        self.preview_data = result
        self.contact_tracks=dict(getattr(result,'contact_tracks',{}))
        try:self.apply_preview()
        finally:
            self.motion_view.auto_fit=auto_fit
            self.restore_workspace_camera(consume=True)
            if previous is not None:previous.close()

    @batch_readiness
    @read_only_checks
    def apply_preview(self):
        data = self.preview_data
        signature=project_signature(data.project)
        if data.signature != signature:
            # Hide stale payloads without forgetting this project's camera
            # anchor while the next committed correction is loading.
            self.motion_view.set_motion(None,path=data.project/(self.active_result+'_joints.npz'))
            self.motion_view.observed_spans=None
            self.slider.set_observed(None)
            self.slider.has_motion=False;self.slider.update()
            if not self.busy and signature!=self.preview_signature:self.refresh_results()
            return
        self.video_view.set_observations(data.arrays)
        display_kind=self.active_result
        corrected=display_kind.startswith('corrected_')
        native_soma=self.model_choice.currentData()=='gemx'
        if native_soma and corrected and (display_kind+'_joints' not in data.arrays or not result_is_current(self.project, display_kind, ignore_contacts=True)):
            display_kind='initial'
        name = display_kind + "_joints"
        from .result_artifacts import artifacts_current
        coherent=artifacts_current(self.project,display_kind)
        current=coherent and result_is_current(self.project,display_kind)
        pending=coherent and corrected and not current and result_is_current(self.project,display_kind,ignore_contacts=True)
        available = name in data.arrays and (current or pending)
        self.correction_note.hide()
        self.refine_preview_note.hide()
        if native_soma and display_kind != self.active_result:
            self.correction_note.setText('Showing original SOMA motion. Use Apply Correction to update it.')
            self.correction_note.show()
        # A selected variant that was never computed (HandFlow before its
        # inference, or a correction that was never applied) must not blank the
        # viewport. Keep showing the closest available preview of the same stack
        # and state in the note which motion is displayed instead.
        fallback_note=''
        if not available:
            fallback_note=self.unavailable_result_note(display_kind)
            for candidate in self.preview_display_fallbacks(display_kind):
                if (candidate+'_joints' in data.arrays and artifacts_current(self.project,candidate)
                        and result_is_current(self.project,candidate)):
                    display_kind=candidate;name=display_kind+'_joints'
                    corrected=display_kind.startswith('corrected_');available=True
                    break
        if not available:
            note=self.correction_note if corrected else self.refine_preview_note
            note.setText(fallback_note)
            note.show()
        warning=' '.join(filter(None,(data.summaries.get(name,{}).get('correction_warning',''),data.summaries.get(name,{}).get('handflow_warning',''))))
        pelvis_summary=data.summaries.get(name,{})
        auto_drop=pelvis_summary.get('pelvis_auto_drop') if corrected else None
        self.pelvis_auto_note.setText('Automatic lowering varies by frame.' if auto_drop is None else
            f'Last applied: automatic lowering up to {auto_drop*100:.1f} cm; '
            f'automatic crouch lift up to {pelvis_summary.get("pelvis_auto_raise",0.)*100:.1f} cm; '
            f'user lift up to {pelvis_summary.get("pelvis_applied_lift",0.)*100:.1f} cm.')
        if self.people_root is not None:
            self.sync_shared_floor_controls()
        elif available and self.auto_floor.isChecked():
            self.floor_height.blockSignals(True)
            self.floor_height.setValue(float(data.summaries[name]['floor']))
            self.floor_height.blockSignals(False)
        if available and warning:
            self.correction_note.setText(warning);self.correction_note.show()
        if available and pending:
            message = ('Correction settings or contact timing changed. Preview shows the last applied correction.\nApply Correction to update the SOMA bones and surface.'
                       if native_soma else 'Contact-lock timing changed. Preview shows the last applied correction.\nApply Correction to update the motion and FBX.')
            self.correction_note.setText(message);self.correction_note.show()
        self.slider.has_motion = available
        self.slider.set_observed(getattr(data,'observed_spans',None))
        self.motion_view.observed_spans=getattr(data,'observed_spans',None)
        self.slider.update()
        overlay_name=display_kind+'_joints'
        overlay_current=artifacts_current(self.project,display_kind) and result_is_current(self.project,display_kind,ignore_contacts=True)
        overlay_available=overlay_current and overlay_name in data.arrays
        from .bone_overlay import preview_current
        target_current=overlay_available and (display_kind in data.target_assets if native_soma else
            preview_current(self.project,display_kind,data.target_info.get(display_kind)))
        camera_name=name if available else overlay_name
        rig_points=data.arrays.get(camera_name)
        # Source pose is the comparison reference. Never deform this surface
        # to make an imperfect export skeleton appear to fit it exactly.
        selected_mesh=None
        # Preserve a selected legacy baked preview if the export skin has not
        # been prepared yet. Never pair it with a different result's bones.
        self.motion_view.set_motion((rig_points if rig_points is not None else data.arrays.get(camera_name)) if available or overlay_available else None,
                                   data.summaries.get(camera_name),data.project/(camera_name+'.npz'),
                                   rotations=data.arrays.get(camera_name.replace('_joints','_rotations')) if available or overlay_available else None,
                                   mesh_frames=(selected_mesh if selected_mesh is not None else data.arrays.get(name.replace('_joints','_mesh'))) if available else None,mesh_asset=data.mesh_asset)
        self.motion_view.target_frames=data.arrays.get(display_kind+'_rig') if target_current else None
        self.motion_view.target_asset=data.target_assets.get(display_kind) if target_current else None
        self.motion_view.mesh_binding=self.motion_view.target_asset.get('binding') if selected_mesh is not None and self.motion_view.target_asset else None
        self.motion_view.raw_asset=None
        self.motion_view.native_vertices=None
        self.motion_view.native_required=False
        if native_soma or self.model_choice.currentData()=='flowhmr':
            self.motion_view.raw_asset=self.motion_view.target_asset
            self.motion_view.native_vertices=data.arrays.get(display_kind+'_native') if target_current else None
            self.motion_view.native_required=True
        info=data.target_info.get(display_kind,{})
        warning=info.get('correction',{}).get('warning','')
        source_warning=data.summaries.get(name,{}).get('correction_warning','') if corrected else ''
        if source_warning and source_warning not in warning:warning=' '.join(filter(None,(warning,source_warning)))
        help_text='The skeleton is fitted to the selected motion mesh. Small differences may remain.'
        if info.get('fit_version',0)>=15:
            help_text='The skeleton follows the selected motion mesh, with steadier arm and leg directions. Errors in the AI motion may remain.'
        if corrected:
            short=int(pelvis_summary.get('short_contact_segments',0))
            if short:help_text+=f'\n{short} short foot-lock intervals never reach full strength because of boundary blending.'
        notices=[]
        if fallback_note:notices.append(fallback_note)
        if available and pelvis_summary.get('occlusion_frames'):
            notices.append(f"{pelvis_summary['occlusion_frames']} frames where the person was not detected were re-estimated from the motion on both sides.")
        if not available and corrected:notices.append('Selected motion unavailable. Use Apply Correction to update the preview.')
        if warning:notices.append(warning)
        selected_fit=info.get('selected_source_fit',{})
        if selected_fit:
            help_text+=f'\nSkeleton fit: mean {selected_fit["mean_m"]*100:.1f} cm; max {selected_fit["max_m"]*100:.1f} cm.'
            facing=selected_fit.get('anatomical_facing_deg',{})
            for label,joints in (('Knee',('LeftLeg','RightLeg')),('Elbow',('LeftForeArm','RightForeArm'))):
                values=[facing[j] for j in joints if j in facing]
                if values:help_text+=f'\n{label} anatomical axis vs AI source: mean {sum(v["mean"] for v in values)/len(values):.1f}°; max {max(v["max"] for v in values):.1f}°.'
        fitted=info.get('selected_source_fit',info.get('source_fit',{})).get('joints',{})
        elbow_offset=max((fitted.get(side+'ForeArm',fitted.get('mixamorig:'+side+'ForeArm',{})).get('max_m',0.) for side in ('Left','Right')),default=0.)
        if elbow_offset>.08:
            notices.append(f'Elbow pose differs from AI source (up to {elbow_offset*100:.1f} cm) after anatomical alignment.')
        if self.model_choice.currentData()=='flowhmr':
            from .flow_placement import needs_update
            if needs_update(self.project):notices.append('FlowHMR scene placement needs updating. Use Prepare Preview; no AI inference is required.')
            knee_offset=max((fitted.get(side+'Leg',{}).get('max_m',0.) for side in ('Left','Right')),default=0.)
            if knee_offset>.08:notices.append(f'Knee pose differs from AI source (up to {knee_offset*100:.1f} cm) after anatomical alignment.')
        if native_soma:
            help_text='Native SOMA surface and 77-bone overlay.'
        self.bone_overlay.setToolTip(tooltip_text(help_text if native_soma else 'Show the skeleton used for FBX export. Display only.\n'
            'In Blender, use Scale 1 and leave Automatic Bone Orientation off to keep its axes.\n'+help_text))
        if available and pending:notices.append('Contact-lock timing changed. Preview shows the last applied correction. Apply Correction to update the motion and FBX.')
        if corrected and available and not pending and not fallback_note and not result_is_current(self.project,self.active_result):notices.append('The last applied correction is outdated or unavailable. Apply Correction before export.')
        self.preview_help.setText(help_text)
        self.correction_note.setText('\n'.join(dict.fromkeys(notices)));self.correction_note.setVisible(bool(notices))
        if available and self.motion_view.target_asset is None and not self.busy and os.environ.get('QT_QPA_PLATFORM')!='offscreen' and (self.project/RESULTS[base_result(display_kind)][0]).is_file():
            token=(str(self.project),display_kind,self.preview_signature)
            attempts=getattr(self,'_rig_attempts',deque(maxlen=16))
            if token not in attempts:
                attempts.append(token);self._rig_attempts=attempts
                QTimer.singleShot(0,self.prepare_target_preview)
        self.update_foot_contacts()
        self.update_readiness()

    def preview_failed(self, generation, message):
        if generation == self.data_loader.generation:
            changed=self.project is not None and self.preview_signature!=project_signature(self.project)
            self.preview_signature = None
            if changed:
                self.refresh_results()
                return
            self.status.setText("Cannot load the motion preview. " + message)

    def save_fbx(self):
        if self.selected_character() is not None:
            self.export_character(self.selected_character());return
        if self.busy or self.current_export() is None:
            self.status.setText('Selected motion is unavailable or has pending correction edits. Apply Correction / Prepare Preview before exporting.');return
        from .source_export_dialog import SourceExportDialog
        dialog=SourceExportDialog(self)
        if dialog.exec()!=QDialog.Accepted:return
        if self.current_export() is None:
            self.status.setText('Motion changed while choosing export options. Apply Correction / Prepare Preview first.');return
        self._source_export_options=dialog.options()
        if self._source_export_options['format']=='blend_scene':
            from .blend_export_ui import export_scene
            export_scene(self);return
        if self.export_checked_people():return
        source = self.current_export()
        if source:
            from .person_names import export_stem
            fmt=self._source_export_options['format']
            default=export_stem(self.export_person_name(self.project))+'.'+fmt
            name, _ = QFileDialog.getSaveFileName(self, "Export", default, fmt.upper()+" (*."+fmt+")",options=QFileDialog.DontConfirmOverwrite if fmt=='blend' else QFileDialog.Option(0))
            if name:
                if not Path(name).suffix:name+='.'+fmt
                self.submit_fbx_exports([(self.project,self.active_result,source,Path(name))])

    def submit_fbx_exports(self,files):
        if self.busy or not files or self.current_export() is None:return
        import uuid
        from .storage import atomic_json
        request=self.project/('.fbx-export-'+uuid.uuid4().hex+'.json')
        rows=[]
        if getattr(self,'_source_export_options',{}).get('format')=='blend':
            from .blend_export_ui import targets
            files=targets(files)
        try:
            for project,kind,source,target in files:
                row=dict(project=str(project),kind=kind,source=str(source),target=str(target))
                row['rig_name']=self.export_person_name(project)
                row['prepend_tpose']=getattr(self,'_source_export_options',{}).get('prepend_tpose',False)
                row['bvh_profile']=getattr(self,'_source_export_options',{}).get('bvh_profile','generic')
                for key in ('place_at_origin','symmetric_rest'):
                    row[key]=getattr(self,'_source_export_options',{}).get(key,False) and getattr(self,'_source_export_options',{}).get('format','fbx') in ('fbx','bvh')
                row['origin_group']=str(Path(project).resolve())
                if self.people_state:
                    from .people_store import safe_child
                    for person in self.people_state['people']:
                        if any(safe_child(self.people_root,t['project'])==Path(project).resolve() for t in person.get('takes',[])):
                            row['origin_group']='person:'+str(person['id']);break
                for key,default in (('animate_camera',True),('background_refine',False),('camera_translation_scale',1.)):
                    row[key]=getattr(self,'_source_export_options',{}).get(key,default)
                if self.people_root is not None:
                    from .scene_export import context
                    row['scene']=context(self.people_root,project)
                rows.append(row)
        except (OSError,ValueError,KeyError) as error:
            self.status.setText('Cannot export shared scene: '+str(error));return
        atomic_json(request,rows)
        self._motion_export_targets=[str(target) for _,_,_,target in files]
        self._fbx_export_cleanup=(request,[target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial') for _,_,_,target in files])
        stage='copy_bvh' if getattr(self,'_source_export_options',{}).get('format','fbx')=='bvh' else 'save_fbx'
        if getattr(self,'_source_export_options',{}).get('format')=='blend':
            from .blend_support import partial
            stage='save_blend';self._fbx_export_cleanup=(request,[partial(target.parent,request) for _,_,_,target in files])
        self.start_jobs([(stage,['--project',str(self.project),'--output',str(request)])])

    def cleanup_fbx_export(self):
        pending=getattr(self,'_fbx_export_cleanup',None)
        if pending:
            request,paths=pending
            for path in [request,*paths]:
                try:
                    if path.is_dir():
                        from .blend_support import cleanup
                        cleanup(path,request)
                    else:path.unlink(missing_ok=True)
                except (OSError,ValueError):pass
            self._fbx_export_cleanup=None

    def open_folder(self):
        from .project_storage import storage_root
        target = self.workspace_root or ((storage_root(self.project) or self.project) if self.project else PROJECTS)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.busy:
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and not self.busy:
            if self._pear_active:self.load_pear_videos([Path(url.toLocalFile()) for url in urls])
            elif self.model_choice.currentData()=='mamma':self.load_mamma_videos([Path(url.toLocalFile()) for url in urls])
            else:
                local = urls[0].toLocalFile()
                if local:
                    self.load_video(Path(local))
                else:
                    from .video_url import is_url
                    if is_url(urls[0].toString()):
                        self.choose_video_url(urls[0].toString())

    def closeEvent(self, event):
        if getattr(self, '_preview_export', None) is not None:
            self._preview_export.cancel(close_window=True)
            event.ignore()
            return
        self.stop_preloaded_worker()
        if self.busy and self.process:
            answer = QMessageBox.question(self, "Processing", "Cancel the current job and close the application?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel_job()
            worker = self.process
            if worker:
                worker.waitForFinished(3000)
        self.pear_leave()
        self.pear_stop_scan()
        if self.character_process is not None:
            self.cancel_character();self.character_process.waitForFinished(5000)
        if getattr(self, 'live_channel', None) is not None:
            self.stop_live_body()
        source_fetch = getattr(self, 'source_fetch', None)
        if source_fetch is not None:
            # A download still running must not be destroyed under its thread.
            source_fetch.stopping = True
            source_fetch.wait(10000)
            self.close_source_fetch()
        if getattr(self, 'live_link', None) is not None:
            self.live_link.stop()
        if getattr(self, 'cascadeur_link', None) is not None:
            self.cascadeur_link.stop()
            try:
                self.cascadeur_link.disconnect(wait=1.0)
            except Exception:
                pass
        self.character_loader.close()
        if getattr(self, 'camera_check', None) is not None:
            self.camera_check.close()
        self.task_timer.stop()
        if self.workspace_root is not None:
            try:self.remember_workspace_model()
            except (OSError,ValueError) as error:self.log.append('Could not save project view: '+str(error))
        self.shutdown_preview()
        event.accept()
