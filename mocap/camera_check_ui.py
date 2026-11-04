"""Off-thread camera-motion measurement for the workspace hint.

Decoding sampled frames takes about a second, so it never runs on the GUI thread.
"""
from pathlib import Path
from threading import Lock
from PySide6.QtCore import QThread, Signal
from .camera_motion import report


class CameraCheck(QThread):
    ready = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lock = Lock()
        self.pending = None
        self.stop = False

    def check(self, project):
        """Queue one project; a newer request replaces an unstarted one."""
        with self.lock:
            self.pending = str(Path(project))
            self.stop = False
        if not self.isRunning():
            self.start()

    def close(self):
        with self.lock:
            self.stop = True
            self.pending = None
        self.wait()

    def run(self):
        while True:
            with self.lock:
                if self.stop or self.pending is None:
                    return
                path = self.pending
                self.pending = None
            try:
                value = report(path)
            except Exception:
                value = None
            if not self.stop:
                self.ready.emit(path, value)
