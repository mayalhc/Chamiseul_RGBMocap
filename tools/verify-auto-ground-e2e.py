"""End-to-end check: run the app's own 3-person reconstruction, then verify the
automatic Flat Ground grounding from both the app state and the saved reports.

The group's saved mode is deliberately set to Multi Level first (the legacy state
that used to leave everyone floating), so the run has to override it and
recompute real corrected results. Motion inference stages are reused from their
existing receipts; correction and export run for real (~3 s / 6 s per actor).

Run from the app directory:
  runtime/python.exe -B tools/verify-auto-ground-e2e.py [group_root] [timeout_minutes]
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
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ.update(RGB_MOCAP_MANAGED="1", RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE="1",
                  HF_HOME=str(APP / "cache/huggingface"), TORCH_HOME=str(APP / "cache/torch"),
                  XDG_CACHE_HOME=str(APP / "cache"))

GROUP = Path(sys.argv[1] if len(sys.argv) > 1 else
             APP / "projects/Download_35b14c2294807c_project/exmr").resolve()
TIMEOUT_MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
KIND = "corrected_initial"
SHOT = BASE / "screenshots/auto-ground-e2e.png"
REPORT_PATH = BASE / "screenshots/auto-ground-e2e.json"

from mocap.runtime_registry import activate  # noqa: E402

activate(APP, "gui")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from mocap.gui import MainWindow  # noqa: E402
from mocap.people_store import load_people  # noqa: E402
from mocap.scene_floor import ensure_config, read_config, save_config  # noqa: E402
from mocap.motion_tools import result_is_current  # noqa: E402

report = {"group": str(GROUP), "timeout_minutes": TIMEOUT_MINUTES}
started = time.monotonic()


def write_report():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def takes():
    state = load_people(GROUP)
    return [(person["name"], GROUP / take["project"])
            for person in state["people"] if person.get("checked")
            for take in person.get("takes", [])]


def metrics():
    rows = []
    for name, project in takes():
        path = project / f"{KIND}_correction.json"
        row = {"person": name, "report_present": path.is_file(),
               "current": result_is_current(project, KIND)}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("mode", "floor", "max_contact_goal_error_m", "contact_sole_height_after_m",
                        "sole_below_floor_after_m", "contact_sole_penetration_over_1cm"):
                if key in data:
                    row[key] = data[key]
        rows.append(row)
    return rows


def main() -> int:
    # 1. Legacy state: a saved Multi Level plane that used to leave feet floating.
    save_config(GROUP, {**ensure_config(GROUP), "mode": "multi_level"})
    report["before"] = metrics()
    report["before_mode"] = read_config(GROUP).get("mode", "flat")

    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1500, 950)
    first_take = takes()[0][1]
    window.load_project(first_take)
    window.show()
    window.raise_()
    window.motion_view.show_names = True
    report["loaded_project"] = str(window.project)
    report["people_root"] = str(window.people_root) if window.people_root else None

    state = {"ticks": 0, "armed": False, "log_at_start": 0}

    def finish(code):
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_report()
        app.exit(code)

    def wait_for_input():
        state["ticks"] += 1
        if state["ticks"] > 200:
            report["error"] = "video preview never became readable"
            return finish(1)
        if window.info is None or not window.video_readable:
            return QTimer.singleShot(300, wait_for_input)
        start_pipeline()

    def start_pipeline():
        window.motion_correction.setChecked(True)
        # The saved Multi Level must still be visible before the run starts.
        report["mode_before_run"] = window.correction_mode.currentData()
        report["preset"] = window.correction_preset.currentData()
        window.preset_description.setText("")
        state["log_at_start"] = len(window.log.toPlainText().splitlines())
        report["result_kind_requested"] = None
        window.run_full()
        report["busy_after_run_full"] = bool(window.busy)
        report["queued_jobs"] = [stage for stage, _ in getattr(window, "jobs", [])]
        QTimer.singleShot(500, poll)

    def poll():
        state["ticks"] += 1
        if state["ticks"] > int(TIMEOUT_MINUTES * 60 / 0.5):
            report["error"] = "pipeline timed out"
            report["current_stage"] = getattr(window, "current_stage", None)
            return screenshot(1)
        if window.busy:
            report["current_stage"] = getattr(window, "current_stage", None)
            return QTimer.singleShot(500, poll)
        report["status_text"] = window.status.text()
        report["failed"] = bool(window.failed)
        report["cancelled"] = bool(window.cancelled)
        report["cancelled_or_failed"] = bool(window.failed or window.cancelled)
        report["mode_after_run"] = window.correction_mode.currentData()
        report["active_result"] = window.active_result
        lines = window.log.toPlainText().splitlines()
        report["log_tail"] = lines[-25:]
        # The shared scene reloads asynchronously after the batch.
        state["ticks"] = 0
        QTimer.singleShot(500, wait_scene)

    def wait_scene():
        state["ticks"] += 1
        ready = window.scene_data is not None and len(window.scene_data.takes) >= 3
        if not ready and state["ticks"] > 120:
            report["scene_error"] = "shared scene did not reload"
            return screenshot(1)
        if not ready:
            return QTimer.singleShot(500, wait_scene)
        present()
        QTimer.singleShot(1200, lambda: screenshot(0))

    def present():
        span = float(getattr(window, "source_end", 0.0) or 0.0) or 27.4
        frame = int(round(min(span * 0.5, span - 0.5) * 30))
        window.present_frame(frame)
        window.motion_view.frame_all()
        window.motion_view.update()
        window.motion_view.repaint()
        app.processEvents()
        report["frame"] = frame

    def screenshot(code):
        from mocap.capture import capture_window
        window.present_frame(report.get("frame", 0))
        window.motion_view.update()
        window.motion_view.repaint()
        app.processEvents()
        image = capture_window(window)
        SHOT.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(SHOT))
        report["screenshot"] = str(SHOT)
        report["after"] = metrics()
        # A saved flat plane omits the key entirely; anything else is explicit.
        report["after_mode"] = read_config(GROUP).get("mode", "flat")
        report["scene"] = dict(
            people=len(window.scene_data.takes) if window.scene_data else 0,
            presented=len(window.motion_view.scene_presented))
        report["correction_mode_help"] = window.floor_description.text()
        finish(code)

    QTimer.singleShot(1200, wait_for_input)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
