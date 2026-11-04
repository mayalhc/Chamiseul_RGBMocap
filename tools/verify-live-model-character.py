"""Check that the live character follows the AI Motion Model selection.

  runtime/python.exe -B tools/verify-live-model-character.py [workspace]

Opens the model-folder workspace on the real window and, for each model that has
a prepared result, switches AI Motion Model, opens the camera and starts the
live character the way Find person does. It records the rig and skin binding the
live worker was handed (the character identity), what the 3D view will skin, and
a screenshot per model.

It then repeats the thing a person actually does: leave the character running
and switch the model. The live character must come back as the newly selected
model's own character, not keep following the previous model's rig.

Frames only appear once the solver sees a person, so the identity checks above
are what is asserted; a person in front of the camera additionally fills in
live_frames/mesh_frames and makes the screenshots show the body.
"""
from __future__ import annotations

import argparse
import hashlib
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

from mocap.runtime_registry import activate  # noqa: E402

activate(APP, "gui")

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from mocap import project_workspace as workspace  # noqa: E402
from mocap.gui import MainWindow  # noqa: E402

DEFAULT_WORKSPACE = APP / "projects/Download_35b14c2294807c_project"
MODELS = ("exmr", "gvhmr", "flowhmr")
REPORT = BASE / "screenshots/live-model-character-report.json"
SHOTS = BASE / "screenshots"

parser = argparse.ArgumentParser()
parser.add_argument("workspace", nargs="?", default=str(DEFAULT_WORKSPACE))
parser.add_argument("--models", default="", help="comma separated model keys (default: every model with a saved result)")
parser.add_argument("--skip-switch", action="store_true", help="only check each model on its own")
args = parser.parse_args()
root = Path(args.workspace).resolve()
report = {"workspace": str(root), "models": {}, "switch": {}, "camera": None, "errors": []}
original = None


def log(message):
    print('%6.1f %s' % (time.monotonic() % 1000, message), flush=True)


def model_project(model):
    """The folder this model owns inside the workspace, or None."""
    try:
        return workspace.model_project(root, model)
    except (OSError, ValueError, KeyError) as error:
        report["errors"].append(f"{model}: {error}")
        return None


def skin_summary(path):
    """Height, vertex count and identity hash of a prepared skin binding."""
    if not path:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"], np.float64)
            frames = list(np.asarray(data["frames"]).shape)
    except (OSError, ValueError, KeyError) as error:
        report["errors"].append(f"{path}: {error}")
        return None
    return dict(vertices=int(vertices.shape[0]), frames=frames,
                height=round(float(vertices[:, 1].max() - vertices[:, 1].min()), 4),
                digest=hashlib.sha256(np.ascontiguousarray(vertices, np.float32).tobytes()).hexdigest()[:16])


def rig_rest(path):
    """The 22-joint rest of a take's prepared character rig."""
    if not path:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["rest"], np.float32)[:22]
    except (OSError, ValueError, KeyError) as error:
        report["errors"].append(f"{path}: {error}")
        return None


def rig_digest(rest):
    if rest is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(rest, np.float32).tobytes()).hexdigest()[:16]


def drawn_bound_to(window, rest):
    """Does the surface the 3D view skins sit on this take's own rig?

    A take that writes a display binding is checked by that skin's identity.
    FlowHMR keeps native SMPL-H, so the live worker skins the base surface on
    the take's rig; that binding's rest IS the take's rig rest.
    """
    drawn = getattr(window, "_live_mesh_binding", None)
    if rest is None or not drawn or "rest" not in drawn:
        return False
    return bool(np.allclose(np.asarray(drawn["rest"], np.float32)[:22], rest, atol=1e-5))


def prepared_assets(project):
    """Every prepared character rig in a model's project folder.

    FlowHMR prepares a native SMPL-H rig with a generation token
    (<kind>_flow-<token>-rig-asset.npz); the other models write
    <result>_rig-<token>-asset.npz.
    """
    if not project:
        return []
    folder = Path(project)
    return sorted(folder.glob('*_rig-*-asset.npz')) + sorted(folder.glob('*-rig-asset.npz'))


def camera_session():
    """The live preview capture the Webcam dialog would hand to the workspace."""
    from mocap import camera_capture
    devices = camera_capture.devices()
    if not devices:
        return None, "no camera"
    device = devices[0]
    sizes = camera_capture.formats(device)
    chosen = next((entry for entry in sizes if (entry["width"], entry["height"]) == (1920, 1080)),
                  sizes[0] if sizes else {"width": 1280, "height": 720, "fps": 30})
    session = camera_capture.Session(camera_capture.default_path(APP), device,
                                     chosen["width"], chosen["height"], chosen["fps"],
                                     preview_width=min(960, chosen["width"]))
    try:
        session.start_preview()
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return None, "camera failed: " + str(error)[:120]
    return session, device


def drawn_digest(window):
    """Identity of the skin the 3D view is holding for the live character."""
    drawn = getattr(window, "_live_mesh_binding", None)
    if not drawn or "vertices" not in drawn:
        return None
    vertices = np.ascontiguousarray(np.asarray(drawn["vertices"], np.float32))
    return hashlib.sha256(vertices.tobytes()).hexdigest()[:16]


def worker_state(window):
    """What the running live worker was told to present."""
    worker = getattr(window, "live_worker", None)
    argv = list(worker.args) if worker is not None else []
    rig = argv[argv.index("--rig") + 1] if "--rig" in argv else ""
    binding = argv[argv.index("--binding") + 1] if "--binding" in argv else ""
    drawn = getattr(window, "_live_mesh_binding", None)
    try:
        drawn_key = str(drawn["key"][0]) if drawn and drawn.get("key") else ""
    except (AttributeError, IndexError, TypeError):
        drawn_key = ""
    channel = getattr(window, "live_channel", None)
    return dict(pid=getattr(worker, "pid", None), rig=rig, binding=binding, drawn=drawn_key,
                channel=bool(channel), live_frames=int(getattr(window, "live_shown", 0)),
                mesh_frames=int(channel.mesh_count()) if channel is not None else 0,
                status=window.status.text(), filename=window.filename.text())


def in_project(path, project):
    if not path or project is None:
        return False
    return Path(path).resolve().is_relative_to(Path(project).resolve())


def main():
    global original
    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1400, 900)
    if not window.load_project(root):
        report["error"] = "workspace did not open: " + window.status.text()
        return finish(app, window, 1)
    window.show()
    original = window.model_choice.currentData()
    listed = [window.model_choice.itemData(index) for index in range(window.model_choice.count())]
    candidates = [key for key in (args.models.split(",") if args.models else MODELS) if key] or listed
    wanted = [key for key in candidates if key in listed and model_project(key) is not None
              and (model_project(key) / "project.json").is_file()]
    report["models_in_selector"] = listed
    report["original_model"] = original
    report["checked_models"] = wanted
    report["folders"] = {model: str(model_project(model)) for model in listed}
    log('selector %s' % listed)
    log('checking %s' % wanted)

    state = {"index": -1, "phase": "ready", "since": time.monotonic(), "model": None,
             "session": None, "shot": "model", "before": None, "started": 0.0,
             "switched": False}

    def snapshot(tag):
        path = SHOTS / f"live-model-{tag}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        app.processEvents()
        window.grab().save(str(path))
        return path

    def next_model():
        state["index"] += 1
        if state["index"] >= len(wanted):
            state["phase"] = "end"
            begin_switch()
            return
        state["model"] = wanted[state["index"]]
        state["phase"] = "ready"
        state["since"] = time.monotonic()

    def begin_switch():
        """Leave the character running and switch the model under it."""
        if state["switched"] or args.skip_switch or len(wanted) < 2:
            restore_and_exit()
            return
        state["switched"] = True
        state["before"] = wanted[0]
        state["model"] = wanted[1]
        state["shot"] = "switch"
        state["phase"] = "ready"
        state["since"] = time.monotonic()

    def choose(model):
        if window.busy or getattr(window, "contact_saving", False):
            if time.monotonic() - state["since"] > 600:
                log(f'{model}: app stayed busy')
                report["models"][model] = {"error": "app stayed busy"}
                state["phase"] = "next"
            return False
        index = window.model_choice.findData(model)
        if index < 0:
            log(f'{model}: not in the AI Motion Model selector')
            report["models"][model] = {"error": "not in the selector"}
            state["phase"] = "next"
            return False
        log(f'{model}: selecting (current project {window.project})')
        state["phase"] = "select"
        state["since"] = time.monotonic()
        window.model_choice.setCurrentIndex(index)
        return True

    def record_model(model):
        project = model_project(model)
        prepared = prepared_assets(project)
        now = worker_state(window)
        rest = rig_rest(now["rig"])
        skin, drawn = skin_summary(now["binding"]), drawn_digest(window)
        bound = drawn_bound_to(window, rest)
        now.update(project=str(project), expects_character=bool(prepared),
                   drawn_digest=drawn, rig_rest_digest=rig_digest(rest),
                   bound_to_take_rig=bound,
                   rig_in_own_folder=in_project(now["rig"], project),
                   binding_in_own_folder=(not now["binding"]
                                          or in_project(now["binding"], project)),
                   skin=skin,
                   drawn_matches_binding=bool(drawn) and bool(skin) and drawn == skin["digest"],
                   screenshot=str(snapshot(model).relative_to(BASE)))
        report["models"][model] = now
        log(f'{model}: rig={now["rig"]} binding={now["binding"]} '
            f'own_folder={now["rig_in_own_folder"]}/{now["binding_in_own_folder"]} '
            f'view_skins_it={now["drawn_matches_binding"] or bound}')

    def record_switch():
        before, after = state["before"], state["model"]
        project = model_project(after)
        now = worker_state(window)
        skin = skin_summary(now["binding"])
        bound = drawn_bound_to(window, rig_rest(now["rig"]))
        report["switch"] = dict(
            before=before, after=after,
            before_rig=report["models"].get(before, {}).get("rig"),
            rig=now["rig"], binding=now["binding"],
            # The claim is that the rig now belongs to the newly selected
            # model's own folder; which take the workspace points at can move
            # while its previews prepare.
            rig_in_new_folder=(in_project(now["rig"], project)
                               or in_project(now["rig"], root / after)),
            drawn_matches_new_binding=(bool(drawn_digest(window)) and bool(skin)
                                       and drawn_digest(window) == skin["digest"]) or bound,
            bound_to_new_take_rig=bound,
            rig_changed=now["rig"] != report["models"].get(before, {}).get("rig"),
            pid=now["pid"], pid_changed=now["pid"] != report["models"].get(before, {}).get("pid"),
            screenshot=str(snapshot("switch").relative_to(BASE)),
            status=now["status"])
        log(f'switch {before}->{after}: rig={now["rig"]} in_new_folder={in_project(now["rig"], project)} '
            f'changed={report["switch"]["rig_changed"]}')

    def tick():
        if state["model"] is None:
            return
        if state["phase"] == "ready":
            if state["session"] is None:
                session, detail = camera_session()
                report["camera"] = report["camera"] or detail
                if session is None:
                    report["errors"].append(detail)
                    state["phase"] = "next"
                    return
                state["session"] = session
                window.start_live_camera(session)
            if choose(state["model"]):
                pass
            return
        if state["phase"] == "select":
            target = model_project(state["model"])
            loaded = target is not None and window.project is not None \
                and Path(window.project).resolve() == Path(target).resolve()
            if loaded and not window.busy:
                # Find person's 20 s walk-in wait is not needed: nothing is
                # recorded here and the character identity is already decided.
                log(f'{state["model"]}: project loaded, starting the live character')
                window.start_live_body()
                state["started"] = time.monotonic()
                state["phase"] = "settle"
            elif time.monotonic() - state["since"] > 180:
                report["models"].setdefault(state["model"], {})["error"] = "project never loaded"
                state["phase"] = "next"
            return
        if state["phase"] == "settle":
            # The worker is spawned synchronously; this grace period lets it and
            # the view settle, and catches frames if a person is in view. A
            # take with no SMPL-X display binding (FlowHMR keeps SMPL-H) gets
            # its surface from the worker itself, so wait for that binding.
            elapsed = time.monotonic() - state["started"]
            footprints = bool(getattr(window, "live_worker", None)) and not worker_state(window)["binding"]
            if elapsed > (90 if footprints else 8):
                if state["shot"] == "switch":
                    record_switch()
                else:
                    record_model(state["model"])
                state["phase"] = "next"
            return
        if state["phase"] == "next":
            state["phase"] = "idle"
            QTimer.singleShot(800, next_model)
            return
        if state["phase"] == "end":
            state["phase"] = "idle"
            return

    def restore_and_exit():
        session = state["session"]
        window.stop_live_camera()
        if session is not None:
            session.stop()
        if args.skip_switch and len(wanted) > 1:
            # The switch status is "not checked", not "passed".
            report["switch"] = {"skipped": True}
        if original:
            window.model_choice.setCurrentIndex(window.model_choice.findData(original))
            app.processEvents()
        entries = [entry for entry in report["models"].values() if "error" not in entry]
        # A model with no prepared result has no character of its own; the live
        # view then shows the neutral base body, which is correct, not a miss.
        ok = bool(entries) and len(entries) == len(report["models"]) and all(
            (entry["rig_in_own_folder"] and entry["binding_in_own_folder"]
             and (entry["drawn_matches_binding"] or entry["bound_to_take_rig"]))
            if entry.get("expects_character") else (not entry["rig"] and not entry["binding"])
            for entry in entries)
        digests = {entry.get("rig_rest_digest") or (entry["skin"] or {}).get("digest")
                   for entry in entries if entry.get("rig_rest_digest") or entry.get("skin")}
        report["distinct_characters"] = len(digests) > 1
        if report["switch"] and not report["switch"].get("skipped"):
            ok = ok and report["switch"]["rig_in_new_folder"] and report["switch"]["rig_changed"] \
                and report["switch"]["drawn_matches_new_binding"] and report["switch"]["pid_changed"]
        report["pass"] = bool(ok)
        finish(app, window, 0 if ok else 1)

    def finish(app, window, code):
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("VERIFY-LIVE-MODEL-CHARACTER", json.dumps(report), flush=True)
        try:
            window.shutdown_preview()
        except Exception:  # noqa: BLE001 - leaving anyway
            pass
        app.exit(code)
        return code

    QTimer.singleShot(800, next_model)
    timer = QTimer()
    timer.setInterval(500)
    timer.timeout.connect(tick)
    timer.start()
    QTimer.singleShot(600000, lambda: (report["errors"].append("timed out"), finish(app, window, 1)))
    app.exec()
    os._exit(0 if report.get("pass") else 1)


if __name__ == "__main__":
    sys.exit(main() or 0)
