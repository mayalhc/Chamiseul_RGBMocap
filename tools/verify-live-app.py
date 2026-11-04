"""Check the app side of the live link with the real window.

  runtime/python.exe -B tools/verify-live-app.py [take_project] [seconds]

Opens the 3-person project, enables File > Live Link to Blender, presents a run
of frames, and receives them with a socket client. Verifies the handshake, the
frame packets, that consecutive frames actually differ, and that the last packet
equals what the viewport was showing.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
BASE = APP.parent
sys.path.insert(0, str(APP))
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ.update(RGB_MOCAP_MANAGED="1", RGB_MOCAP_DATA=str(APP), PYTHONDONTWRITEBYTECODE="1",
                  HF_HOME=str(APP / "cache/huggingface"), TORCH_HOME=str(APP / "cache/torch"),
                  XDG_CACHE_HOME=str(APP / "cache"))

DEFAULT_TAKE = (APP / "projects/Download_35b14c2294807c_project/exmr/people"
                / "bbca4b505984444eb3248fd8f62c948e/person_0001/takes/000000000_000000822")

from mocap.runtime_registry import activate  # noqa: E402

activate(APP, "gui")

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from mocap.gui import MainWindow  # noqa: E402
from mocap.live_link import MAGIC_JOINTS as MAGIC, NAMES, to_blender  # noqa: E402

TAKE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TAKE
PORT = 9234
os.environ['RGB_MOCAP_LIVE_PORT'] = str(PORT)
REPORT = BASE / "screenshots/live-app-report.json"
report = {"take": str(TAKE), "port": PORT}
packets = []
handshake = {}
reader_done = threading.Event()


def read_client(stop):
    try:
        client = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    except OSError as error:
        report["client_error"] = str(error)
        reader_done.set()
        return
    client.settimeout(0.5)
    buffer = b""
    size = 4 + 8 + len(NAMES) * 12
    try:
        # The handshake is one JSON line; only then do binary frames follow.
        while b"\n" not in buffer and not stop.is_set():
            try:
                buffer += client.recv(65536)
            except socket.timeout:
                continue
        line, _, buffer = buffer.partition(b"\n")
        if line:
            handshake.update(json.loads(line.decode("utf-8")))
        while not stop.is_set():
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= size:
                packet, buffer = buffer[:size], buffer[size:]
                frame, fps = struct.unpack("<If", packet[4:12])
                values = np.array(struct.unpack("<%df" % (len(NAMES) * 3), packet[12:]))
                packets.append((frame, fps, values.reshape(-1, 3), packet[:4]))
    finally:
        client.close()
        reader_done.set()


def main():
    app = QApplication(sys.argv[:1])
    window = MainWindow(None)
    window.resize(1200, 800)
    window.load_project(TAKE)
    window.show()
    state = {"ticks": 0, "stop": threading.Event(), "thread": None}

    def finish(code):
        state["stop"].set()
        state["thread"].join(3) if state["thread"] else None
        window.live_link_action.setChecked(False)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        app.exit(code)

    def step():
        state["ticks"] += 1
        if window.info is None or not window.video_readable:
            if state["ticks"] > 120:
                report["error"] = "video never became readable"
                return finish(1)
            return QTimer.singleShot(300, step)
        start()

    def start():
        action = getattr(window, "live_link_action", None)
        report["menu_action"] = dict(present=action is not None,
                                     text=action.text() if action else None,
                                     checkable=bool(action.isCheckable()) if action else None)
        if action is None:
            report["error"] = "menu action missing"
            return finish(1)
        state["thread"] = threading.Thread(target=read_client, args=(state["stop"],), daemon=True)
        state["thread"].start()
        time.sleep(0.3)
        action.setChecked(True)
        report["status_after_enable"] = window.status.text()
        report["link"] = window.live_link.status() if getattr(window, "live_link", None) else None
        QTimer.singleShot(400, play)

    def play():
        frames = list(range(120, 150))
        report["presented_frames"] = frames
        for frame in frames:
            window.present_frame(frame)
            app.processEvents()
            time.sleep(1.0 / 45.0)
        report["applied_pose_frame"] = frames[-1]
        QTimer.singleShot(400, verify)

    def verify():
        if not packets:
            report["error"] = "no frames received"
            report["handshake"] = handshake
            return finish(1)
        frames = [item[0] for item in packets]
        unique = np.array([item[2] for item in packets])
        last_frame, fps, points, magic = packets[-1]
        expected = to_blender(window.live_pose(report["applied_pose_frame"]))
        report["handshake"] = dict(version=handshake.get("version"), joints=len(handshake.get("names", [])),
                                   names_match=handshake.get("names") == NAMES,
                                   label=handshake.get("label"), fps=handshake.get("fps"),
                                   frames=handshake.get("frames"))
        report["packets"] = dict(count=len(packets), magic=bool(magic == MAGIC),
                                 first_frame=frames[0], last_frame=frames[-1],
                                 increasing=all(b > a for a, b in zip(frames, frames[1:])),
                                 motion_range_m=float(np.ptp(unique, axis=0).max()),
                                 unique_frames=int(len(set(frames))),
                                 fps=fps)
        report["position_error"] = float(np.abs(points - expected).max())
        report["link_status"] = window.live_link.status() if getattr(window, "live_link", None) else None
        ok = (report["menu_action"]["present"] and report["packets"]["magic"]
              and report["packets"]["increasing"] and report["packets"]["motion_range_m"] > 0.01
              and report["handshake"]["names_match"] and report["position_error"] < 1e-5)
        report["pass"] = bool(ok)
        print("VERIFY-LIVE-APP", json.dumps(report), flush=True)
        finish(0 if ok else 1)

    QTimer.singleShot(800, step)
    QTimer.singleShot(90000, lambda: (report.setdefault('error', 'timed out'), finish(1)))
    app.exec()
    # Qt teardown of this preview window ends the process with its own status,
    # so leave deliberately: the verdict is the test result.
    os._exit(0 if report.get('pass') else 1)


if __name__ == "__main__":
    raise SystemExit(main())
