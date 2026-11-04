"""Live preview and capture from a webcam, on the ffmpeg the app already ships.

One ffmpeg process owns the camera: it writes the H.264 clip and, from the same
input, a small raw stream that the dialog paints as a preview. Opening the
camera twice is not possible on Windows, so preview and recording have to be the
same capture - that is what ``Session`` is for.

The camera is asked for a rate it rarely reaches (this QHD webcam delivers about
22 fps at 1080p, not 30). Wallclock timestamps plus variable frame timing keep
the clip the length it really was; without them a 5 s capture played back in
3.3 s and every motion was sped up.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .video import executable, probe

CREATE_NO_WINDOW = 0x08000000

# A killed parent used to leave its ffmpeg holding the camera for days - a
# zombie that Windows would not even let a normal taskkill touch. Assigning
# every capture process to one job object with KILL_ON_JOB_CLOSE makes the
# children die with this process, whatever way it dies.
_JOB_HANDLE = None


def _camera_job():
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    import ctypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('ReadOperationCount', 'WriteOperationCount',
                     'OtherOperationCount', 'ReadTransferCount',
                     'WriteTransferCount', 'OtherTransferCount')]

    class BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                    ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', ctypes.c_uint32),
                    ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', ctypes.c_uint32),
                    ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', ctypes.c_uint32),
                    ('ProcessLimit', ctypes.c_uint32)]

    class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BASIC_LIMIT_INFORMATION),
                    ('IoInfo', IO_COUNTERS),
                    ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateJobObjectW(None, None)
    limits = EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(limits),
                                            ctypes.sizeof(limits)):
        return None
    _JOB_HANDLE = handle
    return handle


def _assign_to_job(process):
    handle = _camera_job()
    if handle is None or process is None:
        return
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.AssignProcessToJobObject(handle, int(process._handle))


# Long-lived helper children (the live solver worker) die with this process too.
assign_to_job = _assign_to_job


def _ffmpeg(*args, timeout=25.0):
    return subprocess.run([executable("ffmpeg"), "-hide_banner", *args],
                          capture_output=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=CREATE_NO_WINDOW)


def _lines(done):
    return (done.stderr or "").splitlines() + (done.stdout or "").splitlines()


def devices(timeout=25.0):
    """DirectShow video devices, in the order ffmpeg lists them."""
    try:
        done = _ffmpeg("-list_devices", "true", "-f", "dshow", "-i", "dummy", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in _lines(done):
        match = re.search(r'"([^"]+)"\s+\(video\)', line)
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def formats(device, timeout=25.0):
    """Capture formats a device offers: [{width, height, fps}], largest first."""
    try:
        done = _ffmpeg("-list_options", "true", "-f", "dshow", "-i", "video=" + device,
                       timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    seen = {}
    for line in _lines(done):
        match = re.search(r"max s=(\d+)x(\d+) fps=([\d.]+)", line)
        if not match:
            continue
        width, height, fps = int(match.group(1)), int(match.group(2)), float(match.group(3))
        if width <= 0 or height <= 0 or fps <= 0:
            continue
        seen[(width, height)] = max(seen.get((width, height), 0.0), fps)
    return [{"width": w, "height": h, "fps": round(fps, 3)}
            for (w, h), fps in sorted(seen.items(),
                                      key=lambda item: (-item[0][0] * item[0][1], -item[1]))]


def default_path(root):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(root) / "cache" / "webcam"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / ("webcam-%s.mp4" % stamp)


ROTATE_FILTERS = {90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}


class Session:
    """One camera, one clip, one live preview, all from the same process.

    ``rotation`` turns the capture upright (0, 90 clockwise, 180, 270): a
    webcam stood on its side then feeds portrait frames to the view, the
    solver and the recorded clip alike, and ``width``/``height`` are the
    turned (logical) sizes while ``capture_width``/``capture_height`` stay
    the camera's own.
    """

    def __init__(self, path, device, width, height, fps, preview_width=960, rotation=0):
        self.path = Path(path)
        self.raw_path = self.path.with_suffix(".ts")
        self.device = str(device)
        self.rotation = int(rotation) % 360
        if self.rotation in (90, 270):
            self.width, self.height = int(height), int(width)
        else:
            self.width, self.height = int(width), int(height)
        self.capture_width, self.capture_height = int(width), int(height)
        self.fps = int(fps)
        self.preview_width = min(int(preview_width), self.width)
        self.preview_width -= self.preview_width % 2
        self.preview_height = int(round(self.height * self.preview_width / float(self.width)))
        self.preview_height -= self.preview_height % 2
        self.frame = None
        self.frames = 0
        self.error = ""
        self.started = 0.0
        self._process = None
        self._reader = None
        self._writes_file = True
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _camera_filters(self, scale=None):
        parts = []
        rotate = ROTATE_FILTERS.get(self.rotation)
        if rotate:
            parts.append(rotate)
        if scale:
            parts.append("scale=%d:%d" % scale)
        return ",".join(parts)

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    @property
    def writes_file(self):
        """True once this capture writes its clip (record mode)."""
        return self._writes_file

    def _spawn(self, outputs):
        self._process = subprocess.Popen(outputs, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
        _assign_to_job(self._process)
        self.started = time.monotonic()
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_frames, name="rgbmocap-webcam-preview",
                                        daemon=True)
        self._reader.start()

    def start_preview(self):
        """Open the camera for a live image only - nothing is written to disk.

        The live workspace previews on this: nothing is recorded until the user
        presses Record window, which captures a window of its own.
        """
        self._writes_file = False
        self._spawn([executable("ffmpeg"), "-hide_banner", "-loglevel", "error",
                     "-rtbufsize", "1024M",
                     "-f", "dshow",
                     "-video_size", "%dx%d" % (self.capture_width, self.capture_height),
                     "-framerate", str(self.fps), "-i", "video=" + self.device,
                     "-vf", self._camera_filters((self.preview_width, self.preview_height)),
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])

    def start(self, seconds=None):
        """Start capturing; ``seconds`` makes ffmpeg finish the clip by itself.

        A capture that is killed leaves the muxer's buffer behind (measured: six
        of fifty-nine frames). Given a length, the file output ends on its own
        and every frame is written, so recording always passes one; the Stop
        button still truncates, it just cannot keep the last second.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        clip_filters = self._camera_filters()
        command = [executable("ffmpeg"), "-hide_banner", "-loglevel", "error",
                   "-rtbufsize", "1024M", "-use_wallclock_as_timestamps", "1",
                   "-f", "dshow",
                   "-video_size", "%dx%d" % (self.capture_width, self.capture_height),
                   "-framerate", str(self.fps), "-i", "video=" + self.device,
                   # the clip, turned upright at full resolution
                   "-map", "0:v",
                   *(("-vf", clip_filters) if clip_filters else ()),
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-fps_mode", "vfr",
                   *(["-t", str(float(seconds))] if seconds else []),
                   "-flush_packets", "1", "-f", "mpegts", "-y", str(self.raw_path),
                   # the preview, from the same frames
                   "-map", "0:v", "-vf", self._camera_filters((self.preview_width, self.preview_height)),
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._writes_file = True
        self._spawn(command)

    def _read_frames(self):
        size = self.preview_width * self.preview_height * 3
        stream = self._process.stdout
        pending = b""
        while not self._stop.is_set() and stream is not None:
            # A pipe returns at most what is buffered, so a frame can arrive in
            # pieces. Reading one short chunk as a frame used to end this loop,
            # after which nobody drained the pipe and ffmpeg stalled - the clip
            # held four frames.
            chunk = stream.read(size - len(pending))
            if not chunk:
                break
            pending += chunk
            if len(pending) < size:
                continue
            frame = np.frombuffer(pending[:size], dtype=np.uint8)
            pending = pending[size:]
            with self._lock:
                self.frame = frame.reshape(self.preview_height, self.preview_width, 3)
                self.frames += 1

    def snapshot(self):
        with self._lock:
            return self.frame

    def elapsed(self):
        return max(0.0, time.monotonic() - self.started)

    def stop(self):
        """Stop the capture and finish the clip.

        The camera writes MPEG-TS while it runs: an mp4 needs its index written
        at the end, and a capture stopped on demand has no end - the file was
        left unreadable. Matroska looked like the answer but buffers a cluster,
        so a short clip that was killed held 0 bytes. MPEG-TS flushes packet by
        packet, survives the kill, and one fast remux turns it into the mp4 the
        pipeline expects.
        """
        self._stop.set()
        process, self._process = self._process, None
        if process is None:
            return {"path": str(self.path), "bytes": 0, "frames": self.frames}
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(8.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(5.0)
        error = ""
        try:
            error = (process.stderr.read() or b"").decode("utf-8", "replace").strip()
        except OSError:
            pass
        self._reader = None
        if self._writes_file and self.raw_path.is_file() and self.raw_path.stat().st_size:
            try:
                subprocess.run([executable("ffmpeg"), "-hide_banner", "-loglevel", "error",
                                "-i", str(self.raw_path), "-c", "copy", "-movflags", "+faststart",
                                "-y", str(self.path)], check=True, capture_output=True,
                               creationflags=CREATE_NO_WINDOW)
                self.raw_path.unlink(missing_ok=True)
            except (OSError, subprocess.CalledProcessError) as error:
                self.error = str(error)[:200]
        if not self._writes_file:
            return {"path": None, "bytes": 0, "frames": self.frames,
                    "device": self.device, "width": self.width, "height": self.height,
                    "requested_fps": self.fps, "fps": float(self.fps),
                    "seconds": round(self.elapsed(), 2)}
        if error and not self.path.is_file():
            self.error = error.splitlines()[-1]
        report = {"path": str(self.path), "bytes": self.path.stat().st_size
                  if self.path.is_file() else 0, "frames": self.frames,
                  "device": self.device, "width": self.width, "height": self.height,
                  "requested_fps": self.fps, "fps": float(self.fps), "seconds": round(self.elapsed(), 2)}
        if report["bytes"]:
            try:
                found = probe(self.path)
                report.update(fps=round(float(found.get("fps") or self.fps), 2),
                              frames=int(found.get("frames") or self.frames),
                              seconds=round(float(found.get("duration") or report["seconds"]), 2))
            except Exception:               # noqa: BLE001 - a report, not the clip
                pass
        return report
