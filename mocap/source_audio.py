"""The source video's own sound, played with the preview.

Mocap only needs the picture, so nothing here is analysed: this exists so a
clip can be watched (and its rhythm heard) while the character follows it.
One ffmpeg decodes the soundtrack of the same source the picture comes from -
a local file, or the stream address and headers a link session already holds -
as raw PCM, and a push-mode QAudioSink plays it. The playhead, the preview
speed and a link that plays again are all followed here, and a source without
a soundtrack simply writes nothing.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque

from PySide6.QtCore import QObject, QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from .video import executable

CREATE_NO_WINDOW = 0x08000000
SAMPLE_RATE = 48000
CHANNELS = 2
PUSH_SECONDS = 0.043      # audio handed to the device per timer tick


def _tempo(speed):
    """An atempo chain for any preview speed (one filter covers 0.5x-2x)."""
    value = float(speed or 1.0)
    if value <= 0 or abs(value - 1.0) < 1e-3:
        return ''
    steps = []
    while value > 2.0:
        steps.append('atempo=2.0')
        value /= 2.0
    while value < 0.5:
        steps.append('atempo=0.5')
        value /= 0.5
    steps.append('atempo=%.4f' % value)
    return ','.join(steps)


class SourceAudio(QObject):
    """Plays one source's soundtrack; starting another replaces it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sink = None
        self._device = None
        self._process = None
        self._reader = None
        self._buffer = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._command = None
        self._loop = False
        self._volume = 1.0
        # Filled in by the sink: the device decides the PCM it can play, and
        # ffmpeg is asked for exactly that.
        self._rate, self._channels = SAMPLE_RATE, CHANNELS
        self._codec, self._sample_bytes = 's16le', 2
        self._chunk = 8192
        # Reading ahead is bounded: one second of audio is enough to ride out
        # a network stall, and ffmpeg waits behind the pipe instead of the
        # buffer growing for as long as a clip lasts.
        self._limit = SAMPLE_RATE * CHANNELS * 2
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._pump)

    # -- controls --------------------------------------------------------
    @property
    def playing(self):
        return self._timer.isActive()

    def play(self, source, start=0.0, speed=1.0, loop=False):
        """Play ``source`` from ``start`` seconds; returns False when it cannot."""
        self.stop()
        # The device is opened first: its format decides what ffmpeg produces.
        if not self._open_sink():
            return False
        command = self._command_for(source, start, speed)
        if command is None:
            self.stop()
            return False
        self._command = command
        self._loop = bool(loop)
        self._stop = threading.Event()
        if not self._spawn():
            self.stop()
            return False
        self._timer.start()
        return True

    def set_volume(self, value):
        self._volume = min(1.0, max(0.0, float(value)))
        if self._sink is not None:
            self._sink.setVolume(self._volume)

    def stop(self):
        self._timer.stop()
        self._stop.set()
        sink, self._sink = self._sink, None
        if sink is not None:
            sink.stop()
            sink.deleteLater()
        self._device = None
        process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(3.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(3.0)
        with self._lock:
            self._buffer.clear()
        self._command = None
        self._loop = False

    # -- pipeline --------------------------------------------------------
    def _command_for(self, source, start, speed):
        try:
            ffmpeg = executable('ffmpeg')
        except RuntimeError:
            return None
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin']
        if isinstance(source, dict):
            address = str(source.get('address') or '')
            headers = ''.join('%s: %s\r\n' % (key, value)
                              for key, value in (source.get('headers') or {}).items())
            if headers:
                command += ['-headers', headers]
        else:
            address = str(source or '')
        if not address:
            return None
        if start > 0:
            command += ['-ss', '%.3f' % start]
        command += ['-i', address, '-vn', '-ac', str(self._channels), '-ar', str(self._rate)]
        tempo = _tempo(speed)
        if tempo:
            command += ['-af', tempo]
        command += ['-f', self._codec, 'pipe:1']
        return command

    def _open_sink(self):
        device = QMediaDevices.defaultAudioOutput()
        if device is None or device.isNull():
            return False
        wanted = QAudioFormat()
        wanted.setSampleRate(SAMPLE_RATE)
        wanted.setChannelCount(CHANNELS)
        wanted.setSampleFormat(QAudioFormat.Int16)
        # Devices vary: many offer only float samples. Take the device's own
        # format rather than failing, and ask ffmpeg for that sample type.
        audio_format = wanted if device.isFormatSupported(wanted) else device.preferredFormat()
        sample_format = audio_format.sampleFormat()
        if sample_format == QAudioFormat.Int16:
            self._codec, self._sample_bytes = 's16le', 2
        elif sample_format == QAudioFormat.Float:
            self._codec, self._sample_bytes = 'f32le', 4
        else:
            return False
        self._rate = audio_format.sampleRate() or SAMPLE_RATE
        self._channels = audio_format.channelCount() or CHANNELS
        frame = max(1, self._sample_bytes * self._channels)
        self._chunk = max(frame, int(self._rate * PUSH_SECONDS) * frame)
        self._limit = self._rate * frame
        sink = QAudioSink(device, audio_format, self)
        sink.setVolume(self._volume)
        device_io = sink.start()
        if device_io is None:
            sink.deleteLater()
            return False
        self._sink, self._device = sink, device_io
        return True

    def _spawn(self):
        """Open one ffmpeg pass and start reading it on a thread."""
        command = self._command
        if command is None or self._stop.is_set():
            return False
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       creationflags=CREATE_NO_WINDOW)
        except OSError:
            return False
        self._process = process
        thread = threading.Thread(target=self._read, args=(process,), name='rgbmocap-audio',
                                  daemon=True)
        self._reader = thread
        thread.start()
        return True

    def _wait_for_room(self):
        """Let ffmpeg wait while the device is behind, instead of reading ahead."""
        while not self._stop.is_set():
            with self._lock:
                buffered = sum(len(chunk) for chunk in self._buffer)
            if buffered < self._limit:
                return
            time.sleep(0.02)

    def _read(self, process):
        try:
            while not self._stop.is_set():
                self._wait_for_room()
                if self._stop.is_set():
                    break
                data = process.stdout.read(self._chunk)
                if not data:
                    break
                with self._lock:
                    self._buffer.append(data)
        except (OSError, ValueError):
            pass
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass
            if self._loop and not self._stop.is_set():
                # A link preview plays its clip again; the sound does too, from
                # the same second, rather than falling silent after one pass.
                self._spawn()

    def _pump(self):
        sink, device_io = self._sink, self._device
        if sink is None or device_io is None:
            self.stop()
            return
        free = sink.bytesFree()
        while free >= self._chunk:
            with self._lock:
                data = self._buffer.popleft() if self._buffer else None
            if data is None:
                break
            device_io.write(data)
            free -= len(data)
        # A source without a soundtrack, or one that has played to its end,
        # leaves nothing to push: stop instead of holding the audio device.
        with self._lock:
            buffered = bool(self._buffer)
        reader = self._reader
        if not buffered and (reader is None or not reader.is_alive()) and not self._loop:
            self.stop()
