"""A web video as the live source instead of a camera.

The live character only ever needs the newest frame, in order: the solver reads
every frame once and never seeks back. That is exactly what a network stream can
do, so a link can drive the live preview without downloading the video first -
nothing is written to disk unless the user records a window.

Sites that keep their media address behind a player (YouTube, TikTok and
similar) are resolved by the optional yt-dlp extractor in metadata-only mode;
the stream address it hands back, and the headers it needs, go straight to
ffmpeg. A page that exposes `og:video` (an X post) is resolved by the same
helpers the Open Video from URL dialog uses.

Full reconstruction is a different matter: it re-reads the whole clip several
times (tracking, features, inference, preview) and needs a local file, so that
path still downloads once. Only this live preview is stream-only.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.parse
from fractions import Fraction
from pathlib import Path

import numpy as np

from .video import executable
from .video_url import (STREAM_SUFFIXES, VIDEO_SUFFIXES, _Silent, direct_media, is_url,
                        ytdlp)

CREATE_NO_WINDOW = 0x08000000
MAX_PREVIEW = 960          # longest preview edge handed to the view and solver
CONNECT_TIMEOUT = 45.0     # seconds ffmpeg may spend opening the stream


class UrlSession:
    """One web video presented like ``camera_capture.Session``.

    The view and the live workers only use ``width``/``height``,
    ``preview_width``/``preview_height``, ``running``, ``snapshot()`` and
    ``stop()``, so a stream slots in where the camera was.
    """

    writes_file = False
    rotation = 0

    def __init__(self, url, preview_width=MAX_PREVIEW):
        self.url = url.strip()
        if not is_url(self.url):
            raise ValueError('Enter an http:// or https:// video link.')
        self.source = resolve(self.url)
        self.address = self.source['address']
        self.headers = self.source['headers']
        # The soundtrack is its own stream when the site serves video-only
        # formats; the picture above stays exactly as it was.
        self.audio_address = self.source.get('audio_address') or self.address
        self.audio_headers = self.source.get('audio_headers') or self.headers
        self.title = self.source['title']
        size = (self.source.get('width'), self.source.get('height'))
        duration = float(self.source.get('duration') or 0.0)
        if not (size[0] and size[1]) or not duration:
            width, height, length = _probe_media(self.address, self.headers)
            size = size if size[0] and size[1] else (width, height)
            duration = duration or length
        if not (size[0] and size[1]):
            raise ValueError('This link has no video the app can read. Open the post and use its '
                             'direct video address (often ending in .mp4), or save the file and '
                             'use Open Video.')
        self.width, self.height = int(size[0]), int(size[1])
        self.device = urllib.parse.urlsplit(self.url).hostname or 'link'
        self.label = 'Live link · ' + (self.title or self.device)
        self.fps = float(self.source.get('fps') or 30.0)
        # Fit the preview inside MAX_PREVIEW, both axes, keeping the picture
        # upright (a portrait clip must not be squashed into a wide frame).
        scale = min(float(preview_width) / self.width, float(MAX_PREVIEW) / self.height, 1.0)
        self.preview_width = max(2, int(round(self.width * scale)) // 2 * 2)
        self.preview_height = max(2, int(round(self.height * scale)) // 2 * 2)
        self.frame = None
        self.frames = 0
        self.error = ''
        self.started = 0.0
        # Length of the clip in seconds, or 0 for a stream that has no end.
        # The position control only has meaning with a known length.
        self.duration = max(0.0, duration)
        # Where this pass of the stream began, so the playhead can be reported
        # as clip time rather than as time since the process opened.
        self._start_at = 0.0
        self._seek_offset = None
        # Set once the stream ends for good; a pass that is reopening is not
        # finished, so a looping clip never looks ended under a live label.
        self.finished = False
        # Where the playhead stands while the clip is parked on its last frame.
        self._frozen = None
        # A clip is not a camera: it ends. Playing it again keeps frames
        # arriving while the character's model is still loading, so the live
        # preview follows moving video instead of one frozen last frame.
        self.loop = True
        self._process = None
        self._reader = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._restarting = False

    # -- capture ---------------------------------------------------------
    def _command(self):
        headers = ''.join('%s: %s\r\n' % (key, value)
                          for key, value in self.headers.items())
        return [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error',
                # -re paces the link like a camera: read at its own rate instead
                # of pulling the whole file as fast as the socket allows.
                '-re', '-rw_timeout', str(int(CONNECT_TIMEOUT * 1e6)),
                # The playhead: reopening the stream at the chosen second is how
                # a stream is seeked - it cannot be scrubbed in place.
                *(['-ss', '%.3f' % self._start_at] if self._start_at > 0 else ()),
                *(['-headers', headers] if headers else ()),
                '-i', self.address, '-an',
                '-vf', 'scale=%d:%d,fps=%g' % (self.preview_width, self.preview_height,
                                                min(self.fps, 30.0)),
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']

    @property
    def running(self):
        if self._restarting:
            return not self._stop.is_set()
        return self._process is not None and self._process.poll() is None

    @property
    def ended(self):
        """The stream finished on its own (a clip ended, not a stop)."""
        return self.finished

    def start_preview(self, seconds=0.0):
        """Open the link read-only: nothing is written to disk."""
        if self.running:
            return
        self._stop.clear()
        self._seek_offset = None
        self._start_at = max(0.0, float(seconds))
        self.finished = False
        self._frozen = None
        self._process = subprocess.Popen(self._command(), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=CREATE_NO_WINDOW)
        self.started = time.monotonic()
        self._reader = threading.Thread(target=self._read_frames, name='rgbmocap-link-preview',
                                        daemon=True)
        self._reader.start()

    def _read_frames(self):
        try:
            self._read_passes()
        finally:
            # The reader only returns when the stream is done with, so a link
            # that failed is reported ended rather than silently frozen.
            if not self._stop.is_set():
                self.finished = True

    def _read_passes(self):
        size = self.preview_width * self.preview_height * 3
        while not self._stop.is_set():
            process = self._process
            stream = process.stdout if process is not None else None
            if stream is None:
                break
            pending = b''
            while not self._stop.is_set():
                try:
                    chunk = stream.read(size - len(pending))
                except (OSError, ValueError):
                    # The stream was closed under this thread by a stop.
                    chunk = b''
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
                # The view asks for another part of the clip: stop this pass at
                # a frame boundary and reopen the stream at that point, rather
                # than closing the pipe under the read below.
                if self._seek_offset is not None:
                    break
            # A seek cut this pass short while the stream was still playing, so
            # its stderr must not be read: that read waits for the end of a
            # process that has not ended, and the pipe would fill behind it.
            seek, self._seek_offset = self._seek_offset, None
            if seek is not None and not self._stop.is_set():
                try:
                    self.restart(seek)
                except OSError:
                    break
                continue
            error = ''
            try:
                if process.stderr is not None:
                    error = (process.stderr.read() or b'').decode('utf-8', 'replace').strip()
            except OSError:
                pass
            if error and not self._stop.is_set():
                self.error = error.splitlines()[-1][:300]
            if self._stop.is_set():
                break
            if seek is None and (not self.loop or self.frames == 0):
                if self.frames == 0:
                    # Nothing ever arrived; another pass would fail the same way.
                    break
                # Paused on the last frame. Stay ready on this one reader: a new
                # position, or Loop turned on, reopens the stream from here.
                self._frozen = self.position()
                self.finished = True
                while not self._stop.is_set() and self._seek_offset is None and not self.loop:
                    time.sleep(0.05)
                self.finished = False
                if self._stop.is_set():
                    break
                seek, self._seek_offset = self._seek_offset, None
            try:
                self.restart(seek)
            except OSError:
                break

    def restart(self, seconds=None):
        """Reopen the stream, from its start or from the given clip second."""
        if self._stop.is_set():
            return
        self._restarting = True
        old, self._process = self._process, None
        _close_process(old)
        if self._stop.is_set():
            self._restarting = False
            return
        if seconds is not None:
            self._start_at = max(0.0, float(seconds))
        self.finished = False
        self._frozen = None
        self.error = ''
        self._process = subprocess.Popen(self._command(), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=CREATE_NO_WINDOW)
        self.started = time.monotonic()
        self._restarting = False

    # -- playback --------------------------------------------------------
    def position(self):
        """Seconds into the clip the stream is showing (0 when unknown)."""
        if self._seek_offset is not None:
            # A seek is in flight: show the requested second rather than the
            # tail of the pass that is being closed.
            return self._seek_offset
        if self._frozen is not None:
            # Parked on the last frame: the playhead stays there instead of
            # creeping past the clip's end as the clock runs on.
            return self._frozen
        if self.started <= 0:
            return 0.0
        shown = self._start_at + max(0.0, time.monotonic() - self.started)
        return min(shown, self.duration) if self.duration else shown

    def report(self):
        """What the playback controls show: playhead, length and loop."""
        return {'position': self.position(), 'duration': self.duration, 'loop': self.loop,
                'running': self.running, 'finished': self.finished}

    def seek(self, seconds):
        """Move the playhead: the stream reopens at that second of the clip.

        A network video cannot be scrubbed in place, so seeking is a restart at
        the new offset. Reopening takes a moment; frames resume by themselves.
        """
        if not self.duration:
            return False
        seconds = min(max(0.0, float(seconds)), max(0.0, self.duration - 0.05))
        reader = self._reader
        if reader is None or not reader.is_alive():
            # The reader is gone, so no one would consume the new offset.
            self.start_preview(seconds)
            return True
        # The reader reopens the stream at this second at the next frame, which
        # also resumes a clip that is parked on its last frame.
        self._seek_offset = seconds
        return True

    def set_loop(self, enabled):
        """Play the clip again when it ends, or stop on its last frame.

        Turning loop back on also resumes a clip that already ran to its end,
        because the reader waits there for exactly this.
        """
        self.loop = bool(enabled)
        return self.loop

    def snapshot(self):
        with self._lock:
            return self.frame

    def elapsed(self):
        return max(0.0, time.monotonic() - self.started)

    def stop(self):
        """Close the stream. Nothing was written, so there is no clip to finish."""
        self._stop.set()
        process, self._process = self._process, None
        _close_process(process)
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(8.0)
        if process is None:
            return {'path': None, 'bytes': 0, 'frames': self.frames}
        return {'path': None, 'bytes': 0, 'frames': self.frames, 'device': self.device,
                'width': self.width, 'height': self.height, 'requested_fps': self.fps,
                'fps': float(self.fps), 'seconds': round(self.elapsed(), 2)}


def _close_process(process):
    if process is None:
        return
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
    for stream in (process.stdout, process.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


def _headers_argument(headers):
    return ''.join('%s: %s\r\n' % (key, value) for key, value in headers.items())


def _probe_media(address, headers):
    """(width, height, duration) of a remotely decodable video.

    Duration is what makes the position control meaningful: a clip knows where
    its end is, a live stream does not and reports 0.
    """
    command = [executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height:format=duration', '-of', 'json']
    headers = _headers_argument(headers)
    if headers:
        command += ['-headers', headers]
    command += ['-i', address]
    try:
        done = subprocess.run(command, capture_output=True, encoding='utf-8', timeout=60,
                              creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return (0, 0)
    if done.returncode:
        return (0, 0)
    try:
        info = json.loads(done.stdout or '{}')
        streams = info.get('streams') or []
        stream = streams[0] if streams else {}
        duration = float((info.get('format') or {}).get('duration') or 0.0)
        return (int(stream.get('width') or 0), int(stream.get('height') or 0),
                0.0 if duration != duration or duration < 0 else duration)
    except (ValueError, TypeError, IndexError):
        return (0, 0, 0.0)


def _rate(value):
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def _from_extractor(url):
    """Ask yt-dlp for a playable stream address, without downloading."""
    vtdlp = ytdlp()
    if vtdlp is None:
        return None
    options = {'quiet': True, 'no_warnings': True, 'noplaylist': True, 'noprogress': True,
               'skip_download': True, 'logger': _Silent(),
               'format': 'bv*[height<=1080]/bv*/b[height<=1080]/b'}
    try:
        with vtdlp.YoutubeDL(options) as extractor:
            info = extractor.extract_info(url, download=False)
            # Sites such as TikTok hand the stream out only to the session that
            # solved their challenge: their media server answers 403 without the
            # cookies the extractor collected, so they travel with the address.
            cookies = '; '.join('%s=%s' % (cookie.name, cookie.value)
                                for cookie in extractor.cookiejar)
    except Exception:                      # noqa: BLE001 - another resolver may work
        return None
    if info.get('entries'):
        info = next((entry for entry in info['entries'] if entry), info)
    formats = info.get('formats') or []
    playable = [entry for entry in formats
                if entry.get('url') and entry.get('vcodec') not in (None, 'none')]
    if not playable:
        return None
    # Prefer H.264 (every decoder here reads it), then the tallest picture that
    # is not heavier than a live preview needs.
    def score(entry):
        codec = str(entry.get('vcodec') or '')
        height = int(entry.get('height') or 0)
        return (codec.startswith('avc1'), 1080 >= height, height)
    chosen = max(playable, key=score)
    headers = dict(info.get('http_headers') or {})
    headers.update(chosen.get('http_headers') or {})
    # The picture is taken from a video-only format whenever the site offers
    # one, so the soundtrack has its own address: the smallest audio-only
    # format, or the muxed format that carries both.
    audible = [entry for entry in formats
               if entry.get('url') and entry.get('acodec') not in (None, 'none')]
    sound = next((entry for entry in audible if entry.get('vcodec') in (None, 'none')),
                 audible[0] if audible else None)
    audio_headers = dict(info.get('http_headers') or {})
    if sound is not None:
        audio_headers.update(sound.get('http_headers') or {})
    if cookies:
        headers['Cookie'] = cookies
        audio_headers['Cookie'] = cookies
    # ffmpeg needs a plain Unix-style newline separated block; yt-dlp's own
    # headers already carry the referer and cookie some CDNs insist on.
    return dict(address=chosen['url'], headers=headers, title=info.get('title') or '',
                audio_address=(sound or chosen)['url'], audio_headers=audio_headers,
                width=int(chosen.get('width') or info.get('width') or 0),
                height=int(chosen.get('height') or info.get('height') or 0),
                duration=float(info.get('duration') or 0.0),
                fps=_rate(chosen.get('fps') or info.get('fps')) or 30.0)


def resolve(url):
    """A playable stream address for a link: {'address', 'headers', ...}.

    A direct file or playlist address is used as it is; a page is read for its
    media address; a player-only site goes through the extractor.
    """
    url = url.strip()
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    if suffix in VIDEO_SUFFIXES | STREAM_SUFFIXES:
        return dict(address=url, headers={}, title='', width=0, height=0, duration=0.0, fps=0.0,
                    audio_address=url, audio_headers={})
    extracted = _from_extractor(url)
    if extracted is not None:
        return extracted
    try:
        media, hint = direct_media(url)
    except ValueError as error:
        raise ValueError(str(error)) from error
    return dict(address=media, headers={}, title=hint, width=0, height=0, duration=0.0,
                fps=0.0, audio_address=media, audio_headers={})


def suggest_title(url):
    """A short label for a link, used in the window text."""
    host = urllib.parse.urlsplit(url).hostname or ''
    return re.sub(r'^www\.', '', host)
