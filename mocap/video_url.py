"""Open a video from a web address instead of picking a local file.

A link may point straight at a video file or at a page that only plays one in a
browser, such as an X post. The live preview resolves the page to its real
media address and plays it without saving anything. A stage that reads the clip
more than once (reconstruction) needs a local ``input.mp4``, so the workspace's
Download button fetches it into the downloads folder on request.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout)

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.ts'}
STREAM_SUFFIXES = {'.m3u8', '.mpd'}
CONTENT_SUFFIXES = {'video/mp4': '.mp4', 'video/webm': '.webm', 'video/quicktime': '.mov',
                    'video/x-matroska': '.mkv', 'video/mp2t': '.ts', 'video/mpeg': '.mpg'}
X_HOSTS = {'x.com', 'www.x.com', 'twitter.com', 'www.twitter.com', 'mobile.twitter.com'}
SYNDICATION = 'https://cdn.syndication.twimg.com/tweet-result'
MAX_BYTES = 4 * 1024 ** 3
MAX_PAGE = 4 * 1024 ** 2
CHUNK = 1 << 20
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RGBMoCap/1.0'


class FetchCancelled(Exception):
    pass


def is_url(text):
    return urllib.parse.urlsplit(text.strip()).scheme.lower() in ('http', 'https')


def _text(value):
    return html.unescape(value.replace('\\/', '/'))


def _fetch(url, accept='text/html,*/*'):
    request = urllib.request.Request(url.strip(), headers={'User-Agent': USER_AGENT, 'Accept': accept})
    try:
        return urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        raise ValueError(f'The link answered HTTP {error.code}. Check that it opens without signing in.')
    except urllib.error.URLError as error:
        raise ValueError(f'Cannot reach the link: {error.reason}')


def _meta(page, *names):
    """Content of the first <meta> whose property or name is one of ``names``."""
    for tag in re.findall(r'<meta\b[^>]*>', page, re.I):
        key = re.search(r'(?:property|name)\s*=\s*["\']([^"\']+)', tag, re.I)
        if not key or key.group(1).strip().lower() not in names:
            continue
        content = re.search(r'content\s*=\s*["\']([^"\']+)', tag, re.I)
        if content:
            return _text(content.group(1))
    return ''


def _best_variant(block):
    """Highest-quality mp4, or the stream playlist when that is all there is."""
    best, playlist = ('', -1), ''
    for item in re.findall(r'\{[^{}]*\}', block):
        address = re.search(r'"url"\s*:\s*"([^"]+)"', item)
        if not address:
            continue
        kind = (re.search(r'"content_type"\s*:\s*"([^"]+)"', item) or [None, ''])[1]
        rate = re.search(r'"bitrate"\s*:\s*(\d+)', item)
        address = _text(address.group(1))
        if 'mp4' not in kind:
            playlist = playlist or address
            continue
        score = int(rate.group(1)) if rate else 1
        if score > best[1]:
            best = (address, score)
    return best[0] or playlist


def _variants(page):
    found = [_best_variant(block) for block in re.findall(r'"variants"\s*:\s*\[(.*?)\]', page, re.S)]
    found = [address for address in found if address]
    return next((address for address in found if '.mp4' in address.split('?')[0].lower()),
                found[0] if found else '')


def _x_media(url):
    """An X post hides its video: ask the public syndication endpoint for it."""
    path = urllib.parse.urlsplit(url).path
    status = re.search(r'/status(?:es)?/(\d+)', path)
    if not status:
        return ''
    with _fetch(f'{SYNDICATION}?id={status.group(1)}&lang=en&token=a', 'application/json') as response:
        endpoint, raw = response.geturl(), response.read(MAX_PAGE)
    try:
        data = json.loads(raw)
    except ValueError:
        return ''
    found = ''
    for media in data.get('mediaDetails') or []:
        block = json.dumps((media.get('video_info') or {}).get('variants') or [])
        found = found or _best_variant(block)
    return urllib.parse.urljoin(endpoint, found) if found else ''


def _x_name(url):
    path = urllib.parse.urlsplit(url).path
    status = re.search(r'/status(?:es)?/(\d+)', path)
    if not status:
        return ''
    handle = path.split('/status')[0].strip('/').split('/')[-1] if '/status' in path else ''
    handle = re.sub(r'[^0-9A-Za-z_.-]+', '_', handle)
    return f'{handle}_{status.group(1)}' if handle and handle not in ('i', 'web') else status.group(1)


def direct_media(url):
    """Return ``(media address, file-name hint)`` for a file or page link."""
    url = url.strip()
    path = urllib.parse.urlsplit(url).path
    if Path(path).suffix.lower() in VIDEO_SUFFIXES | STREAM_SUFFIXES:
        return url, ''
    host = (urllib.parse.urlsplit(url).hostname or '').lower()
    if host in X_HOSTS:
        found = _x_media(url)
        if found:
            return found, _x_name(url)
    with _fetch(url) as response:
        kind = (response.headers.get('Content-Type') or '').lower()
        final = response.geturl()
        if kind.startswith('video/'):
            return final, ''
        if not kind.startswith(('text/', 'application/xhtml', 'application/json')):
            return final, ''
        page = response.read(MAX_PAGE).decode('utf-8', 'replace')
    candidates = [_variants(page),
                  _meta(page, 'og:video:secure_url', 'og:video:url', 'og:video', 'twitter:player:stream')]
    candidates += re.findall(r'<(?:video|source)\b[^>]*\bsrc\s*=\s*["\']([^"\']+)', page, re.I)
    for candidate in candidates:
        if candidate:
            return urllib.parse.urljoin(final, _text(candidate)), ''
    raise ValueError('This link has no video the app can read. Open the post, use the video address '
                     '(often ending in .mp4), or save the file and use Open Video.')


def file_name(url, content_type=''):
    """A safe file name, with a video extension the analysis can trust."""
    path = urllib.parse.unquote(urllib.parse.urlsplit(url.strip()).path)
    path = '' if path.endswith('/') else path
    stem = re.sub(r'[^0-9A-Za-z _.-]+', '_', Path(path).stem).strip(' ._') or 'video'
    suffix = Path(path).suffix.lower()
    if suffix in STREAM_SUFFIXES:
        suffix = '.mp4'   # the playlist is remuxed into one file
    elif suffix not in VIDEO_SUFFIXES:
        suffix = CONTENT_SUFFIXES.get(content_type.split(';')[0].strip().lower(), '.mp4')
    return stem[:80] + suffix


def free_path(path):
    """Never overwrite an earlier download of the same name."""
    path = Path(path)
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f'{path.stem} ({index}){path.suffix}')
        if not candidate.exists():
            return candidate
    raise ValueError('Too many files with the same name in the downloads folder.')


def _checked(path):
    from .video import probe
    try:
        probe(path)
    except Exception as error:
        Path(path).unlink(missing_ok=True)
        raise ValueError('The download is not a video this app can read: ' + str(error))
    return Path(path)


def _fetch_file(url, target, progress, cancelled):
    response = _fetch(url, 'video/*,*/*')
    with response:
        content_type = (response.headers.get('Content-Type') or '').lower()
        length = int(response.headers.get('Content-Length') or 0)
        if content_type.startswith(('text/', 'application/json', 'application/xhtml')):
            raise ValueError('This link is a web page, not a video file. Use the direct video link '
                             '(often ending in .mp4), or download the file and use Open Video.')
        if length > MAX_BYTES:
            raise ValueError(f'This video is {length / 1024 ** 3:.1f} GB. Download it and use Open Video.')
        temporary = target.with_name('.' + target.name + '.partial')
        done = 0
        try:
            with open(temporary, 'wb') as file:
                while True:
                    if cancelled is not None and cancelled():
                        raise FetchCancelled('Download cancelled.')
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    file.write(chunk)
                    done += len(chunk)
                    if done > MAX_BYTES:
                        raise ValueError('The video is larger than 4 GB. Download it and use Open Video.')
                    if progress is not None:
                        progress(done, length)
            if not done:
                raise ValueError('The link returned an empty file.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _fetch_stream(url, target, progress, cancelled):
    """Remux an HLS/DASH playlist into one analysis-ready mp4 with the bundled ffmpeg."""
    from .video import executable
    temporary = target.with_name('.' + target.name + '.partial.mp4')
    if progress is not None:
        progress(0, 0)
    try:
        process = subprocess.Popen([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                                    '-i', url, '-map', '0:v:0', '-c', 'copy', str(temporary)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, creationflags=0x08000000)
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    process.kill()
                    raise FetchCancelled('Download cancelled.')
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.kill()
            errors = (process.stderr.read() if process.stderr else b'').decode('utf-8', 'replace')
            if process.stderr is not None:
                process.stderr.close()
            process.wait()
        if process.returncode:
            raise ValueError('Cannot read this stream link: ' + (errors.strip()[-400:] or 'ffmpeg failed'))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _safe(text):
    return re.sub(r'[^0-9A-Za-z _.-]+', '_', text).strip(' ._') or 'video'


def _save_media(media, folder, progress=None, cancelled=None, hint=''):
    name = file_name(media)
    if hint:
        name = _safe(hint)[:80] + Path(name).suffix
    target = free_path(Path(folder) / name)
    if Path(urllib.parse.urlsplit(media).path).suffix.lower() in STREAM_SUFFIXES:
        return _checked(_fetch_stream(media, target, progress, cancelled))
    return _checked(_fetch_file(media, target, progress, cancelled))


def ytdlp():
    """The optional extractor for sites that never expose a plain video address."""
    try:
        import yt_dlp
    except Exception:
        return None
    return yt_dlp


class _Silent:
    """Keep the extractor's console chatter out of the app log."""

    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def _extract(url, folder, progress=None, cancelled=None):
    """Download with yt-dlp, which handles YouTube, TikTok and similar players."""
    from .video import executable
    yt_dlp = ytdlp()
    if yt_dlp is None:
        raise ValueError('This site needs the yt-dlp extractor, which is missing from this installation.')
    folder = Path(folder)

    def hook(data):
        if cancelled is not None and cancelled():
            raise yt_dlp.utils.DownloadCancelled('Download cancelled.')
        if data.get('status') == 'downloading' and progress is not None:
            total = data.get('total_bytes') or data.get('total_bytes_estimate') or 0
            progress(int(data.get('downloaded_bytes') or 0), int(total))

    options = {
        'format': 'bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b',
        'merge_output_format': 'mp4',
        'outtmpl': {'default': str(folder / '%(title).80B [%(id)s].%(ext)s')},
        'noplaylist': True, 'quiet': True, 'no_warnings': True, 'noprogress': True,
        'windowsfilenames': True, 'retries': 3, 'fragment_retries': 3, 'overwrites': False,
        'logger': _Silent(), 'progress_hooks': [hook],
        'ffmpeg_location': str(Path(executable('ffmpeg')).parent),
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except Exception as error:
        if cancelled is not None and cancelled():
            raise FetchCancelled('Download cancelled.')
        raise ValueError('Cannot download this link: '
                         + re.sub(r'^ERROR:\s*', '', str(error).strip())[:300])
    produced = [Path(entry['filepath']) for entry in (info.get('requested_downloads') or [])
                if entry.get('filepath') and Path(entry['filepath']).is_file()]
    if not produced:
        raise ValueError('The extractor finished without a video file.')
    return _checked(max(produced, key=lambda path: path.stat().st_mtime))


def download(url, folder, progress=None, cancelled=None):
    """Fetch one video link (or the video on one page link) and return the local file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = urllib.parse.urlsplit(url.strip()).path
    if Path(path).suffix.lower() in VIDEO_SUFFIXES | STREAM_SUFFIXES:
        return _save_media(url, folder, progress, cancelled)
    problems = []
    host = (urllib.parse.urlsplit(url).hostname or '').lower()
    if host in X_HOSTS:
        try:
            media, hint = direct_media(url)
        except ValueError as error:
            problems.append(str(error))
        else:
            return _save_media(media, folder, progress, cancelled, hint)
    if ytdlp() is not None:
        try:
            return _extract(url, folder, progress, cancelled)
        except FetchCancelled:
            raise
        except ValueError as error:
            problems.append(str(error))
    try:
        media, hint = direct_media(url)
    except ValueError as error:
        problems.append(str(error))
    else:
        return _save_media(media, folder, progress, cancelled, hint)
    raise ValueError(problems[-1])


class FetchVideo(QThread):
    """Download off the GUI thread so the window keeps repainting."""

    progress = Signal(int, int)
    fetched = Signal(object)
    failed = Signal(str)

    def __init__(self, url, folder, parent=None):
        super().__init__(parent)
        self.url, self.folder, self.stopping = url, Path(folder), False

    def run(self):
        try:
            path = download(self.url, self.folder, self.progress.emit, lambda: self.stopping)
        except Exception as error:
            if not self.stopping:
                self.failed.emit(str(error))
            return
        self.fetched.emit(path)


class VideoUrlDialog(QDialog):
    """One link, then the live preview. Keeping the clip is a separate step.

    Preview plays the link as the live source and downloads nothing, which is
    what Find person needs. Full reconstruction re-reads a clip several times,
    so it still needs the file: the Download button beside the video title
    saves the previewed link once, on request.
    """

    def __init__(self, parent, url=''):
        super().__init__(parent)
        self.setWindowTitle('Open Video from URL')
        self.setMinimumWidth(480)
        self.path = None
        self.url = ''
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Video URL'))
        self.field = QLineEdit(url)
        self.field.setPlaceholderText('https://x.com/.../status/... or https://example.com/clip.mp4')
        self.field.setClearButtonEnabled(True)
        layout.addWidget(self.field)
        note = QLabel('Paste a video link, a YouTube or TikTok link, a post link that contains a '
                      'video, or a file path. Preview plays it as the live source without '
                      'downloading it; press Find person to move the character with the video. '
                      'Reconstruct 3D needs the file, so press Download next to the video title '
                      'first.')
        note.setWordWrap(True)
        note.setObjectName('hint')
        layout.addWidget(note)
        self.state = QLabel('')
        self.state.setWordWrap(True)
        layout.addWidget(self.state)
        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel = QPushButton('Cancel')
        self.cancel.clicked.connect(self.reject)
        actions.addWidget(self.cancel)
        self.start = QPushButton('Preview')
        self.start.setDefault(True)
        self.start.setToolTip('Play this link in the live view without saving it. Download beside '
                              'the video title keeps it for full reconstruction.')
        self.start.clicked.connect(self.begin)
        actions.addWidget(self.start)
        layout.addLayout(actions)
        self.field.textChanged.connect(lambda: self.state.setText(''))
        self.field.returnPressed.connect(self.begin)

    def begin(self):
        text = self.field.text().strip()
        if not text:
            self.state.setText('Enter a video URL.')
            return
        if not is_url(text):
            local = Path(text).expanduser()
            if local.is_file():
                self.path = local
                self.accept()
                return
            self.state.setText('Enter an http:// or https:// video link, or an existing file path.')
            return
        self.path = None
        self.url = text
        self.accept()

    def reject(self):
        super().reject()
