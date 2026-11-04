"""Check File > Open Video from URL...: page links, downloads, rejection, dialog.

Serves a real clip, an HLS playlist and several page shapes from a loopback HTTP
server, so page resolution and streaming run without touching the internet or
the user's downloads folder.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(tempfile.mkdtemp(prefix='rgbmocap-url-'))
os.environ['RGB_MOCAP_DATA'] = str(DATA)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(ROOT))
GUI = ROOT / 'runtime_gui'   # PySide6 lives in the GUI overlay of the shared runtime
if GUI.is_dir():
    sys.path.insert(0, str(GUI))

from mocap import video_url  # noqa: E402
from mocap.video import executable, probe  # noqa: E402


def sample_video():
    for candidate in sorted((ROOT / 'projects').rglob('input.mp4'), key=lambda p: p.stat().st_size):
        if candidate.stat().st_size < 12 * 1024 ** 2:
            return candidate
    raise SystemExit('No sample input.mp4 under projects/ to serve.')


def serve(directory):
    quiet = type('Quiet', (SimpleHTTPRequestHandler,), {'log_message': lambda *args, **kwargs: None})
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(quiet, directory=str(directory)))
    server.daemon_threads = True
    __import__('threading').Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


def prepare_media(media):
    media.mkdir(parents=True, exist_ok=True)
    source = sample_video()
    shutil.copyfile(source, media / 'clip.mp4')
    shutil.copyfile(source, media / 'clip_hd.mp4')
    (media / 'page.html').write_text(
        '<html><head><meta property="og:video" content="/clip.mp4"></head>'
        '<body><video src="/clip.mp4"></video></body></html>', encoding='utf-8')
    (media / 'variants.html').write_text(
        '<html><script>{"video_info":{"variants":['
        '{"bitrate":256000,"content_type":"video/mp4","url":"/clip.mp4?tag=low"},'
        '{"bitrate":2176000,"content_type":"video/mp4","url":"/clip_hd.mp4?tag=high"},'
        '{"content_type":"application/x-mpegURL","url":"/hls/clip.m3u8"}]}}</script></html>',
        encoding='utf-8')
    (media / 'blank.html').write_text('<html><body><p>No video here.</p></body></html>', encoding='utf-8')
    (media / 'tag.html').write_text(
        '<html><video controls><source src="clip_hd.mp4" type="video/mp4"></video></html>',
        encoding='utf-8')
    (media / 'tweet-result').write_text(json.dumps({
        '__typename': 'Tweet',
        'id_str': '2115605908969652430',
        'mediaDetails': [{
            'type': 'video',
            'video_info': {'variants': [
                {'content_type': 'application/x-mpegURL', 'url': '/hls/clip.m3u8'},
                {'bitrate': 256000, 'content_type': 'video/mp4', 'url': '/clip.mp4'},
                {'bitrate': 2176000, 'content_type': 'video/mp4', 'url': '/clip_hd.mp4'}]}}]}),
        encoding='utf-8')
    (media / 'hls').mkdir()
    subprocess.run([executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(media / 'clip.mp4'), '-c', 'copy', '-f', 'hls', '-hls_time', '1',
                    '-hls_playlist_type', 'vod', '-hls_segment_filename', str(media / 'hls' / 'seg%03d.ts'),
                    str(media / 'hls' / 'clip.m3u8')], check=True, creationflags=0x08000000)


def main():
    report, failures = {}, []
    media, downloads = DATA / 'media', DATA / 'downloads'
    prepare_media(media)
    server, base = serve(media)
    try:
        seen = []
        first = video_url.download(f'{base}/clip.mp4', downloads,
                                   progress=lambda done, total: seen.append((done, total)))
        info = probe(first)
        report['direct'] = {'name': first.name, 'frames': info['frames'], 'fps': round(info['fps'], 3),
                            'progress_to_last_byte': bool(seen and seen[-1][0] == first.stat().st_size
                                                          and seen[-1][1] > 0)}
        if first.name != 'clip.mp4' or info['frames'] < 4:
            failures.append('a direct link did not produce a readable clip.mp4')

        second = video_url.download(f'{base}/clip.mp4', downloads)
        if second.name != 'clip (2).mp4' or second.stat().st_size != first.stat().st_size:
            failures.append('a repeated download overwrote or reused the first file')

        for url, label in ((f'{base}/blank.html', 'page without video'), (f'{base}/missing.mp4', 'missing file')):
            try:
                video_url.download(url, downloads)
            except ValueError as error:
                report[label] = str(error)
            else:
                failures.append(f'{label} was saved instead of rejected')

        report['resolved'] = {
            'og:video': video_url.direct_media(f'{base}/page.html')[0].rsplit('/', 1)[-1],
            'variants': video_url.direct_media(f'{base}/variants.html')[0].rsplit('/', 1)[-1],
            'video tag': video_url.direct_media(f'{base}/tag.html')[0].rsplit('/', 1)[-1]}
        if report['resolved'] != {'og:video': 'clip.mp4', 'variants': 'clip_hd.mp4?tag=high',
                                  'video tag': 'clip_hd.mp4'}:
            failures.append('a page link did not resolve to its best video address')

        # An X post only exposes its video through the public syndication JSON.
        original = video_url.SYNDICATION
        video_url.SYNDICATION = f'{base}/tweet-result'
        try:
            post = 'https://x.com/HaileyAILabs/status/2115605908969652430/video/1'
            address, hint = video_url.direct_media(post)
            report['x'] = {'address': address.rsplit('/', 1)[-1], 'name': hint}
            if address.rsplit('/', 1)[-1] != 'clip_hd.mp4' or hint != 'HaileyAILabs_2115605908969652430':
                failures.append('an X post did not resolve to its best mp4')
            from_post = video_url.download(post, downloads)
            if from_post.name != 'HaileyAILabs_2115605908969652430.mp4' or probe(from_post)['frames'] < 4:
                failures.append('downloading an X post did not name and read the file')
            report['x']['saved'] = from_post.name
        finally:
            video_url.SYNDICATION = original

        hls = video_url.download(f'{base}/hls/clip.m3u8', downloads)
        hls_info = probe(hls)
        report['hls'] = {'name': hls.name, 'frames': hls_info['frames']}
        if not hls.name.endswith('.mp4') or hls_info['frames'] != probe(first)['frames']:
            failures.append('an HLS playlist was not remuxed into a matching mp4')

        lock = json.loads((ROOT / 'installer/runtime-lock.json').read_text(encoding='utf8'))
        pinned = next((p['version'] for g in lock['groups'] for p in g['packages']
                       if p['name'] == 'yt-dlp'), None)
        extractor = video_url.ytdlp()
        report['extractor'] = {'pinned': pinned,
                               'installed': extractor.version.__version__ if extractor else None}
        if pinned is None or extractor is None or pinned != extractor.version.__version__:
            failures.append('the installed extractor does not match installer/runtime-lock.json')

        report['names'] = [video_url.file_name('https://example.com/a/clip.mp4?t=9'),
                           video_url.file_name('https://example.com/stream', 'video/webm'),
                           video_url.file_name('https://example.com/')]
        if report['names'][:2] != ['clip.mp4', 'stream.webm']:
            failures.append('link names are not derived from the path and content type')

        # The dialog hands the link over for the preview and saves nothing; the
        # workspace's Download button is what keeps the clip for reconstruction.
        from PySide6.QtWidgets import QApplication
        application = QApplication.instance() or QApplication([])
        link = 'https://x.com/HaileyAILabs/status/2115605908969652430/video/1'
        dialog = video_url.VideoUrlDialog(None, link)
        dialog.begin()
        application.processEvents()
        report['dialog'] = {'url': dialog.url, 'path': str(dialog.path) if dialog.path else None,
                            'accepted': dialog.result() == 1, 'state': dialog.state.text()}
        if dialog.url != link or dialog.path is not None:
            failures.append('the URL dialog did not hand over the previewed link')
        dialog.deleteLater()
    finally:
        server.shutdown()
        server.server_close()
    report['downloads'] = sorted(path.name for path in downloads.glob('*'))
    report['pass'] = not failures
    report['failures'] = failures
    print('CHECK-VIDEO-URL ' + json.dumps(report, ensure_ascii=False))
    shutil.rmtree(DATA, ignore_errors=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
