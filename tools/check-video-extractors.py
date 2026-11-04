"""Check YouTube and TikTok links through the app's own download path.

This needs the internet: it resolves and downloads one short public video from
each site into a temporary folder, then reads it with the app's own probe.
It does not touch the project data or the user's downloads folder.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(tempfile.mkdtemp(prefix='rgbmocap-extract-'))
os.environ['RGB_MOCAP_DATA'] = str(DATA)
sys.path.insert(0, str(ROOT))
GUI = ROOT / 'runtime_gui'
if GUI.is_dir():
    sys.path.insert(0, str(GUI))

from mocap import video_url  # noqa: E402
from mocap.video import probe  # noqa: E402

LINKS = {
    'youtube': 'https://www.youtube.com/watch?v=jNQXAC9IVRw',   # 19 second public clip
    'tiktok': 'https://www.tiktok.com/@tiktok/video/7106594312292453675',
}


def main():
    report, failures = {}, []
    yt_dlp = video_url.ytdlp()
    report['extractor'] = yt_dlp.version.__version__ if yt_dlp else None
    if yt_dlp is None:
        failures.append('yt-dlp is not importable in the GUI runtime')
        report['pass'] = False
        report['failures'] = failures
        print('CHECK-VIDEO-EXTRACTORS ' + json.dumps(report, ensure_ascii=False))
        return 1
    lock = json.loads((ROOT / 'installer/runtime-lock.json').read_text(encoding='utf8'))
    pinned = next((p['version'] for g in lock['groups'] for p in g['packages'] if p['name'] == 'yt-dlp'), None)
    report['pinned'] = pinned
    if pinned != report['extractor']:
        failures.append('the installed extractor does not match installer/runtime-lock.json')
    for name, url in LINKS.items():
        folder = DATA / name
        folder.mkdir(parents=True)
        started = time.monotonic()
        seen = []
        try:
            path = video_url.download(url, folder, progress=lambda done, total: seen.append(done))
            info = probe(path)
        except Exception as error:
            report[name] = {'error': str(error)[:400]}
            failures.append(f'{name} link failed')
            continue
        report[name] = {'file': path.name, 'frames': info['frames'], 'fps': round(info['fps'], 3),
                        'width': info['width'], 'height': info['height'],
                        'mb': round(path.stat().st_size / 1024 ** 2, 2),
                        'progress_reports': len(seen), 'seconds': round(time.monotonic() - started, 1)}
        if info['frames'] < 4 or not seen:
            failures.append(f'{name} did not produce a readable video with progress')
    report['pass'] = not failures
    report['failures'] = failures
    print('CHECK-VIDEO-EXTRACTORS ' + json.dumps(report, ensure_ascii=False))
    shutil.rmtree(DATA, ignore_errors=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
