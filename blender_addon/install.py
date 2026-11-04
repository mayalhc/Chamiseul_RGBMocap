#!/usr/bin/env python3
"""Build and install the RGB MoCap Bridge Blender extension.

  python blender_addon/install.py [--blender PATH] [--repo user_default] [--build-only]

The package is built from `blender_addon/rgbmocap_bridge` (which holds
blender_manifest.toml) into `blender_addon/dist/*.zip` and installed into a
user repository, replacing any older copy. Works with any Python 3; it only
shells out to Blender's `--command extension` CLI.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'rgbmocap_bridge'
DIST = HERE / 'dist'
MANIFEST = SOURCE / 'blender_manifest.toml'


def manifest_value(key):
    for line in MANIFEST.read_text(encoding='utf-8').splitlines():
        name, _, value = line.partition('=')
        if name.strip() == key:
            return value.strip().strip('"')
    raise SystemExit('%s has no %s' % (MANIFEST, key))


def find_blender(explicit):
    if explicit:
        return Path(explicit)
    if os.environ.get('BLENDER_EXE'):
        return Path(os.environ['BLENDER_EXE'])
    found = []
    for root in ('C:/Program Files/Blender', '/Applications', '/usr/share', '/opt',
                 str(Path.home() / 'Applications')):
        base = Path(root)
        if not base.is_dir():
            continue
        found += list(base.glob('**/blender.exe')) + list(base.glob('**/Blender.app/Contents/MacOS/Blender'))
        found += list(base.glob('**/blender'))
    found = [path for path in found if path.is_file()]
    if not found:
        raise SystemExit('Blender was not found; pass --blender PATH')
    return max(found, key=lambda path: path.stat().st_mtime)


def run(command, **kwargs):
    print('$', ' '.join(str(part) for part in command), flush=True)
    result = subprocess.run([str(part) for part in command], **kwargs)
    if result.returncode:
        raise SystemExit('%s failed with exit code %d' % (command[0], result.returncode))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', help='Blender executable to build and install into')
    parser.add_argument('--repo', default='user_default', help='Target extension repository id')
    parser.add_argument('--build-only', action='store_true', help='Only write the extension zip')
    options = parser.parse_args()

    if not MANIFEST.is_file():
        raise SystemExit('Missing manifest: %s' % MANIFEST)
    blender = find_blender(options.blender)
    print('Blender:', blender)

    DIST.mkdir(parents=True, exist_ok=True)
    for stale in DIST.glob('*.zip'):
        stale.unlink()
    run([blender, '-b', '--command', 'extension', 'build',
         '--source-dir', SOURCE, '--output-dir', DIST])
    package = next(DIST.glob('%s-*.zip' % manifest_value('id')), None)
    if package is None:
        raise SystemExit('The extension build produced no zip in %s' % DIST)
    print('Built:', package, '(%.1f KiB)' % (package.stat().st_size / 1024))

    if options.build_only:
        return 0

    # A legacy scripts/addons copy would register the same UI twice.
    run([blender, '-b', '--python', HERE / 'legacy_cleanup.py'])
    run([blender, '-b', '--command', 'extension', 'install-file',
         '-r', options.repo, '-e', package])
    check = ("import addon_utils;"
             "print('EXTENSION-STATE', addon_utils.check('bl_ext.%s.%s'))"
             % (options.repo, manifest_value('id')))
    run([blender, '-b', '--python-expr', check])
    return 0


if __name__ == '__main__':
    sys.exit(main())
