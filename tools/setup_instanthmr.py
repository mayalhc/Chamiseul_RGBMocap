"""Install the InstantHMR live estimator's one extra package, without touching a runtime.

ONNX Runtime GPU goes into its own folder (runtime_instanthmr) next to the ExMR
runtime, which supplies Python, NumPy and the CUDA libraries. Nothing inside any
existing runtime is changed. The wheel is pinned by URL, size and SHA-256.
"""
from pathlib import Path, PurePosixPath
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.support_components import download  # noqa: E402

TARGET = ROOT / 'runtime_instanthmr'
RECEIPT = TARGET / 'receipt.json'


def spec():
    return json.loads((ROOT / 'installer/instanthmr-runtime.json').read_text(encoding='utf-8'))


def installed():
    try:
        return json.loads(RECEIPT.read_text(encoding='utf-8')).get('sha256') == spec()['wheel']['sha256'] \
            and (TARGET / 'onnxruntime' / '__init__.py').is_file()
    except (OSError, ValueError):
        return False


def extract(wheel, destination):
    total = 0
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            name = item.filename
            path = PurePosixPath(name)
            if chr(92) in name or path.is_absolute() or '..' in path.parts or ':' in name:
                raise ValueError('Unsafe wheel member: ' + name)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Wheel links are not allowed')
            total += item.file_size
            if total > 2 * 1024 ** 3:
                raise ValueError('Wheel is too large')
        archive.extractall(destination)


def verify():
    exmr = ROOT / 'runtime_exmr' / 'python.exe'
    code = ("import sys; sys.path.insert(0, r'%s'); import torch, onnxruntime as o; "
            "print(o.__version__, o.get_available_providers())" % TARGET)
    result = subprocess.run([str(exmr), '-I', '-c', code], capture_output=True, text=True, timeout=300)
    print(result.stdout.strip() or result.stderr.strip()[-400:])
    return result.returncode == 0 and 'CUDAExecutionProvider' in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not (ROOT / 'runtime_exmr' / 'python.exe').is_file():
        print('Run install.cmd first: the ExMR runtime provides Python and CUDA.')
        return 1
    if installed() and not args.force:
        print('ONNX Runtime GPU is already installed for InstantHMR.')
        return 0 if verify() else 1
    row = spec()['wheel']
    cache = ROOT / '.setup' / 'instanthmr-downloads'
    cache.mkdir(parents=True, exist_ok=True)
    wheel = download(cache, 'onnxruntime-gpu', row)
    staging = ROOT / 'runtime_instanthmr.partial'
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        extract(wheel, staging)
        (staging / 'receipt.json').write_text(json.dumps({'sha256': row['sha256'], 'bytes': row['bytes']}),
                                              encoding='utf-8')
        shutil.rmtree(TARGET, ignore_errors=True)
        os.replace(staging, TARGET)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not verify():
        print('InstantHMR support was installed but ONNX Runtime did not report the CUDA provider.')
        return 1
    print('InstantHMR live estimator support is ready.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
