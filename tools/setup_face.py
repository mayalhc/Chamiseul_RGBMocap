"""Optional face capture runtime: MediaPipe Face Landmarker over the shared Python.

The packages go into their own overlay folder (runtime_face) and never into the
shared runtime, whose numpy and OpenCV the body models depend on. The model file
is downloaded from Google's MediaPipe model storage and checked by SHA-256.
Nothing else in the installation, the models or the projects is changed.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'installer' / 'face-runtime-lock.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model(lock):
    row = lock['model']
    target = ROOT / row['path']
    if target.is_file() and sha256(target) == row['sha256']:
        print('Face model already verified.', flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.partial')
    print('Downloading the face model...', flush=True)
    with urllib.request.urlopen(row['url'], timeout=60) as response, open(partial, 'wb') as out:
        shutil.copyfileobj(response, out)
    if sha256(partial) != row['sha256']:
        partial.unlink(missing_ok=True)
        raise RuntimeError('The downloaded face model does not match its checksum.')
    partial.replace(target)
    return target


def verify(folder, model_path):
    code = ('import sys; sys.path.insert(0, %r); import numpy, cv2, mediapipe; '
            'from mediapipe.tasks.python import vision, BaseOptions; '
            'o = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=%r), '
            'output_face_blendshapes=True); '
            'vision.FaceLandmarker.create_from_options(o).close(); '
            'print("Face runtime verified:", mediapipe.__version__, numpy.__version__)'
            % (str(folder), str(model_path)))
    subprocess.run([str(ROOT / 'runtime' / 'python.exe'), '-I', '-X', 'utf8', '-B', '-c', code],
                   check=True, timeout=300)


def setup():
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    python = ROOT / 'runtime' / 'python.exe'
    if not python.is_file():
        raise RuntimeError('Run install.cmd first.')
    folder = ROOT / lock['folder']
    model_path = model(lock)
    stamp = folder / 'RGB_MOCAP_FACE.json'
    if stamp.is_file() and json.loads(stamp.read_text(encoding='utf-8')).get('packages') == lock['packages']:
        verify(folder, model_path)
        print('Face runtime is already installed.', flush=True)
        return 0
    partial = folder.with_name(folder.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    print('Installing the face runtime packages...', flush=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-deps', '--only-binary=:all:', '--target', str(partial),
                    *lock['packages']], check=True, timeout=1800)
    verify(partial, model_path)
    (partial / 'RGB_MOCAP_FACE.json').write_text(
        json.dumps(dict(packages=lock['packages'], model_sha256=lock['model']['sha256']), indent=1),
        encoding='utf-8')
    if folder.exists():
        old = folder.with_name(folder.name + '.old')
        shutil.rmtree(old, ignore_errors=True)
        folder.replace(old)
        shutil.rmtree(old, ignore_errors=True)
    partial.replace(folder)
    print('Face runtime ready. Existing runtimes, models and projects were not changed.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(setup())
