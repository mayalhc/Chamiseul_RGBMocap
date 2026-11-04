"""Register a provider download automatically after the person fetches it.

Several model files (MAMMA, SMPL-X, SMPL, MANO ...) are only offered after a
free registration and sign-in. The person signs in and downloads in their own
browser; this watcher notices the finished file in the Downloads folder,
checks it through the normal validated Import job, stores it in the models
folder and removes the original from Downloads. Nothing is typed into or read
from the browser, and only files whose name (or archive member) a model row
expects are touched.
"""
import shutil
import time
import zipfile
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QThread, QTimer, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from .paths import ROOT

STAGING = ROOT / 'cache' / 'model-incoming'
PARTIAL = ('.crdownload', '.part', '.tmp', '.download', '.partial', '.opdownload')
# Official archives name some files differently from the name the app stores.
ALIASES = {'smpl': ('basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',)}


def downloads_folder():
    found = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
    return Path(found) if found else Path.home() / 'Downloads'


def clear_staging():
    shutil.rmtree(STAGING, ignore_errors=True)


def staged(model):
    """A private path that keeps the exact file name the Import check expects."""
    folder = STAGING / (model.key + '-' + str(time.time_ns()))
    folder.mkdir(parents=True, exist_ok=True)
    return folder / model.path.name


def member_names(model):
    return {model.path.name.lower(), *(alias.lower() for alias in ALIASES.get(model.key, ()))}


def match_file(path, wanted):
    """Model rows a loose downloaded file can fill."""
    name = Path(path).name.lower()
    return [model for model in wanted if name in member_names(model)]


def match_archive(path, wanted):
    """[(model, member)] for archive members a wanted row expects (by base name)."""
    with zipfile.ZipFile(path) as bundle:
        index = {}
        for item in bundle.infolist():
            if not item.is_dir():
                index.setdefault(Path(item.filename).name.lower(), item)
    found = []
    for model in wanted:
        for name in member_names(model):
            if name in index:
                found.append((model, index[name]))
                break
    return found


def unpack(path, matches):
    """Extract matched members under the model's own file name; never by stored path."""
    STAGING.mkdir(parents=True, exist_ok=True)
    found = []
    with zipfile.ZipFile(path) as bundle:
        for model, item in matches:
            target = staged(model)
            with bundle.open(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 8 * 1024 ** 2)
            found.append((model, target))
    return found


class DownloadWatcher(QObject):
    """Polls the Downloads folder for new finished files that a model row expects."""

    def __init__(self, manager, parent=None, report=lambda text: None):
        super().__init__(parent)
        self.manager = manager
        self.report = report
        self.folder = downloads_folder()
        self.started = time.time() - 5
        self.sizes = {}
        self.pending = []
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.scan)

    def start(self):
        clear_staging()
        self.started = time.time() - 5
        self.timer.start()
        self.report('Watching ' + str(self.folder) + ' for the downloaded file...')

    def stop(self):
        self.timer.stop()

    def wanted(self):
        from .models import ready
        return [model for model in self.manager.models if not ready(model)]

    def scan(self):
        if self.pending:
            self.start_pending()
        try:
            entries = list(self.folder.iterdir())
        except OSError:
            return
        wanted = self.wanted()
        if not wanted:
            return
        for path in entries:
            if not path.is_file() or path.suffix.lower() in PARTIAL:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime < self.started:
                continue
            key = str(path)
            signature = (stat.st_size, stat.st_mtime_ns)
            if self.sizes.get(key) != signature:
                self.sizes[key] = signature
                continue  # still changing; checked again on the next scan
            self.take(path, wanted)

    def take(self, path, wanted):
        try:
            if path.suffix.lower() == '.zip':
                matches = match_archive(path, wanted)
                if not matches:
                    return
                found = unpack(path, matches)
            else:
                rows = match_file(path, wanted)
                if not rows:
                    return
                STAGING.mkdir(parents=True, exist_ok=True)
                found = []
                for model in rows:
                    target = staged(model)
                    shutil.copy2(path, target)
                    found.append((model, target))
        except (OSError, zipfile.BadZipFile) as error:
            self.report('Cannot read ' + path.name + ': ' + str(error))
            self.sizes.pop(str(path), None)
            return
        try:
            path.unlink()  # now held in staging; the models folder keeps the only copy
        except OSError as error:
            self.report('Could not remove ' + path.name + ' from Downloads: ' + str(error))
        self.sizes.pop(str(path), None)
        self.pending.extend(found)
        self.start_pending()

    def start_pending(self):
        if not self.pending or self.manager.window.busy:
            return  # retried by the next scan
        jobs = [('import_model', ['--model', model.key, '--video', str(file)]) for model, file in self.pending]
        names = ', '.join(model.name for model, _ in self.pending)
        self.pending = []
        self.manager.start_model_jobs(jobs)
        self.report('Found the download. Checking and registering: ' + names)


SERVER = 'https://download.is.tue.mpg.de/download.php'
SITE_NAMES = {'smpl': 'SMPL (smpl.is.tue.mpg.de)', 'smplx': 'SMPL-X (smpl-x.is.tue.mpg.de)',
              'mamma': 'MAMMA (mamma.is.tue.mpg.de)'}


def provider_url(domain, sfile):
    from urllib.parse import urlencode
    return SERVER + '?' + urlencode({'domain': domain, 'sfile': sfile})


class ProviderDownload(QThread):
    """Official scripted download: the account sign-in is posted with the request.

    This is the method the providers document for their own download scripts.
    The password is used for this one request and never written anywhere.
    """
    progress = Signal(int, int)
    finished_file = Signal(str)
    failed = Signal(str)

    def __init__(self, domain, sfile, user, password, parent=None):
        super().__init__(parent)
        self.domain, self.sfile, self.user, self.password = domain, sfile, user, password
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        import urllib.error
        import urllib.parse
        import urllib.request
        target = None
        try:
            STAGING.mkdir(parents=True, exist_ok=True)
            target = STAGING / (str(time.time_ns()) + '-' + Path(self.sfile).name)
            data = urllib.parse.urlencode({'username': self.user, 'password': self.password}).encode()
            self.user = self.password = None
            request = urllib.request.Request(provider_url(self.domain, self.sfile), data=data)
            with urllib.request.urlopen(request, timeout=60) as reply:
                kind = (reply.headers.get('Content-Type') or '').lower()
                if 'text/html' in kind:
                    raise RuntimeError('Sign-in was not accepted for this file. Check the email and password for this '
                                       'site, and that you accepted its license on the website.')
                total = int(reply.headers.get('Content-Length') or 0)
                done = 0
                with target.open('wb') as output:
                    while True:
                        chunk = reply.read(1024 * 1024)
                        if not chunk:
                            break
                        if self.cancelled:
                            raise RuntimeError('Download cancelled.')
                        output.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)
            if done == 0:
                raise RuntimeError('The server returned an empty file.')
            self.finished_file.emit(str(target))
        except urllib.error.HTTPError as error:
            self.failed.emit('The server refused the request (HTTP %s %s).' % (error.code, error.reason))
            target and target.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError) as error:
            self.failed.emit(type(error).__name__ + ': ' + str(error))
            target and target.unlink(missing_ok=True)


class SignInForm(QDialog):
    """Native sign-in form (no embedded browser)."""

    def __init__(self, parent, site):
        super().__init__(parent)
        self.setWindowTitle('Sign in to download')
        layout = QVBoxLayout(self)
        note = QLabel('Sign in with your own %s account (free registration on the website).\n'
                      'The app uses it only for this download and does not store it.' % SITE_NAMES.get(site, site))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.email = QLineEdit()
        self.email.setPlaceholderText('Email')
        self.password = QLineEdit()
        self.password.setPlaceholderText('Password')
        self.password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.email)
        layout.addWidget(self.password)
        row = QHBoxLayout()
        row.addStretch()
        ok = QPushButton('Download')
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        layout.addLayout(row)
        self.setMinimumWidth(420)

    def credentials(self):
        return self.email.text().strip(), self.password.text()


def download_with_sign_in(manager, key, parent, report):
    """Ask for the account, download the file for `key`, then register it."""
    from .models import PROVIDER_FILES, ready
    domain, sfile = PROVIDER_FILES[key]
    form = SignInForm(parent, domain)
    if form.exec() != QDialog.Accepted:
        return None
    user, password = form.credentials()
    form.password.clear()
    if not user or not password:
        report('Enter the email and password for this site.')
        return None
    worker = ProviderDownload(domain, sfile, user, password, parent)
    password = None
    wanted = [model for model in manager.models if not ready(model)]

    def progress(done, total):
        report('Downloading %s: %.0f%s MB' % (Path(sfile).name, done / 1e6, (' of %.0f' % (total / 1e6)) if total else ''))

    def finished(path):
        try:
            path = Path(path)
            if path.suffix.lower() == '.zip':
                found = unpack(path, match_archive(path, wanted))
                path.unlink(missing_ok=True)
            else:
                rows = match_file(Path(sfile).name, wanted) or [m for m in wanted if m.key == key]
                found = []
                for model in rows:
                    target = staged(model)
                    shutil.move(str(path), str(target))
                    found.append((model, target))
                    break
        except (OSError, zipfile.BadZipFile) as error:
            report('Cannot read the downloaded file: ' + str(error))
            return
        if not found:
            report('The download did not contain a file this tool needs.')
            return
        pending.extend(found)
        register()

    pending = []

    def register():
        if not pending:
            return
        if manager.window.busy:
            QTimer.singleShot(1500, register)
            return
        jobs = [('import_model', ['--model', model.key, '--video', str(file)]) for model, file in pending]
        names = ', '.join(model.name for model, _ in pending)
        pending.clear()
        manager.start_model_jobs(jobs)
        report('Downloaded. Checking and registering: ' + names)

    worker.progress.connect(progress)
    worker.finished_file.connect(finished)
    def failed(text):
        report('Download failed: ' + text)
        QMessageBox.warning(parent, 'Download failed', text)

    worker.failed.connect(failed)
    worker.finished.connect(worker.deleteLater)
    report('Signing in and downloading ' + Path(sfile).name + ' ...')
    worker.start()
    return worker
