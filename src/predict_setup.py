"""Install the optional pinned LocalFold engine, independently of ChimeraX imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

REVISION = '68c59196fa7d32c6edfc061ed369889c1cdb680b'
REPOSITORY = 'https://github.com/sokrypton/localfold.git'
PHASE = 'AlphaLens setup: '


def progress(message):
    print(PHASE + message, flush=True)


def check_platform():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('LocalFold setup requires Apple Silicon macOS.')


def check_prerequisites():
    check_platform()
    if subprocess.run(['/usr/bin/xcode-select', '-p'], capture_output=True).returncode:
        raise RuntimeError('Install Xcode Command Line Tools with xcode-select --install, then retry Predict.')
    for name in ('git', 'clang++', 'clang', 'curl', 'ar'):
        if not shutil.which(name):
            raise RuntimeError(f'Missing {name}. Install Xcode Command Line Tools, then retry Predict.')


def install(prefix, models, source=None):
    """Build under an installation lock; preserve existing tracked source files."""
    check_prerequisites()
    import fcntl
    prefix = Path(prefix).expanduser().resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / 'setup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another LocalFold setup is running. Retry Predict when it finishes.') from None
        source = Path(source).expanduser().resolve() if source else prefix / 'engine' / REVISION
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='.localfold-setup-', dir=source.parent))
            try:
                progress('Downloading LocalFold source')
                subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', REPOSITORY, str(stage)], check=True)
                subprocess.run(['git', '-C', str(stage), 'checkout', '--detach', REVISION], check=True)
                stage.rename(source)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if head != REVISION or changed:
            raise RuntimeError('The source must be clean and at the pinned revision; existing files were preserved.')
        progress('Building ' + ', '.join(m.upper() for m in models) + ' (first use only)')
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '/usr/bin:/bin')
        subprocess.run(['bash', 'metal/build.sh', *models], cwd=source, env=env, check=True)
        target_dir = prefix / 'bin'
        target_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            target = target_dir / f'localfold-{model}'
            temporary = target.with_suffix('.part')
            shutil.copy2(source / f'metal/{model}/localfold-{model}', temporary)
            checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
            info = target.with_suffix('.json')
            receipt = info.with_suffix('.json.part')
            receipt.write_text(json.dumps(dict(repository=REPOSITORY, revision=head, sha256=checksum), indent=2)+'\n', encoding='utf-8')
            temporary.replace(target)
            receipt.replace(info)
            progress(f'Installed {target}')
        shutil.copy2(source / 'LICENSE', target_dir / 'LocalFold-LICENSE.txt')


class SetupJob:
    """Cancellable installer process; only poll() and read_messages() run on the UI thread."""
    def __init__(self, prefix, backend, *, script=None):
        self.prefix = Path(prefix)
        self.backend = backend
        self.script = Path(script) if script else Path(__file__)
        self.state = 'prepared'
        self.process = self.log = None
        self.error = self.cancelled_at = None
        self._offset = 0
        self._pending = ''
        self.stage = 'Preparing LocalFold setup'

    def start(self):
        if self.state != 'prepared':
            raise ValueError('This setup has already been started.')
        check_platform()
        logs = self.prefix / 'setup-logs'
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6] + '.log')
        python = Path(sys.executable)
        if not python.name.startswith('python'):
            python = Path(sys.exec_prefix) / 'bin' / f'python{sys.version_info.major}.{sys.version_info.minor}'
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        self.started = time.monotonic()
        self.log = self.log_path.open('w', encoding='utf-8')
        try:
            self.process = subprocess.Popen([str(python), '-B', '-u', str(self.script), '--prefix', str(self.prefix),
                                             '--models', self.backend], env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            self.log.close()
            self.state = 'failed'
            raise
        self.state = 'running'
        return self

    def read_messages(self):
        with self.log_path.open('rb') as stream:
            stream.seek(self._offset)
            chunk = stream.read(65536)
            self._offset = stream.tell()
        lines = (self._pending + chunk.decode('utf-8', errors='replace')).split('\n')
        self._pending = lines.pop()[-4096:]
        messages = [line[len(PHASE):] for line in lines if line.startswith(PHASE)]
        if messages:
            self.stage = messages[-1]
        return messages

    def cancel(self):
        if self.state == 'running' and self.cancelled_at is None:
            self.cancelled_at = time.monotonic()
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def poll(self):
        if self.state != 'running':
            return self.state
        if self.cancelled_at and time.monotonic() - self.cancelled_at > 1 and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        code = self.process.poll()
        if code is None:
            return self.state
        self.log.close()
        if self.cancelled_at:
            self.state = 'cancelled'
        elif code:
            self.state = 'failed'
            with self.log_path.open('rb') as stream:
                stream.seek(max(0, self.log_path.stat().st_size-1600))
                detail = stream.read().decode('utf-8', errors='replace').strip()
            self.error = f'LocalFold setup failed (status {code}). {detail}\nSetup log: {self.log_path}'
        else:
            target = self.prefix / 'bin' / f'localfold-{self.backend}'
            self.state = 'complete' if target.is_file() and os.access(target, os.X_OK) else 'failed'
            if self.state == 'failed':
                self.error = f'LocalFold setup did not create {target}. Setup log: {self.log_path}'
        return self.state

    def shutdown(self):
        if self.state == 'running':
            self.cancel()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
            self.poll()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/alphalens')
    parser.add_argument('--source', type=Path, help='Existing clean checkout at the pinned revision')
    parser.add_argument('--models', nargs='+', choices=['af2', 'af3'], default=['af2', 'af3'])
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('Setup cancelled.')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        progress('Checking build tools')
        install(args.prefix, args.models, args.source)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'LocalFold setup: {error}', file=sys.stderr, flush=True)
        return 1
    progress('Ready. Model weights download on the first prediction; AF3 still requires terms acceptance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
