import json
import os
from pathlib import Path
import subprocess
import time

import pytest
from alphalens import predict_setup as setup


def finish(job):
    end = time.monotonic()+5
    while job.poll() == 'running' and time.monotonic() < end:
        time.sleep(.02)
    assert job.state != 'running'
    return job


@pytest.fixture
def worker_platform(monkeypatch):
    # Stand-in installer processes are pure Python and can exercise the lifecycle on CI.
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'arm64')


def test_setup_failure_cancel_and_retry(tmp_path, worker_platform):
    script = tmp_path/'worker.py'
    script.write_text('import sys\nprint("controlled failure", flush=True)\nsys.exit(7)\n')
    job = finish(setup.SetupJob(tmp_path/'prefix', 'af3', script=script).start())
    assert job.state == 'failed' and 'controlled failure' in job.error
    script.write_text('import time\nprint("AlphaLens setup: Downloading source", flush=True)\ntime.sleep(60)\n')
    job = setup.SetupJob(tmp_path/'prefix', 'af3', script=script).start()
    time.sleep(.1)
    assert job.read_messages() == ['Downloading source']
    job.shutdown()
    assert job.state == 'cancelled' and job.process.poll() is not None
    script.write_text('''import argparse,pathlib
p=argparse.ArgumentParser();p.add_argument('--prefix');p.add_argument('--models');a=p.parse_args()
b=pathlib.Path(a.prefix)/'bin'/('localfold-'+a.models);b.parent.mkdir(parents=True,exist_ok=True)
b.write_text('#!/bin/sh\\nexit 0\\n');b.chmod(0o755)
print('AlphaLens setup: Installed engine',flush=True)
''')
    job = finish(setup.SetupJob(tmp_path/'prefix', 'af3', script=script).start())
    assert job.state == 'complete'
    assert job.read_messages() == ['Installed engine']


def test_success_requires_an_executable(tmp_path, worker_platform):
    script = tmp_path/'empty.py';script.write_text('pass\n')
    job = finish(setup.SetupJob(tmp_path/'prefix', 'af2', script=script).start())
    assert job.state == 'failed' and 'did not create' in job.error


def test_prerequisite_error_is_actionable(monkeypatch):
    monkeypatch.setattr(setup.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(setup.platform,'machine',lambda:'arm64')
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1))
    with pytest.raises(RuntimeError,match='xcode-select --install'):
        setup.check_prerequisites()


def test_unsupported_platform_is_reported_before_importing_posix_lock(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Windows')
    monkeypatch.setitem(sys.modules, 'fcntl', None)
    with pytest.raises(RuntimeError, match='Apple Silicon macOS'):
        setup.install(tmp_path/'prefix', ['af3'])
    with pytest.raises(RuntimeError, match='Apple Silicon macOS'):
        setup.SetupJob(tmp_path/'prefix', 'af3').start()
    assert not (tmp_path/'prefix').exists()


def test_source_checkout_and_concurrent_setup_are_preserved(tmp_path,monkeypatch):
    import fcntl
    prefix=tmp_path/'prefix';prefix.mkdir()
    source=tmp_path/'source';source.mkdir();(source/'keep').write_text('user data')
    monkeypatch.setattr(setup,'check_prerequisites',lambda:None)
    with (prefix/'setup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Another LocalFold setup'):
            setup.install(prefix,['af2'],source)
    monkeypatch.setattr(setup.subprocess,'check_output',lambda args,**k:setup.REVISION if args[-1]=='HEAD' else ' M tracked-file')
    with pytest.raises(RuntimeError,match='existing files were preserved'):
        setup.install(prefix,['af2'],source)
    assert (source/'keep').read_text()=='user data'
    assert not (prefix/'bin').exists()


@pytest.mark.parametrize('system,machine,native', [
    ('Darwin', 'arm64', 'metal'), ('Linux', 'x86_64', 'cuda'), ('Linux', 'AMD64', 'cuda'),
    ('Darwin', 'x86_64', None), ('Linux', 'aarch64', None), ('Windows', 'AMD64', None)])
def test_native_platform(monkeypatch, system, machine, native):
    monkeypatch.setattr(setup.platform, 'system', lambda: system)
    monkeypatch.setattr(setup.platform, 'machine', lambda: machine)
    assert setup.native_platform() == native


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'x86_64')


def test_cuda_prerequisite_errors_are_actionable(linux, monkeypatch):
    tools = {'git', 'curl', 'gzip', 'sha256sum', 'g++', 'nvidia-smi'}
    monkeypatch.setattr(setup.shutil, 'which', lambda name: f'/usr/bin/{name}' if name in tools else None)
    with pytest.raises(RuntimeError, match='Missing nvcc.*CUDA toolkit'):
        setup.check_prerequisites()
    tools.add('nvcc')
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='\n'))
    with pytest.raises(RuntimeError, match='No NVIDIA GPU'):
        setup.check_prerequisites()
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='8.0\n'))
    monkeypatch.setattr(setup, 'CUDA_BUILD_FILES', ())
    assert setup.check_prerequisites() == 'cuda'


def test_cuda_build_files_of_another_user_are_reported(linux, monkeypatch, tmp_path):
    monkeypatch.setattr(setup.shutil, 'which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='8.0\n'))
    stale = tmp_path / 'localfold-cuda-build.log'
    stale.write_text('')
    monkeypatch.setattr(setup, 'CUDA_BUILD_FILES', (str(stale),))
    monkeypatch.setattr(setup.os, 'access', lambda path, mode: False)
    with pytest.raises(RuntimeError, match='belongs to another user'):
        setup.check_prerequisites()


@pytest.fixture
def cuda_source(tmp_path, monkeypatch):
    """A clean pinned checkout stand-in whose build is scripted by each test."""
    toolkit = tmp_path / 'cuda-12.8'
    (toolkit / 'bin').mkdir(parents=True)
    (toolkit / 'lib64').mkdir()
    (toolkit / 'lib64' / 'libcublas.so.12').write_text('')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'LICENSE').write_text('beerware')
    log = tmp_path / 'localfold-cuda-build.log'
    monkeypatch.setattr(setup, 'CUDA_BUILD_FILES', (str(tmp_path / 'marker'), str(log)))
    monkeypatch.setattr(setup, 'check_prerequisites', lambda: 'cuda')
    monkeypatch.setattr(setup.shutil, 'which', lambda name: str(toolkit / 'bin' / name) if name == 'nvcc' else None)
    state = dict(built=False, calls=[], dirty='', code=0, log='', ports=['af2'])

    def check_output(args, **k):
        if args[-1] == 'HEAD':
            return setup.REVISION
        return state['dirty'] if state['built'] else ''

    def run(args, cwd=None, env=None, check=False, **k):
        if args[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(args, 0, stdout='8.0\n')
        state['calls'].append((args, (env or {}).get('LOCALFOLD_RPATH')))
        if args[0] == 'bash':
            for port in state['ports']:
                binary = Path(cwd) / 'cuda' / port / f'localfold-{port}'
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_text('#!/bin/sh\n')
                binary.chmod(0o755)
            log.write_text(state['log'])
            state['built'] = True
        return subprocess.CompletedProcess(args, state['code'] if args[0] == 'bash' else 0)
    monkeypatch.setattr(setup.subprocess, 'check_output', check_output)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    return source, toolkit, state


def test_cuda_install_builds_with_toolkit_rpath_and_records_backend(tmp_path, cuda_source):
    source, toolkit, state = cuda_source
    setup.install(tmp_path / 'prefix', ['af2'], source)
    assert state['calls'] == [(['bash', 'cuda/build.sh', 'af2'], str(toolkit / 'lib64'))]
    receipt = json.loads((tmp_path / 'prefix' / 'bin' / 'localfold-af2.json').read_text())
    assert receipt['native'] == 'cuda' and receipt['revision'] == setup.REVISION
    assert receipt['cuda_compute_capability'] == '8.0'
    assert os.access(tmp_path / 'prefix' / 'bin' / 'localfold-af2', os.X_OK)


def test_cuda_build_restores_rewritten_probes_and_tolerates_unused_featuriser(tmp_path, cuda_source, capsys):
    # Seen on EL8 (glibc 2.28): rfd4p-featurise fails to link, two committed probes are rewritten,
    # and the requested engine is built. The next engine's setup needs a clean checkout.
    source, _, state = cuda_source
    state.update(code=1, dirty=' M cuda/rfd4p/featurise/assign-probe\n M cuda/rfd4p/featurise/superpose-probe\n',
                 log='built the featurisers\nFAILED rfd4p-featurise\nbuilt af2 for sm_80\n')
    setup.install(tmp_path / 'prefix', ['af2'], source)
    assert (['git', '-C', str(source), 'checkout', '--', 'cuda/rfd4p/featurise/assign-probe',
             'cuda/rfd4p/featurise/superpose-probe'], None) in state['calls']
    assert 'design-model featuriser did not build' in capsys.readouterr().out
    assert (tmp_path / 'prefix' / 'bin' / 'localfold-af2').is_file()


@pytest.mark.parametrize('log,ports', [
    ('FAILED rfd4p-featurise\nFAILED af2 for sm_80\n', ['af2']),       # the engine itself failed
    ('FAILED the standalone object\n', ['af2']),                        # what the engines link
    ('FAILED rfd4p-featurise\n', []),                                   # no engine binary
    ('', ['af2'])])                                                     # nonzero without a FAILED line
def test_cuda_build_failures_that_matter_stop_setup(tmp_path, cuda_source, log, ports):
    source, _, state = cuda_source
    state.update(code=1, log=log, ports=ports)
    with pytest.raises(subprocess.CalledProcessError):
        setup.install(tmp_path / 'prefix', ['af2'], source)
    assert not (tmp_path / 'prefix' / 'bin' / 'localfold-af2').exists()
