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


def test_setup_failure_cancel_and_retry(tmp_path):
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


def test_success_requires_an_executable(tmp_path):
    script = tmp_path/'empty.py';script.write_text('pass\n')
    job = finish(setup.SetupJob(tmp_path/'prefix', 'af2', script=script).start())
    assert job.state == 'failed' and 'did not create' in job.error


def test_prerequisite_error_is_actionable(monkeypatch):
    monkeypatch.setattr(setup.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(setup.platform,'machine',lambda:'arm64')
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1))
    with pytest.raises(RuntimeError,match='xcode-select --install'):
        setup.check_prerequisites()


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
