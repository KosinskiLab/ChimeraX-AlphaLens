"""Opt-in GUI check: long input stays runnable; optional timeout and Cancel work.

Run in a disposable ChimeraX GUI session. ALPHALENS_SOURCE=checkout (default) or
installed selects the package; ALPHALENS_TEST_OUTPUT must be a new directory.
Uses a sleeping test executable, not an actual fold, for process-control checks.
"""
from pathlib import Path
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback

from Qt.QtCore import QTimer

out = Path(os.environ['ALPHALENS_TEST_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
    pkg = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1] / 'src'
    pkg = 'chimerax.alphalens_warning_check'
    spec = importlib.util.spec_from_file_location(pkg, src / '__init__.py',
                                                 submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
ui = importlib.import_module(pkg + '.predict_ui')


def pump(predicate):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        session.ui.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


def main():
    dialog = None
    try:
        engine = out / 'sleep-engine'
        engine.write_text('#!/bin/sh\nexec /bin/sleep 60\n')
        engine.chmod(0o755)
        sequence = 'ACDEFGHIKLMNPQRSTVWY' * 100  # Synthetic UI input: 2,000 residues.
        msa = out / 'query.a3m'
        msa.write_text('>query\n' + sequence + '\n')
        for minutes in (0, 1):
            dialog = ui.PredictionDialog(session=session)
            dialog.show()
            dialog.input_text.setPlainText(sequence)
            dialog.load_button.click()
            pump(lambda: not dialog.lookup_timer.isActive())
            assert '2,000 residues' in dialog.size_warning.text()
            assert not dialog.size_warning.isHidden()
            assert not dialog.start_button.isEnabled(), 'MSA choice must be explicit'
            assert not dialog.msa_hint.isHidden()
            assert dialog.time_limit.value() == 0
            dialog.time_limit.setValue(minutes)
            dialog.executable.setText(str(engine))
            dialog.msa.setText(str(msa))
            assert dialog.start_button.isEnabled(), dialog.preview.text()
            dialog.output.setText(str(out / f'timeout-{minutes}'))
            if minutes == 0:
                dialog.grab().save(str(out / 'warning.png'))
            dialog.start_button.click()
            assert dialog.job and dialog.job.state == 'running', dialog.status.text()
            expected = minutes * 60 or None
            assert dialog.job.timeout == expected
            assert dialog.job.manifest['timeout_seconds'] == expected
            assert not dialog.time_limit.isEnabled()
            dialog.cancel_button.click()
            pump(lambda: not dialog.timer.isActive())
            assert dialog.job.state == 'cancelled', dialog.status.text()
            assert dialog.progress_bar.maximum() == 100
            assert dialog.progress_bar.value() == 0
            assert dialog.progress_label.text() == 'Cancelled'
            assert dialog.time_limit.isEnabled()
            dialog.close()
            dialog = None
        (out / 'results.json').write_text(json.dumps(dict(result='PASS', residues=2000,
            prediction_enabled=True, warning_visible=True, timeout_seconds_checked=[None, 60],
            cancellation_passed=True)) + '\n')
        print('LARGE INPUT WARNING GUI CHECK PASSED', file=sys.__stdout__, flush=True)
        os._exit(0)
    except Exception:
        if dialog is not None:
            dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__)
        os._exit(1)


QTimer.singleShot(1000, main)
