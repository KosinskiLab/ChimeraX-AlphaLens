"""Opt-in native GUI check of progress, MSA guidance, failures and real GPU folds.

ALPHALENS_PILOT names a saved 195-residue barnase/barstar prediction with fold.a3m.
ALPHALENS_TEST_OUTPUT must be a new directory. ALPHALENS_SOURCE selects checkout
(default) or installed. Run in a disposable ChimeraX process.
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
root = Path(os.environ['ALPHALENS_PILOT'])
if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
    pkg = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1] / 'src'
    pkg = 'chimerax.alphalens_progress_check'
    spec = importlib.util.spec_from_file_location(pkg, src / '__init__.py',
                                                 submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
ui = importlib.import_module(pkg + '.predict_ui')
cmd = importlib.import_module(pkg + '.cmd')


def pump(predicate, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


def main():
    dialog = None
    try:
        structure = cmd._open_structure(session, str(root / 'fold.pdb'))
        pae = cmd.alphalens(session, structure, pae_file=str(root / 'fold_confidences.json'), wait=True)
        sequences = [chain.characters for chain in structure.chains]
        assert [len(s) for s in sequences] == [108, 87]
        dialog = ui.PredictionDialog(session=session)
        dialog.show()
        dialog.input_text.setPlainText('\n'.join(f'>{i}\n{s}' for i, s in enumerate(sequences)))
        dialog.load_button.click()
        pump(lambda: not dialog.lookup_timer.isActive())
        assert not dialog.start_button.isEnabled()
        assert not dialog.msa_hint.isHidden()
        assert dialog.msa_hint.height() >= dialog.msa_hint.heightForWidth(dialog.msa_hint.width())
        assert dialog.job is None
        dialog.set_status('Running · 0.0 s · MSA: supplied')
        session.ui.processEvents()
        initial_height = dialog.height()
        initial_status_height = dialog.status.height()
        for tick in range(100):
            dialog.set_status(f'Running · {tick/10:.1f} s · MSA: supplied')
            session.ui.processEvents()
        layout_result = dict(before=initial_height, after=dialog.height(),
                             status_before=initial_status_height, status_after=dialog.status.height())
        (out / 'layout.json').write_text(json.dumps(layout_result))
        assert dialog.height() == initial_height, layout_result
        assert dialog.status.height() == initial_status_height, layout_result
        # Long errors stay fully accessible without enlarging the dialog.
        long_message = '\n'.join(f'Error detail {i}: ' + 'x'*120 for i in range(30))
        dialog.set_status(long_message)
        session.ui.processEvents()
        assert dialog.height() == initial_height
        assert dialog.status.toPlainText() == long_message
        assert dialog.status.verticalScrollBar().maximum() > 0
        dialog.set_status('Review the sequences, choose an MSA, then click Predict.')
        dialog.grab().save(str(out / 'msa-required.png'))
        dialog.search.setChecked(True)
        assert dialog.start_button.isEnabled() and dialog.msa_hint.isHidden()
        dialog.search.setChecked(False)
        assert not dialog.start_button.isEnabled()
        dialog.msa.setText(str(root / 'fold.a3m'))
        assert dialog.start_button.isEnabled()
        dialog.output.setText(str(out / 'jobs'))
        engine = dialog.executable.text()
        dialog.executable.setText(str(out / 'missing-engine'))
        dialog.start_button.click()
        assert dialog.progress_label.text() == 'Could not start'
        assert 'not installed' in dialog.status.toPlainText()
        assert 'bold' in dialog.status.styleSheet()
        assert dialog.job is None
        stub = out / 'progress-engine'
        stub.write_text("#!/bin/sh\nprintf 'search: chain search 1 of 2: queued\\n'\n"
                        "sleep 1\nprintf 'AF2 multimer: 195 residues, 512 MSA rows, 1024 extra, 4 passes\\n'\n"
                        "printf '  pass 1/4: mean pLDDT 90\\n'\nexec /bin/sleep 60\n")
        stub.chmod(0o755)
        dialog.executable.setText(str(stub))
        dialog.start_button.click()
        assert dialog.job and dialog.job.state == 'running', dialog.status.toPlainText()
        pump(lambda: 'queued' in dialog.progress_label.text())
        assert dialog.progress_bar.maximum() == 0
        assert dialog.cancel_button.isEnabled() and not dialog.start_button.isEnabled()
        dialog.grab().save(str(out / 'queued.png'))
        pump(lambda: dialog.progress_bar.maximum() == 4)
        assert dialog.progress_bar.value() == 1
        dialog.grab().save(str(out / 'inference.png'))
        dialog.cancel_button.click()
        pump(lambda: not dialog.timer.isActive())
        assert dialog.job.state == 'cancelled'
        assert dialog.progress_label.text() == 'Cancelled'
        assert dialog.progress_bar.maximum() == 100 and dialog.progress_bar.value() == 0
        stub.write_text('#!/bin/sh\nprintf "controlled engine error\\n"\nexit 7\n')
        dialog.start_button.click()
        pump(lambda: not dialog.timer.isActive())
        assert dialog.job.state == 'failed'
        assert dialog.progress_label.text() == 'Failed'
        assert 'status 7' in dialog.status.toPlainText()
        assert dialog.progress_bar.value() == 0 and dialog.start_button.isEnabled()

        results = []
        for mode in ('sequence', 'region'):
            if mode == 'region':
                dialog.close()
                plot = structure._alphalens_plot
                plot._rectangle_select((118, 10), (184, 97))
                dialog = ui.PredictionDialog(plot)
                dialog.show()
                dialog.flanks.setValue(5)
                assert dialog.reuse.isChecked()
            dialog.executable.setText(engine)
            dialog.output.setText(str(out / mode))
            started = time.monotonic()
            dialog.start_button.click()
            assert dialog.job.state == 'running', dialog.status.toPlainText()
            observations = []
            timer = QTimer(); timer.setInterval(40)
            timer.timeout.connect(lambda: observations.append((time.monotonic(),
                dialog.progress_label.text(), dialog.progress_bar.value(), dialog.progress_bar.maximum())))
            timer.start()
            pump(lambda: not dialog.timer.isActive())
            timer.stop()
            assert dialog.job.state == 'complete', dialog.status.toPlainText()
            assert dialog.output_model is not None, dialog.status.toPlainText()
            assert (dialog.comparison is not None) == (mode == 'region')
            assert dialog.progress_bar.value() == dialog.progress_bar.maximum() == 100
            assert dialog.progress_label.text() == 'Complete'
            assert dialog.status.height() == initial_status_height
            assert any(0 < value < total == 4 for _, _, value, total in observations), observations
            assert max(b[0]-a[0] for a, b in zip(observations, observations[1:])) < 2
            assert not dialog.status.styleSheet(), 'Stale error styling'
            dialog.grab().save(str(out / f'{mode}-complete.png'))
            results.append(dict(mode=mode, seconds=time.monotonic()-started,
                                iptm=dialog.job.result['iptm'], progress=observations))
        dialog.close(); dialog = None
        (out / 'results.json').write_text(json.dumps(dict(result='PASS', runs=results), indent=2))
        print('PREDICTION PROGRESS GUI CHECK PASSED', file=sys.__stdout__, flush=True)
        os._exit(0)
    except Exception:
        if dialog is not None:
            dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__)
        os._exit(1)


QTimer.singleShot(1000, main)
