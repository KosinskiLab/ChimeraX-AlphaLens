"""Opt-in real Metal/GUI test of AF2/AF3, three-chain input and contact mapping.

Uses public barnase/barstar sequences, splitting barstar into two fragments only
as a three-chain plumbing test. This is not a biological validation experiment.
ALPHALENS_PILOT, ALPHALENS_TEST_OUTPUT, ALPHALENS_SOURCE follow the progress check.
AF3 weights must already be available and their terms accepted by the operator.
"""
from pathlib import Path
import importlib, importlib.util, json, os, sys, time, traceback
from Qt.QtCore import QTimer
import numpy as np

out = Path(os.environ['ALPHALENS_TEST_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
root = Path(os.environ['ALPHALENS_PILOT'])
if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
    pkg = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1] / 'src'
    pkg = 'chimerax.alphalens_models_check'
    spec = importlib.util.spec_from_file_location(pkg, src/'__init__.py', submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec); sys.modules[pkg] = module; spec.loader.exec_module(module)
ui = importlib.import_module(pkg+'.predict_ui')


def pump(predicate, seconds=90):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        session.ui.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


def run_dialog(dialog):
    started = time.monotonic()
    ticks = []
    timer = QTimer(); timer.setInterval(50)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    before_seed = dialog.seed.value()
    dialog.start_button.click()
    assert dialog.job and dialog.job.state == 'running', dialog.status.toPlainText()
    assert not dialog.backend.isEnabled() and not dialog.randomize_seed.isEnabled()
    assert dialog.job.options['seed'] != before_seed
    pump(lambda: not dialog.timer.isActive())
    timer.stop()
    assert dialog.job.state == 'complete', dialog.status.toPlainText()
    assert dialog.output_model is not None, dialog.status.toPlainText()
    assert dialog.job.result['has_contact_probabilities']
    assert dialog.backend.isEnabled() and not dialog.seed.isEnabled()
    assert ticks and max(b-a for a,b in zip(ticks,ticks[1:])) < 2
    pae = dialog.output_model.alphafold_pae
    assert pae.has_contact and np.isfinite(pae.metrics['contact']).all()
    report = dialog.job.contact_report
    assert report and report['strongest']
    assert Path(report['csv']).is_file()
    assert 'Top-10 mean contact P:' in dialog.status.toPlainText()
    assert all(0 <= item['probability'] <= 1 for item in report['strongest'])
    return dict(backend=dialog.job.backend, seconds=time.monotonic()-started,
                chains=len(dialog.job.chains), seed=dialog.job.options['seed'],
                iptm=dialog.job.result['iptm'], top10_mean=report['prediction']['top10_mean'],
                directory=str(dialog.job.directory), heartbeat_ticks=len(ticks))


def main():
    dialogs = []
    try:
        seq = [s['protein']['sequence'] for s in json.loads((root/'input.json').read_text())['sequences']]
        sequences = [seq[0], seq[1][:43], seq[1][43:]]
        dialog = ui.PredictionDialog(session=session); dialogs.append(dialog); dialog.show()
        dialog.input_text.setPlainText('\n'.join(f'>{i}\n{s}' for i,s in enumerate(sequences)))
        dialog.load_button.click(); pump(lambda: not dialog.lookup_timer.isActive())
        assert len(dialog.ranges) == 3
        dialog.msa.setText(str(root/'fold.a3m'))
        dialog.output.setText(str(out/'sequence'))
        assert dialog.randomize_seed.isChecked() and not dialog.seed.isEnabled()
        results = [run_dialog(dialog)]
        assert len(dialog.output_model.chains) == 3
        assert dialog.job.contact_report['pairs_evaluated'] == 108*43+108*44+43*44
        dialog.grab().save(str(out/'af2-three-chain.png'))
        structure = dialog.output_model
        plot = structure._alphalens_plot
        # The Y interval spans A+B; X covers C. Exclude intrachain pairs and all flanks.
        plot._rectangle_select((158, 95), (185, 140))
        region = ui.PredictionDialog(plot); dialogs.append(region); region.show()
        region.flanks.setValue(3)
        assert len(region.prediction_input()[0]) == 3
        assert region.reuse.isChecked()
        region.backend.setCurrentIndex(region.backend.findData('af3'))
        region.af3_terms.setChecked(False)
        assert not region.start_button.isEnabled()
        region.af3_terms.setChecked(True)
        assert region.start_button.isEnabled()
        region.output.setText(str(out/'region'))
        expected_pairs = (140-95+1)*(185-158+1)
        results.append(run_dialog(region))
        assert len(region.output_model.chains) == 3
        report = region.job.contact_report
        assert report['pairs_evaluated'] == expected_pairs
        assert report['source']['top10_mean'] is not None
        assert region.comparison and region.comparison['contacts'] == report
        region.grab().save(str(out/'af3-three-chain-region.png'))
        # Repeated timer updates must not enlarge either model's dialog.
        height = region.height()
        for i in range(100):
            region.set_status(f'Running {i/10:.1f} s'); session.ui.processEvents()
        assert region.height() == height
        for dialog in dialogs: dialog.close()
        (out/'results.json').write_text(json.dumps(dict(result='PASS', runs=results), indent=2))
        print('AF2/AF3 MULTICHAIN CONTACT GUI CHECK PASSED', file=sys.__stdout__, flush=True)
        os._exit(0)
    except Exception:
        for dialog in dialogs: dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__)
        os._exit(1)

QTimer.singleShot(1000, main)
