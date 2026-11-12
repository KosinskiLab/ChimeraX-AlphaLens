"""First-use setup, cancellation/retry and a real local prediction in an isolated prefix.

ALPHALENS_PILOT: saved barnase/barstar job with input.json and fold.a3m.
ALPHALENS_TEST_OUTPUT: new directory. ALPHALENS_SOURCE: checkout or installed.
ALPHALENS_FIRST_USE_BACKEND: af3 (default) or af2.
This opt-in check downloads source and compiles the selected engine; existing engines and weights are preserved.
"""
from pathlib import Path
import importlib,importlib.util,json,os,sys,time,traceback
from Qt.QtCore import QTimer
out=Path(os.environ['ALPHALENS_TEST_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
pilot=Path(os.environ['ALPHALENS_PILOT'])
if os.environ.get('ALPHALENS_SOURCE')=='installed':pkg='chimerax.alphalens'
else:
    src=Path(__file__).resolve().parents[1]/'src';pkg='chimerax.alphalens_setup_check'
    spec=importlib.util.spec_from_file_location(pkg,src/'__init__.py',submodule_search_locations=[str(src)])
    module=importlib.util.module_from_spec(spec);sys.modules[pkg]=module;spec.loader.exec_module(module)
ui=importlib.import_module(pkg+'.predict_ui');predict=importlib.import_module(pkg+'.predict')
setup=importlib.import_module(pkg+'.predict_setup')
prefix=out/'managed';predict.data_directory=lambda:prefix
backend=os.environ.get('ALPHALENS_FIRST_USE_BACKEND','af3')

def pump(predicate,timeout=600):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        session.ui.processEvents()
        if predicate():return
        time.sleep(.01)
    raise AssertionError('First-use check timed out')

def main():
    dialog=None
    try:
        data=json.loads((pilot/'input.json').read_text())
        sequences=[s['protein']['sequence'] for s in data['sequences']]
        dialog=ui.PredictionDialog(session=session);dialog.show()
        dialog.input_text.setPlainText('\n'.join(f'>chain{i}\n{s}' for i,s in enumerate(sequences)))
        dialog.load_button.click();pump(lambda:not dialog.lookup_timer.isActive())
        dialog.backend.setCurrentIndex(dialog.backend.findData(backend));dialog.af3_terms.setChecked(True)
        dialog.executable.setText(str(prefix/'bin'/f'localfold-{backend}'))
        dialog.msa.setText(str(pilot/'fold.a3m'));dialog.output.setText(str(out/'predictions'))
        dialog.randomize_seed.setChecked(False);dialog.seed.setValue(42)
        assert dialog.start_button.text().replace('&&', '&')=='Install & Predict'
        assert dialog.start_button.isEnabled() and not prefix.exists()
        real_setup=setup.SetupJob
        worker=out/'fake-setup.py'
        worker.write_text('import time\nprint("AlphaLens setup: Downloading source",flush=True)\ntime.sleep(60)\n')
        setup.SetupJob=lambda *a,**k:real_setup(*a,script=worker,**k)
        dialog.start_button.click();pump(lambda:'Downloading source' in dialog.status.toPlainText(),10)
        assert dialog.setup_job and dialog.job is None
        assert not dialog.input_text.isEnabled() and not dialog.backend.isEnabled()
        dialog.cancel_button.click();pump(lambda:not dialog.timer.isActive(),10)
        assert dialog.pending_prediction is None and dialog.job is None
        assert dialog.progress_label.text()=='Setup cancelled' and dialog.start_button.isEnabled()
        worker.write_text('import sys\nprint("controlled setup failure",flush=True)\nsys.exit(7)\n')
        dialog.start_button.click();pump(lambda:not dialog.timer.isActive(),10)
        assert 'controlled setup failure' in dialog.status.toPlainText() and dialog.job is None
        assert dialog.start_button.isEnabled()
        setup.SetupJob=real_setup
        heartbeats=[];timer=QTimer();timer.setInterval(50);timer.timeout.connect(lambda:heartbeats.append(time.monotonic()))
        timer.start();started=time.monotonic()
        dialog.start_button.click();assert dialog.setup_job is not None
        installer=dialog.setup_job
        pump(lambda:not dialog.timer.isActive())
        timer.stop()
        assert dialog.job and dialog.job.state=='complete',dialog.status.toPlainText()
        assert installer.state=='complete' and dialog.output_model is not None
        assert dialog.job.manifest['revision']==setup.REVISION
        assert dialog.job.executable==prefix/'bin'/f'localfold-{backend}'
        assert dialog.job.result['mean_plddt']>90 and dialog.job.result['iptm']>.85
        assert dialog.start_button.text()=='Predict' and not dialog.needs_setup()
        gaps=[b-a for a,b in zip(heartbeats,heartbeats[1:])]
        assert max(gaps)<3,max(gaps)
        report=(dialog.job.directory/'prediction_report.txt').read_text(encoding='utf-8').strip()
        assert report in dialog.status.toPlainText()
        before=len(list((prefix/'setup-logs').glob('*.log')))
        dialog.start_button.click()
        assert dialog.setup_job is None and dialog.job.state=='running'
        dialog.cancel_button.click();pump(lambda:not dialog.timer.isActive(),15)
        assert len(list((prefix/'setup-logs').glob('*.log')))==before
        result=dict(result='PASS',backend=backend,total_seconds=time.monotonic()-started,heartbeat_max_gap=max(gaps),
                    setup_log=str(installer.log_path),prediction_report=report,managed_prefix=str(prefix))
        (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        dialog.grab().save(str(out/'first-use.png'))
        print('FIRST-USE SETUP CHECK PASSED '+str(out),file=sys.__stdout__,flush=True)
        dialog.close();os._exit(0)
    except Exception:
        if dialog:dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__);os._exit(1)
QTimer.singleShot(500,main)
