"""Opt-in corrected AF3 re-prediction and report persistence in a disposable GUI.

ALPHALENS_QUALITY_JOB points to a saved region job with its source structure and MSA.
ALPHALENS_SOURCE selects checkout or installed; ALPHALENS_TEST_OUTPUT is a new folder.
"""
import importlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from Qt.QtCore import QTimer
from Qt.QtWidgets import QPlainTextEdit,QMenu
from chimerax.core.commands import run,quote_path_if_necessary as quote
out=Path(os.environ['ALPHALENS_TEST_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
old=Path(os.environ['ALPHALENS_QUALITY_JOB'])
if os.environ.get('ALPHALENS_SOURCE','checkout')=='installed':pkg='chimerax.alphalens'
else:
    src=Path(__file__).resolve().parents[1]/'src';pkg='chimerax.alphalens_quality_check'
    spec=importlib.util.spec_from_file_location(pkg,src/'__init__.py',submodule_search_locations=[str(src)])
    module=importlib.util.module_from_spec(spec);sys.modules[pkg]=module;spec.loader.exec_module(module)
ui=importlib.import_module(pkg+'.predict_ui');cmd=importlib.import_module(pkg+'.cmd')

def pump(predicate,timeout=180):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        session.ui.processEvents()
        if predicate():return
        time.sleep(.01)
    raise AssertionError('Timed out')

def main():
    dialog=None
    try:
        saved=json.loads((old/'job.json').read_text())
        source=cmd._open_structure(session,saved['source']['file'])
        cmd.alphalens(session,source,wait=True)
        plot=source._alphalens_plot;plot._prediction_region=tuple(saved['source']['region'])
        dialog=ui.PredictionDialog(plot);dialog.show()
        dialog.backend.setCurrentIndex(dialog.backend.findData('af3'))
        dialog.af3_terms.setChecked(True)
        assert dialog.recycles.value()==10 and dialog.af3_samples.value()==5
        dialog.randomize_seed.setChecked(False);dialog.seed.setValue(saved['options']['seed'])
        dialog.reuse.setChecked(False);dialog.search.setChecked(False)
        dialog.msa.setText(str(old/'fold.a3m'))
        dialog.output.setText(str(out/'jobs'))
        logged=[]
        original_info=session.logger.info
        def record(message,*args,**kwargs):
            logged.append(str(message));return original_info(message,*args,**kwargs)
        session.logger.info=record
        stages=[]
        observer=QTimer();observer.setInterval(50)
        observer.timeout.connect(lambda: stages.append(dialog.progress_label.text()))
        observer.start()
        dialog.start_button.click()
        assert dialog.job and dialog.job.state=='running',dialog.status.toPlainText()
        pump(lambda:not dialog.timer.isActive())
        observer.stop()
        assert dialog.job.state=='complete',dialog.status.toPlainText()
        assert dialog.output_model is not None,dialog.status.toPlainText()
        assert dialog.job.result['geometry']['ca_pairs_under_2A']==0
        assert dialog.job.result['mean_plddt']>65
        assert dialog.job.result['iptm']>.4
        assert dialog.report_button.isEnabled()
        text=(dialog.job.directory/'prediction_report.txt').read_text(encoding='utf-8').strip()
        assert any('Predicting with AF3' in stage for stage in stages),stages
        assert dialog.status.toPlainText().endswith(text)
        assert dialog.status.firstVisibleBlock().text().startswith('Local prediction')
        assert logged.count(text)==1
        assert 'Same residues: pLDDT' in text and 'Same selected pairs:' in text
        dialog.report_button.click()
        report=session._alphalens_prediction_reports[-1]
        assert report.findChild(QPlainTextEdit).toPlainText()==text
        report.close()
        result_plot=dialog.output_model._alphalens_plot
        jobdir=str(dialog.job.directory);comparison=dialog.comparison
        runs=[dict(directory=jobdir,seed=dialog.job.options['seed'],
                   seconds=json.loads((dialog.job.directory/'job.json').read_text())['elapsed_seconds'],
                   result=dialog.job.result)]
        models=[dialog.output_model]
        # Reuse the same dialog to exercise timer, process-lock and control cleanup.
        for repeat in range(1,int(os.environ.get('ALPHALENS_QUALITY_REPEATS','1'))):
            assert dialog.start_button.isEnabled() and not dialog.timer.isActive()
            dialog.randomize_seed.setChecked(True)
            dialog.start_button.click()
            assert dialog.job.state=='running',dialog.status.toPlainText()
            pump(lambda:not dialog.timer.isActive())
            assert dialog.job.state=='complete',dialog.status.toPlainText()
            assert dialog.output_model not in models
            assert not any(model.deleted for model in models)
            models.append(dialog.output_model)
            assert str(dialog.job.directory) not in [r['directory'] for r in runs]
            assert dialog.job.options['seed'] not in [r['seed'] for r in runs]
            assert dialog.report_directory==dialog.job.directory
            assert (Path(jobdir)/'prediction_report.txt').read_text(encoding='utf-8').strip()==text
            assert text in dialog.status.toPlainText()
            dialog.report_button.click();report=session._alphalens_prediction_reports[-1]
            assert f"Seed: {dialog.job.options['seed']}" in report.findChild(QPlainTextEdit).toPlainText()
            report.close()
            runs.append(dict(directory=str(dialog.job.directory),seed=dialog.job.options['seed'],
                             seconds=json.loads((dialog.job.directory/'job.json').read_text())['elapsed_seconds'],
                             result=dialog.job.result))
        dialog.close();dialog=None
        menu=QMenu();result_plot._fill_context_menu(menu,0,0)
        action=next(a for a in menu.actions() if a.text()=='Prediction report…')
        action.trigger();report=session._alphalens_prediction_reports[-1]
        assert report.findChild(QPlainTextEdit).toPlainText()==text
        report.grab().save(str(out/'persistent-report.png'));report.close()
        # Only registered installed bundles can be serialized into a ChimeraX session.
        if pkg == 'chimerax.alphalens':
            session_path=out/'corrected-AF3.cxs'
            run(session,f'save {quote(str(session_path))}')
            run(session,'close session')
            run(session,f'open {quote(str(session_path))}')
            restored=next(m for m in session.models.list()
                          if getattr(m,'filename',None)==str(Path(jobdir)/'fold.pdb'))
            restored_plot=restored._alphalens_plot
            menu=QMenu();restored_plot._fill_context_menu(menu,0,0)
            next(a for a in menu.actions() if a.text()=='Prediction report…').trigger()
            report=session._alphalens_prediction_reports[-1]
            assert report.findChild(QPlainTextEdit).toPlainText()==text
            report.close()
        (out/'results.json').write_text(json.dumps(dict(result='PASS',job_directory=jobdir,
                                                       comparison=comparison,runs=runs),indent=2))
        print('CORRECTED AF3 AND PERSISTENT REPORT CHECK PASSED '+jobdir,file=sys.__stdout__,flush=True)
        os._exit(0)
    except Exception:
        if dialog:dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__);os._exit(1)
QTimer.singleShot(1000,main)
