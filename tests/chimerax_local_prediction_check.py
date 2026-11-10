"""End-to-end GPU test, in a disposable ChimeraX GUI/profile only.

Set ALPHALENS_PILOT to a real LocalFold prediction directory (fold.pdb,
fold_confidences.json, fold.a3m). Set ALPHALENS_TEST_OUTPUT to a new results directory.
Runs a cached-MSA prediction, then a prediction using a cropped source MSA, then its cached repeat.
ALPHALENS_REMOTE_MSA=1 optionally tests a fresh remote search for the public cropped sequences.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from Qt.QtCore import QTimer
from chimerax.alphalens import cmd

root = Path(os.environ['ALPHALENS_PILOT'])
output = Path(os.environ['ALPHALENS_TEST_OUTPUT'])
output.mkdir(parents=True, exist_ok=False)


def pump(predicate, seconds=180):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        session.ui.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('GUI prediction timed out')


def main():
    try:
        structure = cmd._open_structure(session, str(root/'fold.pdb'))
        pae = cmd.alphalens(session, structure, pae_file=str(root/'fold_confidences.json'), wait=True)
        plot = structure._alphalens_plot
        results=[]
        for case, start, end in (('full', (108, 0), (194, 107)),
                                 ('crop', (118, 10), (184, 97)),
                                 ('cached_crop', (118, 10), (184, 97))):
            # This source has no contact probabilities: region selection must still work.
            plot._rectangle_select(start, end)
            assert plot._prediction_region is not None
            dialog = plot._predict_interface()
            dialog.flanks.setValue(5)
            dialog.output.setText(str(output/case))
            dialog.seed.setValue(4 if case == 'cached_crop' else 3)
            if case == 'full':
                dialog.reuse.setChecked(False)
                dialog.msa.setText(str(root/'fold.a3m'))
            elif case == 'cached_crop' or os.environ.get('ALPHALENS_REMOTE_MSA') == '1':
                dialog.reuse.setChecked(False)
                dialog.search.setChecked(True)
            else:
                assert dialog.reuse.isChecked(), 'Did not detect the source alignment'
            assert dialog.start_button.isEnabled(), dialog.preview.text()
            started = time.monotonic()
            dialog.start_button.click()
            assert dialog.job and dialog.job.state == 'running', dialog.status.text()
            ticks=[]
            heartbeat=QTimer(); heartbeat.setInterval(50); heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
            heartbeat.start()
            if case == 'cached_crop':
                # Closing the original plot must not lose the running job or its residue mapping.
                session.models.close([structure])
            pump(lambda: not dialog.timer.isActive())
            heartbeat.stop()
            assert dialog.job.state == 'complete', dialog.status.text()
            assert dialog.output_model is not None, dialog.status.text()
            assert dialog.comparison['aligned_chain_ca_rmsd'] is not None
            assert dialog.comparison['aligned_chain_ca_rmsd'] < 2
            assert structure.deleted == (case == 'cached_crop')
            assert dialog.output_model.alphafold_pae.matrix_size == sum(len(c['sequence']) for c in dialog.job.chains)
            assert not dialog.output_model._alphalens_plot.closed()
            assert ticks, 'Qt did not remain responsive'
            max_gap = max([b-a for a,b in zip(ticks,ticks[1:])] or [0])
            assert max_gap < 2, f'GUI blocked for {max_gap:.1f} seconds'
            if case == 'cached_crop': assert dialog.job.manifest['msa_source']=='cache'
            results.append(dict(case=case, elapsed_seconds=time.monotonic()-started,
                         residues=sum(len(c['sequence']) for c in dialog.job.chains),
                         result=dialog.job.result, comparison=dialog.comparison,
                         job_directory=str(dialog.job.directory), heartbeat_ticks=len(ticks)))
            session.ui.processEvents()
            dialog.grab().save(str(output/f'{case}-dialog.png'))
            dialog.output_model._alphalens_plot.tool_window.ui_area.grab().save(str(output/f'{case}-plot.png'))
            dialog.close()
        (output/'gui_results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2), file=sys.__stdout__, flush=True)
    except Exception:
        for dialog in getattr(session, '_alphalens_prediction_dialogs', []):
            dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__)
        os._exit(1)
    print('AlphaLens real GUI prediction test PASSED',file=sys.__stdout__,flush=True)
    os._exit(0)

QTimer.singleShot(1000, main)
