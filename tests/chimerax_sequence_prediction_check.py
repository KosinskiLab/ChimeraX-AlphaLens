"""Opt-in real-GPU GUI test for FASTA, raw sequence and a live UniProt lookup.

Use the same public ALPHALENS_PILOT and new ALPHALENS_TEST_OUTPUT directories as
chimerax_local_prediction_check.py. Runs actual monomer/multimer predictions and
exits this disposable test application. UniProt receives RNBR_BACAM; no sequence
upload is needed because the known pilot's alignments are reused.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from Qt.QtCore import QTimer
from chimerax.core.commands import run
from chimerax.alphalens import predict

root = Path(os.environ['ALPHALENS_PILOT'])
output = Path(os.environ['ALPHALENS_TEST_OUTPUT'])
output.mkdir(parents=True, exist_ok=False)
sequences = [s['protein']['sequence'] for s in json.loads((root/'input.json').read_text())['sequences']]
mono_msa = output/'monomer.a3m'
predict.crop_a3m(root/'fold.a3m', mono_msa, ''.join(sequences), list(range(len(sequences[0]))))


def pump(predicate, seconds=300):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        session.ui.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


def main():
    results=[]
    try:
        # Refresh metadata after installing a candidate in this disposable profile.
        session.toolshed.reload(session.logger, rebuild_cache=True)
        for case in ('fasta_pair', 'raw_monomer', 'uniprot_monomer'):
            # Exercise actual lazy command registration; no source model is needed.
            run(session, 'ui tool show "AlphaLens Prediction"' if case == 'fasta_pair' else 'alphalens predict')
            dialog = session._alphalens_prediction_dialogs[-1]
            dialog.output.setText(str(output/case))
            if case == 'fasta_pair':
                text = f'>barnase\n{sequences[0]}\n>barstar\n{sequences[1]}'
                dialog.msa.setText(str(root/'fold.a3m'))
            elif case == 'raw_monomer':
                text = sequences[0].lower()
                dialog.msa.setText(str(mono_msa))
            else:
                text = 'RNBR_BACAM'
                dialog.ranges[0].setText('50-157')
                dialog.search.setChecked(True)  # The monomer MSA from the previous case is cached.
            dialog.input_text.setPlainText(text)
            assert not dialog.start_button.isEnabled(), 'Unresolved input could be submitted'
            dialog.load_button.click()
            pump(lambda: not dialog.lookup_timer.isActive(), seconds=50)
            assert dialog.entries, dialog.status.toPlainText()
            assert dialog.start_button.isEnabled(), dialog.preview.text()
            if case == 'uniprot_monomer':
                assert dialog.entries[0]['accession'] == 'P00648'
                assert dialog.prediction_input()[0][0]['sequence'] == sequences[0]
            started=time.monotonic()
            dialog.start_button.click()
            assert dialog.job and dialog.job.state=='running',dialog.status.toPlainText()
            ticks=[]
            heartbeat=QTimer(); heartbeat.setInterval(50); heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
            heartbeat.start()
            pump(lambda: not dialog.timer.isActive())
            heartbeat.stop()
            assert dialog.job.state=='complete',dialog.status.toPlainText()
            assert dialog.output_model is not None,dialog.status.toPlainText()
            assert dialog.comparison is None
            assert ticks
            if case != 'fasta_pair':
                assert dialog.job.model == 'model_1_ptm'
                assert dialog.job.result['iptm'] is None
                assert dialog.output_model.alphafold_pae.matrix_size == 108
            if case == 'uniprot_monomer':
                assert dialog.job.manifest['msa_source'] == 'cache'
                assert dialog.job.chains[0]['residue_map'][0]['number'] == 50
                assert 'pTM' in dialog.status.toPlainText() and 'ipTM' not in dialog.status.toPlainText()
            result=dict(case=case,elapsed_seconds=time.monotonic()-started,model=dialog.job.model,
                        result=dialog.job.result,job_directory=str(dialog.job.directory),heartbeat_ticks=len(ticks))
            results.append(result)
            session.ui.processEvents()
            dialog.grab().save(str(output/f'{case}-dialog.png'))
            dialog.output_model._alphalens_plot.tool_window.ui_area.grab().save(str(output/f'{case}-plot.png'))
            dialog.close()
        # Escape must cancel the process, not merely hide a still-running job.
        run(session, 'alphalens predict')
        dialog = session._alphalens_prediction_dialogs[-1]
        dialog.input_text.setPlainText(sequences[0])
        dialog.load_button.click()
        pump(lambda: not dialog.lookup_timer.isActive())
        dialog.msa.setText(str(mono_msa))
        dialog.output.setText(str(output/'cancelled'))
        dialog.start_button.click()
        assert dialog.job.state == 'running'
        dialog.reject()
        pump(lambda: not dialog.timer.isActive())
        assert dialog.job.state == 'cancelled'
        assert dialog.job.process.poll() is not None
        dialog.close()
        (output/'gui_results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2),file=sys.__stdout__,flush=True)
    except Exception:
        for dialog in getattr(session,'_alphalens_prediction_dialogs',[]): dialog.shutdown()
        traceback.print_exc(file=sys.__stderr__)
        os._exit(1)
    print('AlphaLens sequence/UniProt GUI prediction test PASSED',file=sys.__stdout__,flush=True)
    os._exit(0)

QTimer.singleShot(1000,main)
