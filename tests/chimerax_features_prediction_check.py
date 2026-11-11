"""Real prediction from precomputed AlphaPulldown features, in a real ChimeraX window (GPU, LocalFold).

    ALPHALENS_FEATURES_SOURCE=<AF2/AF3 prediction folder named like Q15014+Q9NV56> \\
    ALPHALENS_TEST_OUTPUT=<new folder> ChimeraX --cmd "open tests/chimerax_features_prediction_check.py"

Opens the prediction, drags a box around its strongest inter-chain contact (or, with
ALPHALENS_FEATURES_WHOLE=1, the whole chain A x chain B block), checks that the dialog took the
chain IDs from the folder name, re-predicts from the features and checks the job: features found
and paired by species, the result opened, the window responsive throughout.

Environment:
  ALPHALENS_FEATURES_ROOT     feature folder or URL (default: the dialog's, /g/alphafold/input_features
                              when /g is mounted, else https://s3.embl.de/alphapulldown)
  ALPHALENS_FEATURES_BACKEND  'af2' (default) or 'af3' (needs the AF3 terms accepted)
  ALPHALENS_SOURCE            'checkout' (default) tests src/ of this checkout; 'installed' the bundle

Reference (M5 Max, lab database, Q15014+Q9NV56 whole block): AF2-Multimer 11 s, ipTM 0.595,
86 -> 86 contacts at P >= 0.5, CA RMSD 0.23 A to the AlphaPulldown model over pLDDT >= 70.
Results in <output>/results.txt; exit status 0 when it passed.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from Qt.QtCore import QTimer
from chimerax.core.commands import run

SRC = Path(__file__).resolve().parents[1] / 'src'
OUT = Path(os.environ['ALPHALENS_TEST_OUTPUT']).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=False)
PRED = os.environ['ALPHALENS_FEATURES_SOURCE']
ROOT = os.environ.get('ALPHALENS_FEATURES_ROOT')
if os.environ.get('ALPHALENS_SOURCE', 'checkout') != 'installed':
    # This checkout under the bundle's own name, so its modules import each other as installed
    for k in list(sys.modules):
        if k == 'chimerax.alphalens' or k.startswith('chimerax.alphalens.'):
            del sys.modules[k]
    spec = importlib.util.spec_from_file_location('chimerax.alphalens', SRC / '__init__.py',
                                                  submodule_search_locations=[str(SRC)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules['chimerax.alphalens'] = pkg
    spec.loader.exec_module(pkg)
    import chimerax
    chimerax.alphalens = pkg
    importlib.import_module('chimerax.alphalens.cmd').register_commands(session.logger)
from chimerax.atomic import AtomicStructure

log = []


def note(*a):
    log.append(' '.join(str(x) for x in a))
    (OUT / 'results.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')


def pump(pred, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    return False


def main():
    code = 1
    try:
        run(session, f'alphalens open "{PRED}"')
        assert pump(lambda: any(getattr(s, '_alphalens_plot', None) is not None
                                and s._alphalens_plot._load_progress is None
                                for s in session.models.list(type=AtomicStructure)), 600)
        s = next(s for s in session.models.list(type=AtomicStructure) if getattr(s, '_alphalens_plot', None))
        plot = s._alphalens_plot
        if os.environ.get('ALPHALENS_SOURCE', 'checkout') != 'installed':
            assert Path(sys.modules['chimerax.alphalens.tool'].__file__).parent == SRC
        from chimerax.alphalens import contacts
        pairs, probs = contacts.contact_pairs(plot.pae, 0.5)
        i, j = (int(v) for v in pairs[0])
        n = plot.pae.matrix_size
        y0, y1, x0, x1 = max(0, i - 20), min(n - 1, i + 20), max(0, j - 20), min(n - 1, j + 20)
        if os.environ.get('ALPHALENS_FEATURES_WHOLE') == '1':      # The whole A x B block: full-length chains
            from chimerax.alphalens.contacts import row_chain_ids
            ids = row_chain_ids(plot.pae)
            a = [k for k, c in enumerate(ids) if c == 'A']; b = [k for k, c in enumerate(ids) if c == 'B']
            y0, y1, x0, x1 = a[0], a[-1], b[0], b[-1]
        plot._rectangle_select((x0, y0), (x1, y1))
        assert plot._prediction_region is not None
        dialog = plot._predict_interface()
        note('prefilled feature IDs:', repr(dialog.features_names.text()), 'root:', dialog.features_root.text())
        from chimerax.alphalens.predict_features import accessions_in
        expected = ', '.join(f'{c}: {a}' for c, a in zip('ABCDEFGH', accessions_in(Path(PRED).name)))
        assert dialog.features_names.text() == expected, dialog.features_names.text()
        dialog.features.setChecked(True)
        if os.environ.get('ALPHALENS_FEATURES_BACKEND') == 'af3':
            dialog.backend.setCurrentIndex(dialog.backend.findData('af3'))
            dialog.af3_terms.setChecked(True)
            dialog.af3_samples.setValue(1)
        assert not dialog.reuse.isChecked() and not dialog.search.isChecked() and not dialog.msa.isEnabled()
        if ROOT:
            dialog.features_root.setText(ROOT)
        dialog.flanks.setValue(5)
        dialog.output.setText(str(OUT / 'predictions'))
        dialog.randomize_seed.setChecked(False)
        dialog.seed.setValue(7)
        assert dialog.start_button.isEnabled(), dialog.preview.text()
        note('preview:', dialog.preview.text())
        t0 = time.monotonic()
        dialog.start_button.click()
        assert dialog.job and dialog.job.state == 'running', dialog.status.toPlainText()
        ticks = []
        hb = QTimer(); hb.setInterval(50); hb.timeout.connect(lambda: ticks.append(time.monotonic())); hb.start()
        pump(lambda: not dialog.timer.isActive(), 1800)
        hb.stop()
        job = dialog.job
        note('state:', job.state, 'elapsed %.1f s' % (time.monotonic() - t0))
        note('status:\n' + dialog.status.toPlainText())
        assert job.state == 'complete', job.error
        assert job.manifest['msa_source'].startswith('precomputed features from'), job.manifest['msa_source']
        feats = job.manifest['features']
        note('features:', json.dumps(feats))
        assert [f['name'] for f in feats] == accessions_in(Path(PRED).name)[:len(feats)]
        logtext = (job.directory / 'prediction.log').read_text(errors='replace')
        note('log head:\n' + '\n'.join(logtext.splitlines()[:14]))
        assert 'features: ' in logtext and 'rows paired by species' in logtext
        assert dialog.output_model is not None and not dialog.output_model._alphalens_plot.closed()
        note('result:', json.dumps({k: job.result.get(k) for k in ('ptm', 'iptm', 'mean_plddt', 'mean_cross_chain_pae')}))
        note('comparison rmsd:', dialog.comparison and dialog.comparison.get('aligned_chain_ca_rmsd'))
        gap = max([b - a for a, b in zip(ticks, ticks[1:])] or [0])
        note('max GUI gap %.2f s over %d ticks' % (gap, len(ticks)))
        assert gap < 2, f'GUI blocked for {gap:.1f} s'
        session.ui.processEvents()
        dialog.grab().save(str(OUT / 'dialog.png'))
        code = 0
        note('PASSED')
    except Exception:
        note('EXCEPTION ' + traceback.format_exc())
        for d in getattr(session, '_alphalens_prediction_dialogs', []):
            try:
                d.shutdown()
            except Exception:
                pass
    os._exit(code)


QTimer.singleShot(1000, main)
