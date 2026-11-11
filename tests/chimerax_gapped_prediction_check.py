"""Discontinuous regions end to end, in a real ChimeraX window (GPU, LocalFold 837a15c or newer).

    ALPHALENS_PILOT=<barnase-barstar LocalFold folder: fold.pdb, fold_confidences.json, fold.a3m,
    input.json> ALPHALENS_TEST_OUTPUT=<new folder> ChimeraX --cmd "open tests/chimerax_gapped_prediction_check.py"

Cuts barnase residues 36-45 out of the pilot (structure, PAE and MSA), as an AlphaPulldown model
of two regions would be, and opens it. Then, with AF2 and AF3: drags a box across the gap against
barstar, re-predicts it reusing the cropped source MSA, and checks that the chain went to
LocalFold as one chain with residueIds, came back numbered 21-35, 46-61 with the gap's ends
apart, and opened with its plot. Last, sequence input with two ranges (1-35,46-108).

Environment:
  ALPHALENS_SOURCE            'checkout' (default) tests src/ of this checkout; 'installed' the bundle
  ALPHALENS_LOCALFOLD(_AF3)   engines to use (default: the managed ones; they must be 837a15c or newer)

Reference (M5 Max): C35-N46 15.0 A (AF2), 13.9 A (AF3), 14.0 A (sequence input); 2-6 s a run.
Results in <output>/results.json; exit status 0 when every check passed.
"""
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from chimerax.core.commands import run

SRC = Path(__file__).resolve().parents[1] / 'src'
OUT = Path(os.environ['ALPHALENS_TEST_OUTPUT']).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=False)
PILOT = Path(os.environ['ALPHALENS_PILOT']).expanduser()
CUT = range(36, 46)
if os.environ.get('ALPHALENS_SOURCE', 'checkout') != 'installed':
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
predict = importlib.import_module('chimerax.alphalens.predict')
ui = importlib.import_module('chimerax.alphalens.predict_ui')
from chimerax.atomic import AtomicStructure

results = []


def check(name, ok, detail=''):
    results.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300]))


def pump(pred, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    return False


def gapped_source(out):
    """The pilot without barnase 36-45: structure, confidences and MSA, numbering kept."""
    out.mkdir()
    atom_keep, tokens, lines = [], [], []
    for line in (PILOT / 'fold.pdb').read_text().splitlines():
        if line.startswith('ATOM'):
            key = (line[21], int(line[22:26]))
            keep = not (key[0] == 'A' and key[1] in CUT)
            atom_keep.append(keep)
            if not tokens or tokens[-1][0] != key:
                tokens.append((key, keep))
            if not keep:
                continue
        lines.append(line)
    (out / 'fold.pdb').write_text('\n'.join(lines) + '\n')
    kept = [k for _, k in tokens]
    conf = json.loads((PILOT / 'fold_confidences.json').read_text())
    for key in ('atom_chain_ids', 'atom_plddts'):
        conf[key] = [x for x, k in zip(conf[key], atom_keep) if k]
    for key in ('token_chain_ids', 'token_res_ids'):
        conf[key] = [x for x, k in zip(conf[key], kept) if k]
    conf['pae'] = [[v for v, kj in zip(row, kept) if kj] for row, ki in zip(conf['pae'], kept) if ki]
    (out / 'fold_confidences.json').write_text(json.dumps(conf))
    shutil.copy(PILOT / 'fold_summary_confidences.json', out)
    a, b = [e['protein']['sequence'] for e in json.loads((PILOT / 'input.json').read_text())['sequences']]
    predict.crop_a3m(PILOT / 'fold.a3m', out / 'fold.a3m', a + b, [i for i, k in enumerate(kept) if k])
    return out, a, b


def atoms(pdb):
    found = {}
    for line in Path(pdb).read_text().splitlines():
        if line.startswith('ATOM'):
            found[(line[21], int(line[22:26]), line[12:16].strip())] = [float(line[30:38]), float(line[38:46]),
                                                                          float(line[46:54])]
    return found


def finish(dialog, label):
    dialog.start_button.click()
    pump(lambda: dialog.job is not None and dialog.job.state != 'running' and not dialog.timer.isActive(), 600)
    job = dialog.job
    check(f'{label}: job complete', job is not None and job.state == 'complete', job and job.error)
    return job


def expect_gapped(label, job, numbers, gap):
    chain = job.chains[0]
    check(f'{label}: residue_ids', chain.get('residue_ids') == numbers, chain.get('residue_ids'))
    sent = json.loads((job.directory / 'input.json').read_text())['sequences']
    check(f'{label}: residueIds sent to LocalFold', sent[0]['protein'].get('residueIds') == numbers)
    check(f'{label}: the continuous chain sends none', 'residueIds' not in sent[1]['protein'])
    numbering = job.manifest.get('numbering') or {}
    cid = (numbering.get('chain_ids') or {}).get('A', 'A')
    found = atoms(job.directory / 'fold.pdb')
    got = sorted({k[1] for k in found if k[0] == cid})
    check(f'{label}: structure in the source numbering', got == numbers, got)
    if numbering.get('scheme') == 'source':
        check(f'{label}: engine files kept', (job.directory / 'engine_output' / 'fold.pdb').is_file())
    gap_distance = math.dist(found[(cid, gap[0], 'C')], found[(cid, gap[1], 'N')])
    check(f'{label}: gap ends apart (C{gap[0]}-N{gap[1]} {gap_distance:.1f} A)', gap_distance > 4, gap_distance)


def main():
    code = 1
    try:
        source, a, b = gapped_source(OUT / 'source')
        run(session, f'alphalens open "{source}"')
        pump(lambda: any(getattr(s, '_alphalens_plot', None) is not None and s._alphalens_plot._load_progress is None
                         for s in session.models.list(type=AtomicStructure)), 120)
        s = next(s for s in session.models.list(type=AtomicStructure) if getattr(s, '_alphalens_plot', None))
        plot = s._alphalens_plot
        n_a = len(s.chains[0].residues)
        check('gapped source opened', n_a == len(a) - len(CUT) and
              len(plot.pae.row_residues_or_atoms()) == len(a) + len(b) - len(CUT))
        plot._set_prediction_region((25, 45, n_a + 10, n_a + 40))       # A:26-35 + A:46-56 against B
        for backend in ('af2', 'af3'):
            dialog = plot._predict_interface()
            dialog.backend.setCurrentIndex(dialog.backend.findData(backend))
            dialog.flanks.setValue(5)
            dialog.output.setText(str(OUT / backend))
            check(f'{backend}: source MSA reused', dialog.reuse.isChecked())
            check(f'{backend}: dialog shows the segments', 'in 2 segments' in dialog.preview.text(),
                  dialog.preview.text())
            check(f'{backend}: Predict, no engine setup', dialog.start_button.text() == 'Predict',
                  dialog.start_button.text())
            dialog.grab().save(str(OUT / f'{backend}-dialog.png'))
            job = finish(dialog, backend)
            if job.state == 'complete':
                expect_gapped(backend, job, list(range(21, 36)) + list(range(46, 62)), (35, 46))
                check(f'{backend}: result opened with its plot', dialog.output_model is not None and
                      getattr(dialog.output_model, '_alphalens_plot', None) is not None)
                check(f'{backend}: comparison written', (job.directory / 'comparison.json').is_file())
            dialog.close()
            pump(lambda: False, 0.3)
        # Sequence input, two ranges in one chain, with the matching cut MSA
        dialog = ui.show_prediction_dialog(session=session)
        dialog.input_text.setPlainText(f'>barnase\n{a}\n>barstar\n{b}\n')
        dialog.load_button.click()
        pump(lambda: not dialog.lookup_timer.isActive(), 30)
        dialog.ranges[0].setText('1-35,46-108')
        dialog.msa.setText(str(source / 'fold.a3m'))
        dialog.output.setText(str(OUT / 'sequence'))
        check('sequence: dialog shows the segments', 'A: 98 residues in 2 segments' in dialog.preview.text(),
              dialog.preview.text())
        job = finish(dialog, 'sequence')
        if job.state == 'complete':
            expect_gapped('sequence', job, list(range(1, 36)) + list(range(46, 109)), (35, 46))
        dialog.close()
        code = 0 if all(r['ok'] for r in results) else 1
    except Exception:
        results.append(dict(name='exception', ok=False, detail=traceback.format_exc()))
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    for r in results:
        session.logger.info(('PASS ' if r['ok'] else 'FAIL ') + r['name'] + ('' if r['ok'] else f"  {r['detail']}"))
    os._exit(code)


main()
