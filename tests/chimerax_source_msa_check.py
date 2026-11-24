"""Re-predicting from the source prediction's own MSA, for each layout, in a real ChimeraX window (GPU).

    ALPHALENS_MSA_FOLDS="<prediction>;<prediction>;..." ALPHALENS_TEST_OUTPUT=<new folder> \
        ChimeraX --cmd "open tests/chimerax_source_msa_check.py"

Each prediction is a folder (opened with alphalens open) or a structure file (such as an AF3
seed-<S>_sample-<K>/model.cif). The layouts:
  * a merged A3M beside the structure: LocalFold fold.a3m or input.a3m, ColabFold <job>.a3m;
  * a native AlphaFold 2 run's features.pkl with msas/chain_id_map.json (its merged MSA);
  * an AlphaFold Server download's msas folder (per-chain unpaired and paired A3Ms);
  * an AlphaFold 3 <name>_data.json(.xz) beside the model or above a sample's folder;
  * AlphaPulldown features in <out>/features of an AlphaPulldownSnakemake <out>/predictions/<fold>.
For each: drags a box around the most confident inter-chain pair (lowest PAE), checks that the
prediction dialog offers the MSA to reuse, then with AF2 and with AF3 re-predicts the box from it
and checks what went to LocalFold (a cropped merged A3M; per-chain and species-paired A3Ms for
AF2; cropped unpairedMsa and pairedMsa in the AF3 job) and that the result opened with DockQ
against the source.

Environment:
  ALPHALENS_NO_MSA_FOLDS  predictions without one (AlphaPulldown AF2): reuse must be off, with
                          the reason as its tooltip
  ALPHALENS_SOURCE        'checkout' (default) tests src/ of this checkout; 'installed' the bundle
  ALPHALENS_BACKENDS      backends to run (default 'af2,af3')
  ALPHALENS_BOX_HALF      residues on each side of the pair in the box, within its chains (default 20)

Results in <output>/results.json; exit status 0 when every check passed.
"""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from chimerax.core.commands import run

SRC = Path(__file__).resolve().parents[1] / 'src'
OUT = Path(os.environ['ALPHALENS_TEST_OUTPUT']).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=False)
FOLDS = [Path(p).expanduser() for p in os.environ.get('ALPHALENS_MSA_FOLDS', '').split(';') if p]
NO_MSA = [Path(p).expanduser() for p in os.environ.get('ALPHALENS_NO_MSA_FOLDS', '').split(';') if p]
BACKENDS = os.environ.get('ALPHALENS_BACKENDS', 'af2,af3').split(',')
HALF = int(os.environ.get('ALPHALENS_BOX_HALF', '20'))
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
from chimerax.atomic import AtomicStructure, Residue
import numpy as np

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


def open_prediction(path):
    run(session, 'close')
    t0 = time.monotonic()
    run(session, f'alphalens open "{path}"')
    pump(lambda: any(getattr(s, '_alphalens_plot', None) is not None and s._alphalens_plot._load_progress is None
                     for s in session.models.list(type=AtomicStructure)), 600)
    s = next(s for s in session.models.list(type=AtomicStructure) if getattr(s, '_alphalens_plot', None))
    return s, time.monotonic() - t0


def best_box(plot, half):
    '''The box (as dragged: (column, row) corners) around the inter-chain protein pair of lowest PAE.'''
    m = np.asarray(plot.pae.matrix('pae'), float)
    tokens = plot.pae.row_residues_or_atoms()
    chains = np.array([r.chain_id for r in tokens])
    protein = np.array([isinstance(r, Residue) and r.polymer_type == Residue.PT_AMINO for r in tokens])
    pae = (m + m.T) / 2
    pae[chains[:, None] == chains[None, :]] = np.inf
    pae[~protein, :] = pae[:, ~protein] = np.inf
    i, j = np.unravel_index(np.argmin(pae), pae.shape)

    def span(k):
        same = np.flatnonzero(chains == chains[k])
        return max(same[0], k - half), min(same[-1], k + half)
    (r0, r1), (c0, c1) = span(i), span(j)
    return (int(c0), int(r0)), (int(c1), int(r1)), chains[i], chains[j], float(pae[i, j])


def predict_from_source(label, plot, ci, cj, backend):
    d = plot._predict_interface()
    src = d.source_msa or {}
    kind = 'merged A3M' if 'a3m' in src else src.get('label', 'none')
    per_chain = src.get('chains') or src.get('names') or {}
    check(f'{label} {backend}: MSA found ({kind})', src and ('a3m' in src or 'merged' in src or
                                                             (ci in per_chain and cj in per_chain)), src)
    check(f'{label} {backend}: reuse offered and chosen', d.reuse.isEnabled() and d.reuse.isChecked(), d.reuse.toolTip())
    d.backend.setCurrentIndex(d.backend.findData(backend))
    d.output.setText(str(OUT / 'jobs'))
    pump(lambda: False, 0.3)
    check(f'{label} {backend}: Predict enabled', d.start_button.isEnabled(), d.msa_hint.text())
    t1 = time.monotonic()
    d.start_button.click()
    pump(lambda: d.job is not None and d.job.state != 'running' and not d.timer.isActive(), 1200)
    job = d.job
    check(f'{label} {backend}: job complete ({time.monotonic() - t1:.0f} s)', job is not None and job.state == 'complete',
          job and job.error)
    if job is None or job.state != 'complete':
        d.close()
        return
    m = job.manifest
    # Paired A3Ms are written when the predicted sequences differ; rows can be paired only between
    # different proteins (and features with UniProt hits)
    parts = ('chain', 'paired') if len({c['sequence'] for c in job.chains}) > 1 else ('chain',)
    sequences = {}
    for r in plot.pae.row_residues_or_atoms():
        sequences.setdefault(r.chain_id, []).append(getattr(r, 'one_letter_code', '?'))
    hetero = len({''.join(sequences[c['source_chain']]) for c in job.chains}) > 1
    if 'a3m' in src:
        check(f'{label} {backend}: source A3M cropped to the box', m['msa_crop'] and m['msa_source'] == 'supplied', m['msa_source'])
    elif 'merged' in src:
        prov = (m.get('features') or [{}])[0]
        check(f'{label} {backend}: merged MSA of features.pkl cropped, {prov.get("rows")} of {prov.get("source_rows")} rows',
              m['msa_source'] == src['origin'] and prov.get('kind') == 'af2_complex' and prov.get('rows', 0) > 1
              and (job.directory / 'features' / 'merged.a3m').is_file(), m.get('features'))
    elif 'names' in src:
        prov = m.get('features') or []
        rows = prov[0].get('paired_rows', 0) if prov else 0
        written = [job.directory / 'features' / f'{part}_{p["chain_id"]}.a3m' for p in prov for part in parts]
        check(f'{label} {backend}: AlphaPulldown features of each chain from {src["where"]}, {rows} rows paired',
              m['msa_source'] == src['origin'] and len(prov) == len(job.chains) and written
              and all(f.is_file() for f in written) and (rows > 0 or not hetero), prov)
    else:
        check(f'{label} {backend}: alignment source recorded', m['msa_source'] == src['origin'], m['msa_source'])
        prov = m.get('features') or []
        check(f'{label} {backend}: provenance of each chain', len(prov) == len(job.chains) and
              all(p['kind'] == 'a3m' for p in prov), prov)
        if backend == 'af2':
            rows = prov[0].get('paired_rows', 0) if prov else 0
            written = [job.directory / 'features' / f'{part}_{p["chain_id"]}.a3m' for p in prov for part in parts]
            check(f'{label} af2: unpaired and paired A3M of each chain, {rows} rows paired by species',
                  written and all(f.is_file() for f in written) and (rows > 0 or not hetero), written)
        else:
            sent = json.loads((job.directory / 'input.json').read_text())['sequences']
            lens = [(len(e['protein']['sequence']), e['protein']['unpairedMsa'].count('>'),
                     e['protein']['pairedMsa'].count('>')) for e in sent]
            ok = all(e['protein']['unpairedMsa'].startswith('>query\n') and e['protein']['pairedMsa'] for e in sent)
            check(f'{label} af3: cropped MSAs in the job (length, unpaired, paired rows): {lens}', ok, lens)
    pump(lambda: d.output_model is not None, 60)
    text = d.status.toPlainText()
    line = next((l for l in text.splitlines() if 'Interface vs source' in l), '')
    check(f'{label} {backend}: result opened, {line.strip() or "no DockQ line"}', d.output_model is not None and line,
          text[-300:])
    d.close()


def main():
    code = 1
    try:
        for path in FOLDS:
            label = f'{path.parent.name}/{path.name}'
            s, seconds = open_prediction(path)
            plot = s._alphalens_plot
            a, b, ci, cj, value = best_box(plot, HALF)
            plot._rectangle_select(a, b)
            pump(lambda: False, 0.5)
            check(f'{label}: opened {s.name} ({len(plot.pae.row_residues_or_atoms())} tokens, {seconds:.0f} s), '
                  f'box rows {ci} x columns {cj} around PAE {value:.1f}', plot._prediction_region is not None)
            for backend in BACKENDS:
                predict_from_source(label, plot, ci, cj, backend)
        for path in NO_MSA:
            label = f'{path.parent.name}/{path.name}'
            s, _ = open_prediction(path)
            plot = s._alphalens_plot
            a, b, *_ = best_box(plot, HALF)
            plot._rectangle_select(a, b)
            d = plot._predict_interface()
            check(f'{label}: no MSA, reuse off with the reason', d.source_msa is None and not d.reuse.isEnabled()
                  and 'None found' in d.reuse.toolTip(), d.reuse.toolTip())
            d.close()
        code = 0 if results and all(r['ok'] for r in results) else 1
    except Exception:
        check('no exception', False, traceback.format_exc())
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    os._exit(code)


main()
