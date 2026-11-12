"""Re-predicted interface against its source, in a real ChimeraX window (GPU, LocalFold).

    ALPHALENS_PILOT=<barnase-barstar LocalFold folder: fold.pdb, fold_confidences.json, fold.a3m>
    ALPHALENS_TEST_OUTPUT=<new folder> ChimeraX --cmd "open tests/chimerax_compare_check.py"

Re-predicts part of the barnase-barstar interface (AF2, then AF3) reusing the source MSA, then
checks: DockQ against the source in comparison.json, the report and the result plot's heading
(also after switching AF3 samples); Compare with source shows only the re-predicted residues of
the source in grey and the result by chain, and restores both exactly; superposing on the first
chain and on the interface moves the result and reports the RMSD.

Environment:
  ALPHALENS_SOURCE            'checkout' (default) tests src/ of this checkout; 'installed' the bundle
  ALPHALENS_LOCALFOLD(_AF3)   engines to use (default: the managed ones)
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

import numpy as np
from chimerax.core.commands import run

SRC = Path(__file__).resolve().parents[1] / 'src'
OUT = Path(os.environ['ALPHALENS_TEST_OUTPUT']).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=False)
PILOT = Path(os.environ['ALPHALENS_PILOT']).expanduser()
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
pc = importlib.import_module('chimerax.alphalens.predict_compare')
from chimerax.atomic import AtomicStructure

results = []


def check(name, ok, detail=''):
    results.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300]))


def pump(seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until is None


def state(m):
    return (m.residues.ribbon_displays.copy(), m.residues.ribbon_colors.copy(), m.atoms.displays.copy(),
            m.atoms.colors.copy(), m.display)


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4])) and a[4] == b[4]


def main():
    code = 1
    try:
        run(session, f'alphalens open "{PILOT}" wait true')
        source = next(m for m in session.models.list(type=AtomicStructure) if getattr(m, '_alphalens_plot', None))
        plot = source._alphalens_plot
        n_a = len(source.chains[0].residues)
        plot._set_prediction_region((25, 90, n_a + 5, n_a + 60))     # Part of barnase against barstar
        for backend in ('af2', 'af3'):
            label = backend
            d = plot._predict_interface()
            d.backend.setCurrentIndex(d.backend.findData(backend))
            d.output.setText(str(OUT / backend))
            check(f'{label}: source MSA reused', d.reuse.isChecked())
            d.start_button.click()
            pump(600, lambda: d.job is not None and d.job.state != 'running' and not d.timer.isActive())
            job = d.job
            check(f'{label}: job complete', job is not None and job.state == 'complete', job and job.error)
            if job is None or job.state != 'complete':
                d.close()
                continue
            comparison = json.loads((job.directory / 'comparison.json').read_text(encoding='utf-8'))
            interface = comparison.get('interface') or {}
            pairs = interface.get('pairs') or []
            check(f'{label}: DockQ against the source', len(pairs) == 1 and 0 <= pairs[0]['dockq'] <= 1
                  and pairs[0]['native_contacts'] > 0, interface)
            report = (job.directory / 'prediction_report.txt').read_text(encoding='utf-8')
            check(f'{label}: DockQ in the report', 'Interface vs source: DockQ' in report
                  and 'source contacts kept' in report, report[-400:])
            result = d.output_model
            rplot = getattr(result, '_alphalens_plot', None)
            heading = rplot._heading.text() if rplot else ''
            check(f'{label}: DockQ in the result plot heading', 'DockQ vs source' in heading
                  and f'{pairs[0]["dockq"]:.2f}' in heading if pairs else False, heading[-200:])
            check(f'{label}: Compare button on', d.compare_button.isEnabled())
            before = {m: state(m) for m in (source, result)}
            d.compare_button.click()
            pump(0.5)
            found = pc.job_files(result)
            matched = pc.matched_residues(source, found[1], result)
            crop = {s for s, _, _, _ in matched if s is not None}
            shown = {r for r, on in zip(source.residues, source.residues.ribbon_displays) if on}
            grey = all(tuple(r.ribbon_color) == pc.SOURCE_RGBA for r in crop)
            check(f'{label}: compare shows only the re-predicted source residues, grey',
                  shown == crop and grey and not source.atoms.displays.any(), f'{len(shown)} shown, {len(crop)} cropped')
            first_chain = [r for _, r, ch, _ in matched if ch is found[1][0] and r is not None]
            check(f'{label}: compare colors the re-prediction by chain',
                  all(tuple(r.ribbon_color) == pc.CHAIN_RGBA[0] for r in first_chain))
            d.compare_button.click()
            pump(0.3)
            check(f'{label}: compare off restores both models', all(same(state(m), before[m]) for m in before)
                  and not pc.comparing(result))
            p0 = result.scene_position.matrix.copy()
            rplot._superpose_on_source('interface')
            p1 = result.scene_position.matrix.copy()
            value = pc.superpose(session, result, 'interface')      # Again: the same place
            check(f'{label}: superposed on the interface', not np.allclose(p0, p1, atol=1e-4)
                  and np.allclose(result.scene_position.matrix, p1, atol=1e-3) and value < 20, f'RMSD {value:.2f}')
            back = pc.superpose(session, result, 'chain')
            check(f'{label}: superposed on the first chain again',
                  np.allclose(result.scene_position.matrix, p0, atol=1e-2), f'RMSD {back:.2f}')
            if backend == 'af3' and rplot is not None:
                models = [m for m in rplot._models if m['path']]
                check(f'{label}: samples in the Model menu', len(models) >= 2, [m['label'] for m in models])
                if len(models) >= 2:
                    rplot._model_chosen(2)
                    pump(20, lambda: rplot.pae.structure is not result)
                    heading2 = rplot._heading.text()
                    check(f'{label}: DockQ follows the shown sample', 'DockQ vs source' in heading2
                          and rplot.pae.structure is not result, heading2[-160:])
            d.close()
            pump(0.3)
        # Later: the source and a saved job folder opened afresh, models shown as ChimeraX opens them
        jobs = sorted((OUT / 'af2').glob('*/job.json'))
        if jobs:
            run(session, 'close')
            run(session, f'alphalens open "{PILOT}" wait true')
            run(session, f'alphalens open "{jobs[0].parent}" wait true')
            opened = session.models.list(type=AtomicStructure)
            source, result = opened[0], opened[-1]
            before = {m: state(m) for m in (source, result)}
            try:
                message = pc.compare_view(session, result, True)
                check('reopened job: compare with source', 'Comparing' in message and pc.comparing(result), message)
            except Exception as e:
                check('reopened job: compare with source', False, repr(e))
            pc.compare_view(session, result, False)
            check('reopened job: restored', all(same(state(m), before[m]) for m in before))
            heading = result._alphalens_plot._heading.text() if getattr(result, '_alphalens_plot', None) else ''
            check('reopened job: DockQ in the heading', 'DockQ vs source' in heading, heading[-160:])
        code = 0 if all(r['ok'] for r in results) else 1
    except Exception:
        results.append(dict(name='exception', ok=False, detail=traceback.format_exc()))
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    for r in results:
        session.logger.info(('PASS ' if r['ok'] else 'FAIL ') + r['name'] + ('' if r['ok'] else f"  {r['detail']}"))
    os._exit(code)


main()
