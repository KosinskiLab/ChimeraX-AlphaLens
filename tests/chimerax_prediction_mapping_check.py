"""Check what a re-prediction reports against known answers, in a real ChimeraX window.

    ALPHALENS_MAPPING_SOURCE=<prediction folder> ALPHALENS_TEST_OUTPUT=<new folder> \\
        ChimeraX --cmd "open tests/chimerax_prediction_mapping_check.py"

No GPU, engine or network is needed. A stand-in engine returns the cropped source structure,
rigidly moved, with the source PAE and contact probabilities of the same residues. Every number
AlphaLens then reports for "before" and "after" must be identical, the prediction must land on
the source, and each CSV row must hold the values of the residues it names. A wrong chain
order, residue offset or matrix index shows up as a difference.

Also checked: the dialog follows a box dragged while it is open, a drag along the pLDDT track
does not recolor the model, a closed result plot can be shown again, a job folder opens with
"alphalens open", sequence input (two chains with a range, one chain), and error styling.

Environment:
  ALPHALENS_MAPPING_SOURCE  prediction folder or model file with contact probabilities and at
                            least two protein chains that touch (it is only read: the test
                            works on links to it in the output folder)
  ALPHALENS_TEST_OUTPUT     new folder for the jobs, results.txt and the alignment cache; the
                            real MSA cache and prediction lock are not used
  ALPHALENS_SOURCE          'checkout' (default) tests src/ of this checkout; 'installed' tests
                            the installed bundle and also saves and restores a session

Results: one PASS/FAIL line per check in results.txt, then a SUMMARY line. Exit status 1 if
any check failed.
"""
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from Qt.QtCore import QTimer
from Qt.QtWidgets import QMenu
from chimerax.core.commands import quote_path_if_necessary as quote, run

SOURCE = Path(os.environ['ALPHALENS_MAPPING_SOURCE']).expanduser().resolve()
OUT = Path(os.environ['ALPHALENS_TEST_OUTPUT']).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=False)
INSTALLED = os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed'
if INSTALLED:
    PACKAGE = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1] / 'src'
    PACKAGE = 'chimerax.alphalens_mapping_check'
    spec = importlib.util.spec_from_file_location(PACKAGE, src / '__init__.py',
                                                  submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    importlib.import_module(PACKAGE + '.cmd').register_commands(session.logger)
predict = importlib.import_module(PACKAGE + '.predict')
ui = importlib.import_module(PACKAGE + '.predict_ui')
distogram_atom = importlib.import_module(PACKAGE + '.contacts').distogram_atom

ENGINE = '''
import json, math, os, pathlib, sys
import numpy as np
args = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in sys.argv[1:])
out = pathlib.Path(args['out'])
job = json.loads(pathlib.Path(args['job']).read_text())
chains = json.loads((out.parent / 'residue_map.json').read_text())
ids = [s['protein']['id'] for s in job['sequences']]
assert [s['protein']['sequence'] for s in job['sequences']] == [c['sequence'] for c in chains]
n = sum(len(c['sequence']) for c in chains)
print(f'AF2 multimer: {n} residues, 512 MSA rows, 1024 extra, 4 passes', flush=True)
for k in range(1, 5):
    print(f'  pass {k}/4: mean pLDDT 80.00', flush=True)
three = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                 'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split()))
lines, serial, token_chain, token_res, plddts = [], 0, [], [], []
source = os.environ.get('ALPHALENS_MAPPING_DATA')
if source:      # The cropped source, rotated and shifted, with its own PAE and contact probabilities
    source = pathlib.Path(source)
    atoms = json.loads((source / 'atoms.json').read_text())
    a, b = math.radians(73), math.radians(-41)
    turn = (np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            @ np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]]))
    rows = [r['source_row'] for c in chains for r in c['residue_map']]
    for cid, c in zip(ids, chains):
        for i, r in enumerate(c['residue_map'], 1):
            token_chain.append(cid)
            token_res.append(i)
            for name, element, xyz, bfactor, resname in atoms[r['source_row']]:
                assert three[r['aa']] == resname, (r, resname)
                serial += 1
                x, y, z = turn @ np.array(xyz) + (55.0, -20.0, 12.5)
                label = f' {name:<3s}' if len(name) < 4 else name
                lines.append(f'ATOM  {serial:5d} {label} {resname} {cid}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}'
                             f'{1.0:6.2f}{bfactor:6.2f}          {element:>2s}')
                plddts.append(bfactor)
        lines.append('TER')
    pae = np.load(source / 'pae.npy')[np.ix_(rows, rows)].astype(float).tolist()
    contact = np.load(source / 'contact.npy')[np.ix_(rows, rows)].astype(float).tolist()
else:           # Sequence input: one helix per chain and random matrices
    rng = np.random.default_rng(int(args.get('seed', 0)))
    for k, (cid, c) in enumerate(zip(ids, chains)):
        for i, aa in enumerate(c['sequence'], 1):
            t = math.radians(100 * i)
            ca = np.array([2.3 * math.cos(t) + 14 * k, 2.3 * math.sin(t), 1.5 * i])
            token_chain.append(cid)
            token_res.append(i)
            side = 1.5 * np.array([math.cos(t), math.sin(t), 0])
            for name, offset in (('N', (0, 0, -0.9)), ('CA', (0, 0, 0)), ('C', (0, 0, 0.9)),
                                 ('O', (0.3, 0.3, 1.6)), ('CB', side)):
                if name == 'CB' and aa == 'G':
                    continue
                serial += 1
                x, y, z = ca + np.array(offset)
                lines.append(f'ATOM  {serial:5d}  {name:<3s} {three[aa]} {cid}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}'
                             f'{1.0:6.2f}{80.0:6.2f}          {name[0]:>2s}')
                plddts.append(80.0)
        lines.append('TER')
    pae = rng.uniform(1, 25, (n, n)).round(2).tolist()
    c = rng.uniform(0, 1, (n, n)) ** 3
    contact = np.maximum(c, c.T).round(3).tolist()
out.write_text('\\n'.join(lines) + '\\nEND\\n')
stem = str(out.with_suffix(''))
pathlib.Path(stem + '_confidences.json').write_text(json.dumps(dict(
    token_chain_ids=token_chain, token_res_ids=token_res, pae=pae, atom_plddts=plddts, contact_probs=contact)))
summary = dict(ptm=0.8, mean_plddt=float(np.mean(plddts)))
if len(chains) > 1:
    summary['iptm'] = 0.7
pathlib.Path(stem + '_summary_confidences.json').write_text(json.dumps(summary))
'''

failed, logged = [], []


def say(text):
    print(text, file=sys.__stdout__, flush=True)
    with (OUT / 'results.txt').open('a', encoding='utf-8') as f:
        f.write(text + '\n')


def check(label, ok, detail=''):
    if not ok:
        failed.append(label)
    say(('PASS  ' if ok else 'FAIL  ') + label + (f'  ::  {detail}' if detail != '' else ''))


def pump(predicate=None, seconds=120.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if predicate is not None and predicate():
            return True
        time.sleep(.01)
    return False


def setup():
    '''The engine, a cache of its own, a record of warnings, and links to the source.'''
    python = Path(sys.executable)
    if not python.name.startswith('python'):
        python = Path(sys.exec_prefix) / 'bin' / f'python{sys.version_info.major}.{sys.version_info.minor}'
    engine = OUT / 'mapping-engine'
    engine.write_text(f'#!{python}\n{ENGINE}')
    engine.chmod(0o755)
    real_job = predict.PredictionJob
    predict.PredictionJob = lambda *a, **k: real_job(*a, **dict(k, cache=OUT / 'cache'))
    for level in ('warning', 'error', 'bug'):
        def record(msg, *a, original=getattr(session.logger, level), level=level, **k):
            logged.append((level, str(msg)))
            return original(msg, *a, **k)
        setattr(session.logger, level, record)
    folder, model = (SOURCE, None) if SOURCE.is_dir() else (SOURCE.parent, SOURCE.name)
    (OUT / 'source').mkdir()
    for entry in folder.iterdir():
        (OUT / 'source' / entry.name).symlink_to(entry)
    return engine, OUT / 'source' / model if model else OUT / 'source'


def box_around_strongest(rra, contact):
    '''Row ranges in two chains around their most probable contact: (first, second).'''
    chains = np.array([r.chain_id for r in rra])
    sym = np.where(chains[:, None] != chains[None, :], np.maximum(contact, contact.T), -1)
    i, j = sorted(np.unravel_index(np.argmax(sym), sym.shape))

    def around(k):
        same = np.flatnonzero(chains == chains[k])
        return max(int(same[0]), k - 25), min(int(same[-1]), k + 25)
    return around(i), around(j)


def predict_region(dialog, engine, tag, rra, contact, region):
    '''Run the dialog's job and compare everything it reports with the known answer.'''
    mark = len(logged)
    dialog.executable.setText(str(engine))
    dialog.output.setText(str(OUT / 'jobs'))
    check(f'{tag}: Predict enabled', dialog.start_button.isEnabled(), dialog.preview.text())
    dialog.start_button.click()
    job = dialog.job
    if job is None or job.state != 'running':
        check(f'{tag}: job started', False, dialog.status.toPlainText()[-300:])
        return None
    pump(lambda: not dialog.timer.isActive())
    check(f'{tag}: job complete', job.state == 'complete', dialog.status.toPlainText()[-300:])
    if job.state != 'complete':
        return None
    comparison, model, report = dialog.comparison, dialog.output_model, job.contact_report
    check(f'{tag}: chains in the order rows, columns',
          [c['source_chain'] for c in job.chains] == [rra[region[0]].chain_id, rra[region[2]].chain_id],
          [c['source_chain'] for c in job.chains])
    check(f'{tag}: mean PAE of the same pairs unchanged',
          abs(comparison['predicted_core_mean_pae'] - job.source['core_mean_pae']) < 1e-4,
          (job.source['core_mean_pae'], comparison['predicted_core_mean_pae']))
    check(f'{tag}: mean pLDDT of the same residues unchanged',
          abs(comparison['predicted_core_mean_plddt'] - job.source['core_mean_plddt']) < 0.02,
          (job.source['core_mean_plddt'], comparison['predicted_core_mean_plddt']))
    check(f'{tag}: contact report the same before and after',
          all(report['source'][k] == report['prediction'][k]
              or abs(report['source'][k] - report['prediction'][k]) < 1e-5
              for k in ('count', 'top10_mean', 'structural_count', 'cutoff_A')),
          (report['source'], report['prediction']))
    check(f'{tag}: superposed on the first chain', comparison['aligned_chain_ca_rmsd'] < 0.01,
          comparison['aligned_chain_ca_rmsd'])
    rows = [r['source_row'] for c in job.chains for r in c['residue_map']]
    deviation = max(float(np.linalg.norm(new.find_atom('CA').scene_coord - rra[i].find_atom('CA').scene_coord))
                    for new, i in zip(model.alphafold_pae.row_residues_or_atoms(), rows))
    check(f'{tag}: whole prediction lies on the source', deviation < 0.02, f'largest CA deviation {deviation:.3f} Å')
    index = {(r.chain_id, str(r.number) + r.insertion_code): i for i, r in enumerate(rra)}
    sym = np.maximum(contact, contact.T)
    y0, y1, x0, x1 = region
    wrong, n = [], 0
    with open(report['csv']) as f:
        for row in csv.DictReader(f):
            n += 1
            i, j = index[(row['chain_1'], row['residue_1'])], index[(row['chain_2'], row['residue_2'])]
            inside = (y0 <= i <= y1 and x0 <= j <= x1) or (y0 <= j <= y1 and x0 <= i <= x1)
            d = float(np.linalg.norm(distogram_atom(rra[i]).coord - distogram_atom(rra[j]).coord))
            if (not inside or abs(float(row['source_probability']) - sym[i, j]) > 1e-5
                    or abs(float(row['prediction_probability']) - sym[i, j]) > 1e-5
                    or abs(float(row['source_distance_A']) - d) > 1e-3
                    or abs(float(row['prediction_distance_A']) - d) > 5e-3):
                wrong.append(row)
    check(f'{tag}: the {n} CSV rows hold the values of the residues they name', n > 0 and not wrong, wrong[:2])
    check(f'{tag}: report saved', (job.directory / 'prediction_report.txt').is_file())
    check(f'{tag}: nothing logged as a warning or error', len(logged) == mark, logged[mark:mark + 3])
    return job


def main():
    engine, source_path = setup()
    pae = run(session, f'alphalens open {quote(str(source_path))} wait true')
    src = pae.structure
    plot = src._alphalens_plot
    rra = pae.row_residues_or_atoms()
    n = len(rra)
    check('source has contact probabilities and two chains',
          pae.has_contact and len({r.chain_id for r in rra}) > 1)
    run(session, f'turn y 40 models #{src.id_string}')         # The scene is not the file's frame
    run(session, f'move x 30 models #{src.id_string}')
    contact = np.clip(np.maximum(pae.metrics['contact'], pae.metrics['contact'].T), 0, 1)
    (OUT / 'data').mkdir()
    (OUT / 'data' / 'atoms.json').write_text(json.dumps(
        [[(a.name, a.element.name, [float(v) for v in a.coord], float(a.bfactor), r.name)
          for a in r.atoms] for r in rra]))
    np.save(OUT / 'data' / 'pae.npy', pae.matrix('pae'))
    np.save(OUT / 'data' / 'contact.npy', contact)
    os.environ['ALPHALENS_MAPPING_DATA'] = str(OUT / 'data')
    (first0, first1), (second0, second1) = box_around_strongest(rra, contact)
    split = next(i for i, r in enumerate(rra) if r.chain_id == rra[second0].chain_id)
    query = ''.join(r.one_letter_code for r in rra)
    stem = Path(src.filename).with_suffix('')
    Path(f'{stem}.a3m').write_text(                              # In the output folder, by the links
        f'>101\t102\n{query}\n>paired\n{query[:first0]}aa{query[first0:]}\n'
        f'>first\n{query[:split]}{"-" * (n - split)}\n>second\n{"-" * split}{query[split:]}\n')

    # A drag along the pLDDT track selects; it must not recolor the model
    colors = src.atoms.colors.copy()
    plot._rectangle_select((first0, n + 5), (first1, n + 8))
    check('track: drag leaves the model colors alone', np.array_equal(src.atoms.colors, colors))
    check('track: drag selects its residues', int(src.residues.selected.sum()) == first1 - first0 + 1,
          int(src.residues.selected.sum()))
    check('track: drag is not a box to re-predict', plot._prediction_region is None)

    # Rows in the first chain, columns in the second, source alignment cropped
    region = (first0, first1, second0, second1)
    plot._rectangle_select((second0, first0), (second1, first1))
    check('box: recorded for re-prediction', plot._prediction_region == region, plot._prediction_region)
    dialog = plot._predict_interface()
    pump(seconds=.3)
    check('box: source alignment found and selected', dialog.reuse.isChecked(), dialog.reuse.toolTip())
    job = predict_region(dialog, engine, 'rows first chain', rra, contact, region)
    if job is not None:
        a3m = (job.directory / 'input.a3m').read_text().splitlines()
        check('cropped alignment starts with the job sequences',
              a3m[1] == ''.join(c['sequence'] for c in job.chains))
        menu = QMenu()
        dialog.output_model._alphalens_plot._fill_context_menu(menu, 0, 0)
        check('result plot offers its report', 'Prediction report…' in [a.text() for a in menu.actions()])

    # The box is dragged the other way round while the dialog is open
    before = dialog.preview.text()
    plot._rectangle_select((first0, second0), (first1, second1))
    pump(seconds=.3)
    region = (second0, second1, first0, first1)
    check('open dialog shows the newly dragged box', dialog.preview.text() != before
          and dialog.preview.text().startswith(rra[second0].chain_id + ':'), dialog.preview.text())
    job = predict_region(dialog, engine, 'rows second chain', rra, contact, region)

    # A local prediction is found again from its own files
    if job is not None:
        model = dialog.output_model
        model._alphalens_plot.delete()
        pump(seconds=.3)
        for label, command, target in (('re-shown after its plot was closed', f'alphalens #{model.id_string}', None),
                                       ('opened from its job folder', 'alphalens open', job.directory),
                                       ('opened from fold.pdb', 'alphalens open', job.directory / 'fold.pdb')):
            try:
                if target is not None:
                    command += ' ' + quote(str(target))
                shown = run(session, command)
                ok, detail = shown.has_contact and not shown.structure._alphalens_plot.closed(), ''
                if target is not None:
                    run(session, f'close #{shown.structure.id_string}')
            except Exception as e:
                ok, detail = False, f'{type(e).__name__}: {e}'
            check(f'local prediction {label}', ok, detail)

    # The source is closed under the open dialog
    mark = len(logged)
    run(session, f'close #{src.id_string}')
    pump(seconds=.5)
    check('source closed: Predict off, with the reason',
          not dialog.start_button.isEnabled() and 'closed' in dialog.preview.text(), dialog.preview.text())
    dialog.close()
    pump(seconds=.3)
    check('source closed: nothing logged as a warning or error', len(logged) == mark, logged[mark:mark + 3])

    # Sequence input
    os.environ.pop('ALPHALENS_MAPPING_DATA')
    for case, text, ranges in (('two chains', '>first\nMKVLAAGIVGLAAHEDCQ\n>second\nGSSPEELLKKALELAKKG\n', ['3-14', '']),
                               ('one chain', 'MKVLAAGIVGLAAHEDCQGSSPEELLKKALELAKKG', [''])):
        mark = len(logged)
        dialog = ui.show_prediction_dialog(session=session)
        dialog.input_text.setPlainText(text)
        dialog.load_button.click()
        pump(lambda: not dialog.lookup_timer.isActive(), 20)
        for field, value in zip(dialog.ranges, ranges):
            field.setText(value)
        chains, _ = dialog.prediction_input()
        msa = OUT / f'{case.replace(" ", "-")}.a3m'
        msa.write_text('>query\n' + ''.join(c['sequence'] for c in chains) + '\n')
        dialog.msa.setText(str(msa))
        dialog.executable.setText(str(engine))
        dialog.output.setText(str(OUT / 'jobs'))
        dialog.start_button.click()
        job = dialog.job
        done = job is not None and pump(lambda: not dialog.timer.isActive()) and job.state == 'complete'
        check(f'sequence, {case}: job complete', done, dialog.status.toPlainText()[-300:])
        if done and case == 'two chains':
            with open(job.contact_report['csv']) as f:
                numbers = {int(r['residue_1']) for r in csv.DictReader(f) if r['chain_1'] == 'A'}
            check('sequence, two chains: CSV numbers residues as in the input',
                  numbers and min(numbers) >= 3 and max(numbers) <= 14, sorted(numbers))
            check('sequence, two chains: no source comparison', dialog.comparison is None)
        elif done:
            check('sequence, one chain: monomer model, no contact report',
                  job.model == predict.MONOMER_MODEL and job.contact_report is None, job.model)
        check(f'sequence, {case}: nothing logged as a warning or error', len(logged) == mark, logged[mark:mark + 3])
        dialog.close()
        pump(seconds=.3)

    # An input error is highlighted; the next lookup is not
    dialog = ui.show_prediction_dialog(session=session)
    for text, highlighted in (('MKV1LAA 22', True), ('MKVLAAGIVGLAAHEDCQ', False)):
        dialog.input_text.setPlainText(text)
        dialog.load_button.click()
        pump(lambda: not dialog.lookup_timer.isActive(), 20)
        check(f'lookup of {text!r}: {"" if highlighted else "not "}shown as an error',
              ('bold' in dialog.status.styleSheet()) == highlighted, dialog.status.styleSheet())
    dialog.close()
    pump(seconds=.3)

    if INSTALLED:   # Sessions name the class by bundle: only the installed one can be saved
        mark = len(logged)
        count = sum(1 for m in session.models.list()
                    if getattr(m, '_alphalens_plot', None) is not None and not m._alphalens_plot.closed())
        run(session, f'save {quote(str(OUT / "mapping.cxs"))}')
        run(session, 'close session')
        run(session, f'open {quote(str(OUT / "mapping.cxs"))}')
        pump(seconds=2)
        restored = sum(1 for m in session.models.list()
                       if getattr(m, '_alphalens_plot', None) is not None and not m._alphalens_plot.closed())
        check('session: plots of local predictions are restored', restored == count > 0, f'{restored} of {count}')
        check('session: nothing logged as a warning or error', len(logged) == mark, logged[mark:mark + 3])


def start():
    try:
        main()
    except Exception:
        check('the check ran to its end', False, traceback.format_exc())
    say(f'SUMMARY: {len(failed)} failed' + (': ' + '; '.join(failed) if failed else ''))
    os._exit(1 if failed else 0)


QTimer.singleShot(1500, start)
