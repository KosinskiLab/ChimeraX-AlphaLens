"""Check that choosing another model in the Model menu updates everything that belongs to a model.

    ALPHALENS_SWITCH_DIRS="/path/to/af2_prediction;/path/to/af3_prediction" \\
        ChimeraX --exit --cmd "open tests/chimerax_model_switch_check.py"

Each folder needs at least four models: AF2 ranked models whose result pickles kept the distogram,
or AF3 samples. The check opens the top model and sets the window up as a user would: the Model
contacts view, a contact threshold, Show Contacts, a clicked contact, a box dragged across the
interface (coloring its residues) and the pLDDT slider. Then it chooses other models in the Model
menu: one, back to the first with the contact lines hidden, three in quick succession after Color
pLDDT (while the first two are being read: only the last is opened), the fourth, and, with two
seeds or models, one in the agreement view. The samples of an AF3 seed must share its distogram. After each switch every value that belongs to a model is compared with
one computed here from the files and coordinates of the model now shown: PAE, pLDDT track,
contact probabilities, spread, the Model contacts counts and line, the dots, the contact lines,
the clicked contact, the hidden residues, the colors (the box's, or the model's own pLDDT), the
CSV export, the heading and what a saved session keeps. Show Contacts and the clicked contact
stay as the user left them, and the models left behind keep nothing of the window's: no lines,
clicked contact or hidden residues.

Environment:
  ALPHALENS_SWITCH_DIRS       prediction folders separated by ';' (required)
  ALPHALENS_SOURCE            'checkout' (default): src/ of this checkout; 'installed': the bundle
  ALPHALENS_SWITCH_AGREEMENT  '0' skips the agreement step (it reads every seed or model)
  ALPHALENS_SWITCH_OUT        folder for results.txt and CSV files (default: a new temporary folder)

Results: one PASS/FAIL line per check in results.txt and the log, then a SUMMARY line. ChimeraX
exits with status 1 if any check failed, else 0.
"""

import csv
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
from chimerax.core.commands import quote_path_if_necessary as quote, run

INSTALLED = os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed'
THRESHOLD = 0.3
PLDDT_HIDE = 70


def _package(session):
    if INSTALLED:
        return 'chimerax.alphalens'
    src = Path(__file__).resolve().parents[1] / 'src'
    name = 'chimerax.alphalens_switch'
    spec = importlib.util.spec_from_file_location(name, src / '__init__.py',
                                                  submodule_search_locations=[str(src)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    importlib.import_module(name + '.cmd').register_commands(session.logger)
    return name


class Checks:
    def __init__(self, session, out):
        self.session, self.out = session, out
        self.results = out / 'results.txt'
        self.results.write_text('')
        self.failed = 0

    def log(self, line):
        with self.results.open('a') as f:
            f.write(line + '\n')
        self.session.logger.info(line)

    def check(self, name, ok, detail=''):
        ok = bool(ok)
        self.failed += not ok
        self.log(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')
        return ok

    def pump(self, seconds=0.3):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.session.ui.processEvents()
            time.sleep(0.02)

    def wait_for(self, cond, what, timeout=600):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout:
                self.check(f'{what} within {timeout} s', False, 'timed out')
                return False
            self.session.ui.processEvents()
            time.sleep(0.05)
        return True


class Expected:
    '''What the shown model's files and coordinates say, read here and not through the plot.'''

    def __init__(self, data):
        self.data = data
        self._files = {}

    def files(self, s):
        key = os.path.realpath(s.filename)
        if key not in self._files:
            data = self.data
            files = data.find_prediction_files(s.filename)
            pae, cp = data.read_pae(files.pae_path)
            metrics = data.load_distogram_metrics(files.distogram_path, 8.0) if files.distogram_path else None
            contact = metrics['contact'] if metrics is not None else cp
            self._files[key] = dict(files=files, pae=pae, contact=contact, metrics=metrics)
        return self._files[key]


def residues_of(plot):
    from chimerax.atomic import Residue
    rra = plot.pae.row_residues_or_atoms()
    assert all(isinstance(r, Residue) for r in rra), 'protein-only predictions expected'
    return rra


def plddt_of(rra):
    p = np.array([r.find_atom('CA').bfactor for r in rra], np.float32)
    return p * 100 if p.max() <= 1 else p


def distogram_atoms(rra):
    return [r.find_atom('CA' if r.name == 'GLY' else 'CB') or r.find_atom('CA') for r in rra]


def pair_set(pairs):
    return {(int(i), int(j)) for i, j in pairs}


def line_pairs(s, group, rows):
    '''Row pairs (i < j) joined by the pseudobonds of group on s.'''
    g = s.pseudobond_group(group, create_type=None)
    if g is None or g.num_pseudobonds == 0:
        return set()
    a1, a2 = g.pseudobonds.atoms
    pairs = set()
    for x, y in zip(a1, a2):
        i, j = rows[x.residue], rows[y.residue]
        pairs.add((min(i, j), max(i, j)))
    return pairs


def in_region(i, j, region):
    if region is None:
        return np.ones(len(i), bool)
    r1, r2, c1, c2 = region
    in_r = lambda k: (k >= r1) & (k <= r2)
    in_c = lambda k: (k >= c1) & (k <= c2)
    return (in_r(i) & in_c(j)) | (in_c(i) & in_r(j))


def verify(c, ctx, plot, s, what, lines, picked, left=()):
    '''
    Every model-specific value of the window against the files and coordinates of s. lines: Show
    Contacts is on (as the user left it); picked: the clicked contact (i, j) or None; left: the
    models shown before, which must keep nothing of the window's.
    '''
    contacts, tool, exp_all = ctx['contacts'], ctx['tool'], ctx['expected']
    pae = plot.pae
    c.check(f'{what}: the window shows {s}', pae.structure is s and getattr(s, 'alphafold_pae', None) is pae
            and getattr(s, '_alphalens_plot', None) is plot and s.display, str(pae.structure))
    rra = residues_of(plot)
    c.check(f'{what}: the plot rows are residues of {s}', all(r.structure is s for r in rra))
    exp = exp_all.files(s)
    c.check(f'{what}: PAE of {s}', pae.pae_matrix.shape == exp['pae'].shape
            and np.allclose(pae.pae_matrix, exp['pae']), os.path.basename(exp['files'].pae_path))
    plddt = plddt_of(rra)
    c.check(f'{what}: pLDDT track of {s}', np.allclose(plot._plddt, plddt))
    P = exp['contact']
    c.check(f'{what}: contact probabilities of {s}', pae.has_contact and np.allclose(pae.metrics['contact'], P, atol=1e-5),
            os.path.basename(exp['files'].distogram_path or exp['files'].pae_path))
    if exp['metrics'] is not None:
        c.check(f'{what}: spread of {s}', pae.has_distogram and np.allclose(pae.matrix('std'), exp['metrics']['std'], atol=1e-4))

    n = len(rra)
    rows = {r: k for k, r in enumerate(rra)}
    chains = np.array([r.chain_id for r in rra])
    upper = np.triu(chains[:, None] != chains[None, :], 1)
    xyz = np.array([a.coord for a in distogram_atoms(rra)], np.float64)
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    t = plot.threshold
    predicted = np.maximum(P, P.T) >= t
    close = d < 8.0
    formed = int((close & predicted & upper).sum())
    missed = int((~close & predicted & upper).sum())
    unpredicted = int((close & ~predicted & upper).sum())
    if plot._view_mode == 'model':
        counts = plot._model_counts or {}
        c.check(f'{what}: Model contacts counts from the coordinates of {s}',
                (counts.get('formed'), counts.get('missed'), counts.get('unpredicted')) == (formed, missed, unpredicted),
                f'plot {counts.get("formed")}/{counts.get("missed")}/{counts.get("unpredicted")}, '
                f'expected {formed}/{missed}/{unpredicted}')
        plot._set_info('')                  # Its idle text: a real pointer over the window would hover
        text = plot._info.text()
        c.check(f'{what}: the line under the plot gives those counts',
                f'{formed} of {formed + missed} predicted' in text and f'{unpredicted} inter-chain contact' in text, text[:120])
        want = pair_set(np.argwhere((close | predicted) & upper))
        got = pair_set(plot._view.dot_pairs) if plot._view.dot_pairs is not None else set()
        c.check(f'{what}: Model contacts dots of {s}', got == want, f'{len(got)} dots, expected {len(want)}')
    else:
        want = pair_set(np.argwhere(predicted & upper))
        got = pair_set(plot._view.dot_pairs) if plot._view.dot_pairs is not None else set()
        c.check(f'{what}: contact dots of {s}', got == want, f'{len(got)} dots, expected {len(want)}')

    # Lines on the structure: the predicted contacts at the threshold, inside the dragged box
    ij = np.argwhere(predicted & upper)
    ij = ij[in_region(ij[:, 0], ij[:, 1], plot._struct_region)]
    got = line_pairs(s, contacts.GROUP_NAME, rows)
    button = plot._contacts_button
    if lines:
        c.check(f'{what}: Show Contacts still on', plot._struct_contacts and button.isChecked()
                and button.text() == 'Hide Contacts', button.text())
        c.check(f'{what}: contact lines on {s}', got == pair_set(ij), f'{len(got)} lines, expected {len(ij)}')
    else:
        c.check(f'{what}: no contact lines on {s} (Show Contacts off)', not got and not button.isChecked(),
                f'{len(got)} lines')
    shown = line_pairs(s, contacts.PICK_GROUP_NAME, rows)
    if picked is not None:
        c.check(f'{what}: clicked contact on {s}', plot._picked == picked and shown == {tuple(sorted(picked))},
                f'{plot._picked} {shown}')
    else:
        c.check(f'{what}: no clicked contact on {s}', plot._picked is None and not shown, str(shown))

    # Hidden residues: pLDDT below the slider, from this model's pLDDT
    hide = plddt < plot._plddt_slider.value()
    ribbons = np.array([r.ribbon_display for r in rra])
    c.check(f'{what}: residues of {s} with pLDDT < {plot._plddt_slider.value()} hidden, the others shown',
            np.array_equal(ribbons, ~hide), f'{int(hide.sum())} to hide, {int((~ribbons).sum())} hidden')
    heading = plot._heading.text()
    c.check(f'{what}: heading names {s}', str(s) in heading.replace('&lt;', '<').replace('&gt;', '>'))

    # CSV export: P and model distances of this model
    path = c.out / f'{what.replace(" ", "_").replace(":", "")}.csv'
    plot.save_contacts_csv(str(path))
    by_id = {(r.chain_id, str(r.number)): k for k, r in enumerate(rra)}
    bad, total = [], 0
    for row in csv.DictReader(path.open()):
        total += 1
        i, j = by_id[(row['chain_1'], row['residue_1'])], by_id[(row['chain_2'], row['residue_2'])]
        p = max(P[i, j], P[j, i])
        ca1, ca2 = rra[i].find_atom('CA'), rra[j].find_atom('CA')
        dist = float(np.linalg.norm(np.asarray(ca1.coord, np.float64) - np.asarray(ca2.coord, np.float64)))
        if abs(float(row['p_contact']) - p) > 6e-4 or abs(float(row['model_distance']) - dist) > 6e-3 \
                or abs(float(row['pae_1_2']) - exp['pae'][i, j]) > 6e-3:
            bad.append(f'{row["chain_1"]}{row["residue_1"]}-{row["chain_2"]}{row["residue_2"]}')
    agr = pae.agreement
    if agr is None:
        expected_rows = len(ij)
    else:                                   # Pairs at the threshold in the shown model or any seed or model
        kk = np.argwhere((predicted | (agr.counts(t) > 0)) & upper)
        expected_rows = int(in_region(kk[:, 0], kk[:, 1], plot._struct_region).sum())
    c.check(f'{what}: CSV export has P, PAE and model distances of {s}', not bad and total == expected_rows,
            f'{total} rows, expected {expected_rows}; wrong: {bad[:3]}')

    snap = plot.take_snapshot(c.session, 0)
    c.check(f'{what}: a saved session keeps {s} and its files', snap['structure'] is s
            and os.path.realpath(snap['model_file']) == os.path.realpath(s.filename)
            and os.path.realpath(snap['pae_file']) == os.path.realpath(exp['files'].pae_path)
            and (snap['distogram_file'] or None) == (exp['files'].distogram_path or None))
    c.check(f'{what}: no progress bar left', plot._progress_row.isHidden() and plot._load_progress is None)

    for o in left:
        rows_o = {r: k for k, r in enumerate(o.residues)}
        lines = line_pairs(o, contacts.GROUP_NAME, rows_o)
        picks = line_pairs(o, contacts.PICK_GROUP_NAME, rows_o)
        hidden = int((~o.residues.ribbon_displays).sum())
        c.check(f'{what}: {o}, left behind, is hidden and keeps nothing of the window', not o.display
                and getattr(o, '_alphalens_plot', None) is None and not lines and not picks and not hidden
                and not getattr(o, '_alphalens_picked', None),
                f'display {o.display}, {len(lines)} lines, {len(picks)} clicked, {hidden} residues hidden')


def box_colors_ok(c, s, rra, coloring, what):
    from chimerax.core.colors import BuiltinColors
    want = np.zeros((len(rra), 4), np.uint8)
    for row1, row2, color, other in coloring:
        if other:
            want[:] = BuiltinColors[other].uint8x4()
        want[row1:row2 + 1] = BuiltinColors[color].uint8x4()
    got = np.array([r.ribbon_color for r in rra])
    return c.check(f'{what}: the dragged box colors the same residues of {s}', np.array_equal(got, want),
                   f'{int((got != want).any(1).sum())} residues differ')


def plddt_colors_ok(c, s, rra, what):
    bands = np.array([(255, 125, 69), (255, 219, 19), (101, 203, 243), (0, 83, 214)])
    p = plddt_of(rra)
    band = np.where(p <= 50, 0, np.where(p <= 70, 1, np.where(p <= 90, 2, 3)))
    got = np.array([r.find_atom('CA').color[:3] for r in rra], int)
    near = np.abs(got[:, None, :] - bands[None]).max(-1) <= 1
    return c.check(f'{what}: {s} colored by its own pLDDT', bool(near[np.arange(len(band)), band].all()),
                   f'{int((~near[np.arange(len(band)), band]).sum())} residues in another band')


def model_index(plot, path):
    real = os.path.realpath(path)
    for k, e in enumerate(plot._menu_entries):
        if e is not None and e[0] == 'model' and plot._models[e[1]]['path'] \
                and os.path.realpath(plot._models[e[1]]['path']) == real:
            return k
    return None


def opened(c, path):
    '''The open structure of a model file, or None.'''
    from chimerax.atomic import AtomicStructure
    real = os.path.realpath(path)
    return next((m for m in c.session.models.list(type=AtomicStructure)
                 if getattr(m, 'filename', None) and os.path.realpath(m.filename) == real), None)


def choose(c, plot, path, wait=True):
    '''Choose the model in the Model menu, as a user does; with wait, until it is shown and read.'''
    k = model_index(plot, path)
    menu = plot._model_combo
    menu.setCurrentIndex(k)
    menu.activated.emit(k)
    if not wait:
        return None
    c.wait_for(lambda: not plot.switch_pending, f'{os.path.basename(path)}: files read')
    s = opened(c, path)
    loaded(c, plot, f'{s} read')
    return s


def loaded(c, plot, what):
    def done():
        pae = plot.pae
        load = getattr(pae, '_distogram_load', None)
        return pae.has_contact and plot._load_progress is None and (load is None or not load.pending)
    ok = c.wait_for(done, what)
    c.pump(0.5)
    return ok


def drag(c, plot, xy1, xy2):
    '''Drag a box on the plot with the mouse, between scene points (x = column, y = row).'''
    from Qt.QtCore import QPointF, Qt
    try:
        from Qt.QtTest import QTest
    except ImportError:                     # ChimeraX's Qt shim leaves QtTest out
        from PyQt6.QtTest import QTest
    v = plot._view
    p1 = v.mapFromScene(QPointF(*xy1))
    p2 = v.mapFromScene(QPointF(*xy2))
    vp = v.viewport()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    for f in (0.3, 0.7, 1.0):
        QTest.mouseMove(vp, p1 + (p2 - p1) * f)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2)
    from Qt.QtCore import QEvent
    v.leaveEvent(QEvent(QEvent.Type.Leave))     # The pointer goes on to the Model menu
    c.pump(0.3)


def check_prediction(c, ctx, label, path):
    session = c.session
    cmd, contacts = ctx['cmd'], ctx['contacts']
    run(session, 'close')
    pae = cmd.alphalens_open(session, path, wait=True)
    plot = pae.structure._alphalens_plot
    loaded(c, plot, f'{label} top model read')
    models = [m['path'] for m in plot._models if m['path']]
    if not c.check(f'{label}: at least four models in the Model menu', len(models) >= 4, str(len(models))):
        return
    if os.path.basename(models[0]).endswith('_model.cif'):
        models = models[:1] + models[2:]     # AF3: the top model is sample #1 again
    a_path, b_path, c_path, d_path = models[:4]
    s_a = choose(c, plot, a_path) if os.path.realpath(pae.structure.filename) != os.path.realpath(a_path) \
        else pae.structure

    # The window as a user sets it up
    plot.set_mode('model')
    plot.set_threshold(THRESHOLD)
    plot._contacts_button.click()
    c.pump(0.3)
    strongest, probs = contacts.contact_pairs(plot.pae, 0.05)       # Strongest first
    if not c.check(f'{label}: an inter-chain pair to click', len(strongest) > 0):
        return
    picked = (int(strongest[0][0]), int(strongest[0][1]))
    c.log(f'INFO  {label}: clicking the strongest inter-chain pair, P {probs[0]:.2f}')
    plot._pick_contact(*picked)
    plot._plddt_slider.setValue(PLDDT_HIDE)
    plot._apply_plddt_threshold()
    chains = plot._view.chains
    (_, a0, a1), (_, b0, b1) = chains[0], chains[1]
    drag(c, plot, (b0 + 0.5 + (b1 - b0) // 4, a0 + 0.5 + (a1 - a0) // 4),
         (b1 - 0.5 - (b1 - b0) // 4, a1 - 0.5 - (a1 - a0) // 4))
    region = plot._struct_region
    c.check(f'{label}: a box dragged across the interface', region is not None
            and plot._prediction_region == region, str(region))
    coloring = None
    if region is not None:
        r3, r4, r1, r2 = region
        coloring = [(r1, r2, 'lime', 'gray'), (r3, r4, 'magenta', None)]
        box_colors_ok(c, s_a, residues_of(plot), coloring, f'{label} first model')
    verify(c, ctx, plot, s_a, f'{label} first model', lines=True, picked=picked)

    # Another model
    s_b = choose(c, plot, b_path)
    what = f'{label} second model'
    verify(c, ctx, plot, s_b, what, lines=True, picked=picked, left=[s_a])
    dist_a = ctx['expected'].files(s_a)['files'].distogram_path
    same_file = dist_a is not None and dist_a == ctx['expected'].files(s_b)['files'].distogram_path
    shared = np.shares_memory(s_a.alphafold_pae.metrics['contact'], s_b.alphafold_pae.metrics['contact'])
    c.check(f'{what}: ' + ('the distogram of its seed, as read for the first sample, not read again' if same_file
                           else 'its own distogram or contact probabilities'), shared == same_file
            and (not same_file or getattr(s_b.alphafold_pae, '_distogram_load', None) is None))
    c.check(f'{what}: the box is kept for Re-predict selected interface', plot._struct_region == region
            and plot._prediction_region == region and plot._view._drag_box is not None,
            f'{plot._struct_region} {plot._prediction_region}')
    if coloring:
        box_colors_ok(c, s_b, residues_of(plot), coloring, what)

    # Lines hidden here, then back to the first model: it must not show its old lines
    plot._contacts_button.click()
    c.pump(0.3)
    choose(c, plot, a_path)
    verify(c, ctx, plot, s_a, f'{label} back to the first model, lines hidden', lines=False, picked=picked,
           left=[s_b])
    plot._contacts_button.click()           # Show Contacts again: all of them, not only the box's
    c.pump(0.3)

    # Three choices in quick succession, back to a model whose distogram is still being read:
    # a late read must not change the window, and its progress shows again
    plot._color_plddt()
    c.pump(0.3)
    plddt_colors_ok(c, s_a, residues_of(plot), f'{label} first model')
    t0 = time.perf_counter()
    choose(c, plot, c_path, wait=False)
    choose(c, plot, d_path, wait=False)
    choose(c, plot, c_path, wait=False)
    blocked = time.perf_counter() - t0
    c.check(f'{label} three quick choices: the window says it is reading, the menu shows the last choice',
            plot.switch_pending and not plot._progress_row.isHidden() and 'Reading' in plot._progress_label.text()
            and plot._menu_entries[plot._model_combo.currentIndex()] == ('model', next(
                i for i, m in enumerate(plot._models) if m['path'] and os.path.realpath(m['path']) == os.path.realpath(c_path))),
            f'{plot._progress_label.text()!r}, the three choices took {blocked:.2f} s')
    c.wait_for(lambda: not plot.switch_pending, f'{label} third model\'s files read')
    s_c = opened(c, c_path)
    loaded(c, plot, f'{label} third model read')
    c.pump(0.5)
    what = f'{label} three choices in quick succession'
    c.check(f'{what}: the model chosen in between was never opened', opened(c, d_path) is None)
    verify(c, ctx, plot, s_c, what, lines=True, picked=picked, left=[s_a, s_b])
    plddt_colors_ok(c, s_c, residues_of(plot), what)

    # A model to read, then at once one shown before: the later choice wins, the read is dropped
    choose(c, plot, d_path, wait=False)
    choose(c, plot, a_path, wait=False)
    c.pump(2.0)                             # Long enough for the dropped read to come back
    c.check(f'{label} a model shown at once supersedes one still being read', plot.pae.structure is s_a
            and not plot.switch_pending and opened(c, d_path) is None, str(plot.pae.structure))

    # The fourth, read now
    s_d = choose(c, plot, d_path)
    what = f'{label} fourth model'
    verify(c, ctx, plot, s_d, what, lines=True, picked=picked, left=[s_a, s_b, s_c])
    plddt_colors_ok(c, s_d, residues_of(plot), what)

    if os.environ.get('ALPHALENS_SWITCH_AGREEMENT', '1') != '0' and len(plot._agreement_runs) >= 2:
        plot.set_mode('agreement')
        c.wait_for(lambda: plot.pae.agreement is not None or plot._agreement_error is not None,
                   f'{label} agreement read')
        c.pump(0.5)
        agr = plot.pae.agreement
        s_b = choose(c, plot, b_path)
        what = f'{label} second model with the agreement view'
        c.check(f'{what}: the agreement is kept', agr is not None and plot.pae.agreement is agr
                and plot._view_mode == 'agreement', str(plot._agreement_error))
        verify(c, ctx, plot, s_b, what, lines=True, picked=picked, left=[s_a, s_c, s_d])
    run(session, 'close')
    c.pump(0.5)


def main(session):
    dirs = [d.strip() for d in os.environ.get('ALPHALENS_SWITCH_DIRS', '').split(';') if d.strip()]
    out = Path(os.environ.get('ALPHALENS_SWITCH_OUT') or tempfile.mkdtemp(prefix='alphalens-switch-'))
    out.mkdir(parents=True, exist_ok=True)
    c = Checks(session, out)
    if not session.ui.is_gui:
        c.check('window available (run without --nogui)', False)
    elif not dirs:
        c.check('ALPHALENS_SWITCH_DIRS lists prediction folders', False)
    else:
        pkg = _package(session)
        c.log(f'INFO  testing {pkg}; output in {out}')
        ctx = {name: importlib.import_module(f'{pkg}.{name}') for name in ('cmd', 'contacts', 'data', 'tool')}
        ctx['expected'] = Expected(ctx['data'])
        for k, path in enumerate(dirs):
            label = f'{k + 1}_{Path(path.rstrip("/")).parent.name}'
            try:
                check_prediction(c, ctx, label, path)
            except Exception:
                c.check(f'{label} completed without an exception', False, traceback.format_exc()[-900:])
                try:
                    run(session, 'close')
                except Exception:
                    pass
    total = sum(1 for line in c.results.read_text().splitlines() if line.startswith(('PASS', 'FAIL')))
    c.log(f'SUMMARY {total - c.failed} passed, {c.failed} failed; results in {c.results}')
    os._exit(1 if c.failed else 0)


main(session)
