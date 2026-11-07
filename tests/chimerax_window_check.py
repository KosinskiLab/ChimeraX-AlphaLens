"""Check the AlphaLens plot in a real ChimeraX window on real predictions.

    ALPHALENS_WINDOW_DIRS="/path/to/af3_prediction;/path/to/af2_prediction" \\
        ChimeraX --exit --cmd "open tests/chimerax_window_check.py"

Run it with a window: Qt's offscreen platform crashes ChimeraX. Each prediction should have a
distogram (AF3 seed-*_distogram/ or AF2 result pickles with a 'distogram' key) so that every view
is exercised; with two or more seeds or models the agreement view is checked too.

Environment:
  ALPHALENS_WINDOW_DIRS     prediction folders (or model files), separated by ';' (required)
  ALPHALENS_SOURCE          'checkout' (default): test src/ of this checkout without installing it;
                            'installed': test the installed bundle, e.g. a release wheel
  ALPHALENS_WINDOW_OUT      folder for results.txt, screenshots and CSV exports (default: a new
                            temporary folder, printed at the start)
  ALPHALENS_WINDOW_TIMEOUT  seconds to wait for each background read (default 1800)
  ALPHALENS_WINDOW_AGREEMENT  '0' skips the agreement view (it reads every seed or model)

Results: one PASS/FAIL line per check in results.txt and the log, then a SUMMARY line. ChimeraX
exits with status 1 if any check failed or a prediction could not be tested, else 0.
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


def _package(session):
    if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
        return 'chimerax.alphalens'
    src = Path(__file__).resolve().parents[1] / 'src'
    name = 'chimerax.alphalens_window'
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
        self.timeout = float(os.environ.get('ALPHALENS_WINDOW_TIMEOUT', 1800))

    def log(self, line):
        with self.results.open('a') as f:
            f.write(line + '\n')
        self.session.logger.info(line)

    def check(self, name, ok, detail=''):
        self.failed += not ok
        self.log(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')

    def pump(self, seconds=0.3):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.session.ui.processEvents()
            time.sleep(0.02)

    def wait_for(self, cond, what):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > self.timeout:
                self.check(f'{what} within {self.timeout:.0f} s', False, 'timed out')
                return False
            self.session.ui.processEvents()
            time.sleep(0.05)
        self.log(f'INFO  {what} after {time.time() - t0:.1f} s')
        return True


def check_prediction(c, pkg, label, path):
    session = c.session
    contacts = importlib.import_module(pkg + '.contacts')
    c.log(f'\n===== {label}: {path}')
    t0 = time.time()
    pae = run(session, f'alphalens open {quote(path)}')        # As a user: no "wait"
    s = pae.structure
    plot = s._alphalens_plot
    c.check(f'{label} plot opened', plot is not None and not plot.closed(), f'{s}, {time.time() - t0:.1f} s')
    if not c.wait_for(lambda: pae.has_distogram, f'{label} distogram read in the background'):
        return
    c.pump(0.5)

    tabs = {k: b.text() for k, b in plot._tabs.items()}
    c.check(f'{label} tab labels', tabs['contact'] == 'Contact probability'
            and tabs['model'] == 'Model distances', str(tabs))
    for key in ('pae', 'contact', 'model', 'spread'):
        plot._tabs[key].click()
        c.pump(0.4)
        heading = plot._heading.text().split('</b>')[0].replace('<html><b>', '')
        c.check(f'{label} view {key}', plot._view_mode == key, heading[:100])
        plot.save_image(str(c.out / f'{label}_{key}.png'))

    # Model distances: model contacts colored by P over grey distances, in one matrix
    plot._tabs['model'].click()
    c.pump(0.4)
    vals = plot._render['vals']
    mask = contacts.model_contacts(pae, 8.0)[0]
    dist = contacts.model_distances(pae)[0]
    P = pae.metrics['contact']
    far = ~mask & ~np.isnan(dist)
    expected_contacts = np.maximum(np.sqrt(np.clip(P[mask], 0, 1)), 1 / 6)
    expected_far = -1 - np.clip((dist[far] - 8) / (contacts.DISTANCE_FAR - 8), 0, 1)
    c.check(f'{label} Model distances values', int((vals >= 0).sum()) == int(mask.sum())
            and np.allclose(vals[mask], expected_contacts, atol=1e-5)
            and np.allclose(vals[far], expected_far, atol=1e-5),
            f'{int(np.triu(mask, 1).sum())} contacts, distances up to {np.nanmax(dist):.0f} Å')
    r = plot._render
    rgb = plot._render_region(0, r['n'], 0, r['n'], r['f'])
    grey = (rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2]) & (rgb[..., 0] < 250)
    green = rgb[..., 1] > rgb[..., 0] + 10
    c.check(f'{label} Model distances image: grey distances and green contacts',
            bool(grey.any() and green.any()), f'{int(grey.sum())} grey, {int(green.sum())} green pixels')
    c.check(f'{label} Model distances colour bar', not plot._cbar.pixmap().isNull())
    for name in ('distance', 'model'):            # "model": the view's name up to 0.5
        run(session, f'alphalens #{s.id_string} mode pae')
        run(session, f'alphalens #{s.id_string} mode {name}')
        c.pump(0.3)
        c.check(f'{label} "mode {name}" command', plot._view_mode == 'model')

    # Hover, structure contacts, clicked contact, pLDDT colors, CSV export
    pairs, _ = contacts.contact_pairs(pae, 0.5)
    i, j = (int(v) for v in (pairs[0] if len(pairs) else
                             np.unravel_index(np.argmax(np.triu(P, 1)), P.shape)))
    plot._report(j, i)
    info = plot._info.text()
    c.check(f'{label} hover', all(t in info for t in ('PAE', 'P(d<', 'σ', 'E[d]', 'model ')), info[:160])
    plot.set_threshold(0.5)
    plot._contacts_button.click()
    c.pump(0.5)
    group = s.pseudobond_group(contacts.GROUP_NAME, create_type=None)
    n = group.num_pseudobonds if group else 0
    colors = group.pseudobonds.colors.astype(int) if n else np.zeros((0, 4), int)
    c.check(f'{label} Show Contacts', n == min(len(pairs), contacts.MAX_LINES)
            and bool((colors[:, 2] > colors[:, 0]).all()), f'{n} blue lines, {len(pairs)} pairs at P ≥ 0.5')
    plot._pick_contact(i, j)
    c.pump(0.3)
    picked = s.pseudobond_group(contacts.PICK_GROUP_NAME, create_type=None)
    c.check(f'{label} clicked contact', picked is not None and picked.num_pseudobonds == 1
            and picked.pseudobonds.colors[0].tolist()[:3] == list(contacts.PICK_RGB))
    plot._color_plddt()
    c.pump(0.3)
    ca = s.atoms.filter(s.atoms.names == 'CA')
    bands = np.array([(255, 125, 69), (255, 219, 19), (101, 203, 243), (0, 83, 214)])
    bf = ca.bfactors * (100 if ca.bfactors.max() <= 1 else 1)
    band = np.where(bf <= 50, 0, np.where(bf <= 70, 1, np.where(bf <= 90, 2, 3)))
    near = np.abs(ca.colors[:, None, :3].astype(int) - bands[None]).max(-1) <= 1
    c.check(f'{label} Color pLDDT bands', bool(near[np.arange(len(band)), band].all()))
    out = c.out / f'{label}_contacts.csv'
    plot.save_contacts_csv(str(out))
    rows = list(csv.DictReader(out.open()))
    c.check(f'{label} Export CSV', len(rows) == len(pairs), f'{len(rows)} rows')

    # Agreement between seeds or models
    tab = plot._tabs['agreement']
    if os.environ.get('ALPHALENS_WINDOW_AGREEMENT', '1') == '0':
        c.log(f'INFO  {label} agreement skipped')
    elif len(plot._agreement_runs) >= 2:
        tab.click()
        if c.wait_for(lambda: pae.agreement is not None or plot._agreement_error is not None,
                      f'{label} agreement read'):
            c.pump(0.5)
            agr = pae.agreement
            c.check(f'{label} agreement view', agr is not None and plot._view_mode == 'agreement',
                    f'{agr.n} {agr.kind}s' if agr else str(plot._agreement_error))
            plot.save_image(str(c.out / f'{label}_agreement.png'))
    else:
        c.check(f'{label} agreement tab disabled with a reason', not tab.isEnabled()
                and len(tab.toolTip()) > 20, tab.toolTip()[:100])

    # Another model of the prediction from the Model menu (as a user's choice: "activated")
    menu = plot._model_combo
    target = next((k for k in range(menu.count()) if k != menu.currentIndex()
                   and k < len(plot._models) and plot._models[k]['path']), None)
    if target is not None:
        menu.setCurrentIndex(target)
        menu.activated.emit(target)
        c.pump(0.5)
        c.wait_for(lambda: plot.pae.has_distogram or plot.pae.has_contact, f'{label} other model read')
        c.pump(0.5)
        plot._tabs['model'].click()
        c.pump(0.3)
        c.check(f'{label} switch model', plot.pae.structure is not s and plot._view_mode == 'model',
                menu.itemText(target))
    run(session, 'close')
    c.pump(0.5)


def main(session):
    dirs = [d for d in os.environ.get('ALPHALENS_WINDOW_DIRS', '').split(';') if d.strip()]
    out = Path(os.environ.get('ALPHALENS_WINDOW_OUT') or tempfile.mkdtemp(prefix='alphalens-window-'))
    out.mkdir(parents=True, exist_ok=True)
    c = Checks(session, out)
    if not session.ui.is_gui:
        c.check('window available (run without --nogui)', False)
    elif not dirs:
        c.check('ALPHALENS_WINDOW_DIRS lists prediction folders', False)
    else:
        pkg = _package(session)
        c.log(f'INFO  testing {pkg}; output in {out}')
        for k, path in enumerate(dirs):
            label = f'{k + 1}_{Path(path.rstrip("/")).name}'.replace(':', '_')
            try:
                check_prediction(c, pkg, label, path.strip())
            except Exception:
                c.check(f'{label} completed without an exception', False, traceback.format_exc()[-600:])
                try:
                    run(session, 'close')
                except Exception:
                    pass
    total = sum(1 for line in c.results.read_text().splitlines() if line.startswith(('PASS', 'FAIL')))
    c.log(f'SUMMARY {total - c.failed} passed, {c.failed} failed; results in {c.results}')
    os._exit(1 if c.failed else 0)


main(session)
