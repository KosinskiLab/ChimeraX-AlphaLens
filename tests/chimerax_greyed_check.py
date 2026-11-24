"""What is greyed out in the plot window and the prediction dialog, and why, in a real ChimeraX window.

    ALPHALENS_GREYED_DIRS="<prediction>;<prediction>;..." ALPHALENS_TEST_OUTPUT=<new folder> \\
        ChimeraX --cmd "open tests/chimerax_greyed_check.py"

Each prediction is a folder or a model file. Rules checked for each, whatever the layout:
  * a view tab is on exactly when its data is there (AlphaLensPlot.available_modes), and an off
    tab's tooltip gives the reason;
  * Show Contacts, Export and the contact-probability slider are on exactly when there are
    contact probabilities (Export not while the distogram loads); off, they say what they need;
  * "hide in matrix too" is on in the contact-probability view only;
  * a Model menu row is off exactly when its model's file is not there, and says so;
  * in the context menu, Hide selected needs selected rows, Show hidden hidden ones, and
    Re-predict selected interface a box;
  * the prediction dialog offers "Reuse and crop the MSA" exactly when the source's own MSA is
    found (predict_ui.source_alignment), with the reason when not; Predict is off only with its
    reason above the buttons.
With two or more predictions, the first one's dialog stays open while its window shows the
second, as the Model menu does, then the first again: the MSA to reuse and the guessed feature
IDs follow the window. Last, closing the shown structure turns Predict off and says why.

Environment:
  ALPHALENS_SOURCE  'checkout' (default) tests src/ of this checkout; 'installed' the bundle
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
FOLDS = [Path(p).expanduser() for p in os.environ.get('ALPHALENS_GREYED_DIRS', '').split(';') if p]
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
cmd = importlib.import_module('chimerax.alphalens.cmd')
data = importlib.import_module('chimerax.alphalens.data')
ui = importlib.import_module('chimerax.alphalens.predict_ui')
pf = importlib.import_module('chimerax.alphalens.predict_features')
from chimerax.atomic import AtomicStructure, Residue
from Qt.QtCore import Qt
from Qt.QtWidgets import QMenu
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


def model_path(path):
    return path if path.is_file() else Path(data.structure_path_for_directory(str(path)))


def settled(plot):
    return pump(lambda: plot._load_progress is None and not plot.switch_pending, 600)


def open_plot(path):
    run(session, f'alphalens open "{model_path(path)}"')
    s = [m for m in session.models.list(type=AtomicStructure) if getattr(m, '_alphalens_plot', None)][-1]
    settled(s._alphalens_plot)
    pump(lambda: False, 0.5)
    return s


def drag_box(plot, half=20):
    '''A box around the inter-chain protein pair of lowest PAE; None for a single chain.'''
    m = np.asarray(plot.pae.matrix('pae'), float)
    tokens = plot.pae.row_residues_or_atoms()
    chains = np.array([r.chain_id for r in tokens])
    protein = np.array([isinstance(r, Residue) and r.polymer_type == Residue.PT_AMINO for r in tokens])
    pae = (m + m.T) / 2
    pae[chains[:, None] == chains[None, :]] = np.inf
    pae[~protein, :] = pae[:, ~protein] = np.inf
    if not np.isfinite(pae).any():
        return None
    i, j = np.unravel_index(np.argmin(pae), pae.shape)

    def span(k):
        same = np.flatnonzero(chains == chains[k])
        return max(same[0], k - half), min(same[-1], k + half)
    (r0, r1), (c0, c1) = span(i), span(j)
    plot._rectangle_select((int(c0), int(r0)), (int(c1), int(r1)))
    pump(lambda: False, 0.3)
    return f'{chains[i]} x {chains[j]}'


def menu_actions(plot):
    menu = QMenu()
    plot._fill_context_menu(menu, 0, 0)
    found = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
    menu.deleteLater()
    return found


def check_plot(label, plot):
    avail = plot.available_modes()
    for key, b in plot._tabs.items():
        on = key in avail and (key != 'agreement' or plot._agreement_status is None)
        check(f'{label}: {key} tab {"on" if on else "off: " + b.toolTip()[:90]}',
              b.isEnabled() == on and (on or len(b.toolTip()) > 20), b.toolTip())
    has, loading = plot.pae.has_contact, plot._load_progress is not None
    for name, widget, on in (('Show Contacts', plot._contacts_button, has),
                             ('Export', plot._export_button, has and not loading),
                             ('contact-probability slider', plot._threshold_row, has)):
        says = widget.toolTip() if hasattr(widget, 'toolTip') else ''
        check(f'{label}: {name} {"on" if on else "off"}', widget.isEnabled() == on
              and (on or name == 'contact-probability slider' or 'Needs' in says or 'Available when' in says), says)
    if 'contact' in avail:
        plot._tabs['contact'].click()
        pump(lambda: False, 0.2)
        check(f'{label}: "hide in matrix too" on in the contact view', plot._mask_box.isEnabled())
    plot._tabs['pae'].click()
    pump(lambda: False, 0.2)
    check(f'{label}: "hide in matrix too" off in the PAE view', not plot._mask_box.isEnabled())
    combo, rows = plot._model_combo, plot._model_combo.model()
    missing = 0
    for i, m in enumerate(plot._models):
        on = bool(rows.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        text = combo.itemText(i)
        missing += m['path'] is None
        if on != (m['path'] is not None) or (('not available locally' in text) != (m['path'] is None)):
            check(f'{label}: Model menu row {text!r}', False, m)
            break
    else:
        check(f'{label}: Model menu rows off exactly for the {missing} model(s) not here '
              f'(of {len(plot._models)})', True)
    actions = menu_actions(plot)
    check(f'{label}: Re-predict selected interface off without a box',
          plot._prediction_region is not None or actions.get('Re-predict selected interface…') is False, actions)
    check(f'{label}: Show hidden off with nothing hidden',
          plot._selected_hidden.any() or actions.get('Show hidden') is False, actions)


def check_dialog(label, plot, d):
    found = ui.source_alignment(plot)
    where = found and found.get('where')
    check(f'{label}: reuse {"offered: " + str(found.get("label")) if found else "off"}',
          d.reuse.isEnabled() == (found is not None) and (d.source_msa or {}).get('where') == where
          and (found is not None or 'None found' in d.reuse.toolTip()), d.reuse.toolTip())
    pump(lambda: False, 0.2)
    off = not d.start_button.isEnabled()
    check(f'{label}: Predict {"off: " + d.msa_hint.text()[:80] if off else "on"}',
          not off or (d.msa_hint.isVisible() and d.msa_hint.text() and d.msa_hint.text() == d.start_button.toolTip()),
          d.msa_hint.text())


def main():
    code = 1
    try:
        for path in FOLDS:
            run(session, 'close')
            label = f'{path.parent.name}/{path.name}'
            s = open_plot(path)
            plot = s._alphalens_plot
            check_plot(label, plot)
            where = drag_box(plot)
            if where is None:
                check(f'{label}: one protein chain, no box to re-predict', True)
                continue
            actions = menu_actions(plot)
            check(f'{label}: box {where}: Re-predict selected interface on',
                  actions.get('Re-predict selected interface…') is True, actions)
            hide = plot._selected_to_hide().any()
            check(f'{label}: Hide selected {"on" if hide else "off"}, as there are {"" if hide else "no "}selected rows',
                  actions.get('Hide selected') == bool(hide), actions)
            if hide:
                plot._hide_selected()
                check(f'{label}: Show hidden on once rows are hidden', menu_actions(plot).get('Show hidden') is True)
                plot._show_hidden()
                check(f'{label}: Show hidden off again', menu_actions(plot).get('Show hidden') is False)
            d = plot._predict_interface()
            check_dialog(label, plot, d)
            d.close()
        if len(FOLDS) >= 2:
            run(session, 'close')
            a, b = FOLDS[:2]
            label = f'switch {a.name} -> {b.name}'
            first = open_plot(a)
            plot = first._alphalens_plot
            second = [m for m in run(session, f'open "{model_path(b)}"') if isinstance(m, AtomicStructure)][0]
            drag_box(plot)
            d = plot._predict_interface()
            guessed = d.features_names.text()
            for target, name in ((second, b.name), (first, a.name)):
                cmd.show_structure(session, plot, target, wait=True)
                settled(plot)
                pump(lambda: False, 0.5)
                check(f'{label}: the window shows {name}', plot.pae.structure is target, plot.pae.structure)
                drag_box(plot)
                check_dialog(f'{label}: {name}', plot, d)
                sequences = ui.structure_chain_sequences(plot)
                expected = pf.format_names(pf.guess_names(target.filename, list(sequences), list(sequences.values())))
                check(f'{label}: {name}: feature IDs {d.features_names.text() or "none"}',
                      d.features_names.text() == expected, (d.features_names.text(), expected))
            check(f'{label}: back on the first, the first guess again', d.features_names.text() == guessed,
                  (d.features_names.text(), guessed))
            run(session, f'close #{first.id_string}')
            pump(lambda: False, 0.5)
            check(f'{label}: closing the shown structure turns Predict off and says why',
                  not d.start_button.isEnabled() and 'closed' in d.msa_hint.text(), d.msa_hint.text())
            d.close()
        code = 0 if results and all(r['ok'] for r in results) else 1
    except Exception:
        check('no exception', False, traceback.format_exc())
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    os._exit(code)


main()
