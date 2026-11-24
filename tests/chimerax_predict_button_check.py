"""When Predict is off, and what it says, in a real ChimeraX window (no GPU, no network).

    ALPHALENS_BUTTON_SOURCE=<a two-chain AF2/AF3 prediction folder> ALPHALENS_TEST_OUTPUT=<new folder> \\
        ChimeraX --cmd "open tests/chimerax_predict_button_check.py"

Walks the prediction dialog through every condition that turns Predict off (no box, a box in one
chain, a box beyond the matrix, a modified residue, a box over non-protein rows only, one protein
chain, no alignment, no A3M file, missing feature IDs, a source MSA without a chain of the box, AF3
terms, no or a missing engine, a platform without LocalFold, a running job, unread or unreadable
input, a residue LocalFold cannot fold) and the ones that must not (ligand rows in the box, X, an
engine being rebuilt). Rows no prediction at hand has (a modified residue, non-protein rows) are
made by editing the rows the crop gets (predict_ui.source_rows). Each time Predict is off, the button's tooltip -
shown by a real hover event on the disabled button - and the line above the buttons give the
same reason. Clicking the chosen alignment again keeps it chosen.

Environment:
  ALPHALENS_SOURCE      'checkout' (default) tests src/ of this checkout; 'installed' the bundle
  ALPHALENS_LOCALFOLD   an engine that exists (default: the managed one)
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
ui = importlib.import_module('chimerax.alphalens.predict_ui')
from chimerax.atomic import AtomicStructure

results = []


def check(name, ok, detail=''):
    results.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300]))


def pump(seconds=0.2, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until is None


def hover_tooltip(widget):
    '''The tooltip Qt shows when the pointer rests on widget (enabled or not).'''
    from Qt.QtGui import QHelpEvent
    from Qt.QtCore import QEvent
    from Qt.QtWidgets import QApplication, QToolTip
    pos = widget.rect().center()
    for attempt in range(3):            # A window just opened may not show it the first time
        QToolTip.hideText()
        QApplication.sendEvent(widget, QHelpEvent(QEvent.Type.ToolTip, pos, widget.mapToGlobal(pos)))
        if pump(0.4, until=QToolTip.isVisible):
            return QToolTip.text()
    return ''


def expect(name, dialog, off=None, says=None):
    '''Predict off (with the reason saying `says`) or on.'''
    dialog.refresh()
    pump(0.1)
    button = dialog.start_button
    if off:
        tip = hover_tooltip(button)
        line = dialog.msa_hint.text() if not dialog.msa_hint.isHidden() else ''
        check(name, not button.isEnabled() and says in button.toolTip() and says in tip and says in line,
              f'enabled={button.isEnabled()} tooltip={tip!r} line={line!r}')
    else:
        check(name, button.isEnabled() and dialog.msa_hint.isHidden(),
              f'enabled={button.isEnabled()} tooltip={button.toolTip()!r}')


def main():
    code = 1
    try:
        run(session, f'alphalens open "{os.environ["ALPHALENS_BUTTON_SOURCE"]}" wait true')
        s = next(m for m in session.models.list(type=AtomicStructure) if getattr(m, '_alphalens_plot', None))
        plot = s._alphalens_plot
        n_a = len(s.chains[0].residues)
        d = plot._predict_interface()
        pump(2, until=lambda: d.isVisible() and d.windowHandle() is not None and d.windowHandle().isExposed())
        d.output.setText(str(OUT / 'jobs'))
        d.backend.setCurrentIndex(d.backend.findData('af2'))
        expect('region: no box', d, True, 'Drag a box')
        plot._set_prediction_region((5, 40, 10, 50))
        expect('region: a box in one chain names it and the block to use', d, True, 'covers chain A only')
        plot._set_prediction_region((5, 40, n_a + 5, n_a + 40))
        for b in (d.reuse, d.features, d.use_a3m, d.search):
            b.setChecked(False)
        expect('region: no alignment chosen', d, True, 'Choose an alignment')
        d.search.click()
        expect('region: search chosen', d, False)
        d.search.click()                    # Clicking the chosen option again must keep it chosen
        check('clicking the chosen alignment keeps it', d.search.isChecked())
        expect('region: still on after a second click', d, False)
        d.use_a3m.click()
        d.msa.clear()
        expect('region: A3M chosen without a file', d, True, 'Enter the A3M file')
        d.features.click()
        d.features_names.setText('A: P0A836')
        expect('region: a chain without a feature ID', d, True, 'feature ID (UniProt accession) of chain B')
        d.features_root.clear()
        expect('region: no feature folder', d, True, 'Enter where the features are')
        d.search.click()
        d.backend.setCurrentIndex(d.backend.findData('af3'))
        d.af3_terms.setChecked(False)
        expect('region: AF3 terms not accepted', d, True, 'AF3 terms')
        d.af3_terms.setChecked(True)
        expect('region: AF3 terms accepted', d, False)
        d.backend.setCurrentIndex(d.backend.findData('af2'))
        engine = d.executable.text()
        d.executable.setText(str(OUT / 'missing-engine'))
        expect('region: missing engine', d, True, 'No LocalFold engine at')
        plain = OUT / 'not-executable'
        plain.write_text('')
        d.executable.setText(str(plain))
        expect('region: engine file not executable', d, True, 'No LocalFold engine at')
        d.executable.setText(engine)
        d.is_busy = lambda: True            # A running job or setup
        expect('region: a prediction running', d, True, 'is running')
        del d.is_busy
        expect('region: ready again', d, False)
        total = len(plot.pae.row_residues_or_atoms())
        plot._set_prediction_region((5, 40, total + 5, total + 20))
        expect('region: a box beyond the matrix', d, True, 'structure changed since the box was drawn')
        plot._set_prediction_region((5, 40, n_a + 5, n_a + 40))
        real_rows = ui.source_rows

        def edited(change):
            def rows(pae):
                r = list(real_rows(pae))
                change(r)
                return r
            return rows

        def none(r, lo, hi):
            r[lo:hi] = [None] * (hi - lo)

        for name, change, says in (
                ('a modified residue inside the box', lambda r: none(r, 20, 21), 'modified or non-protein residue'),
                ('a box over non-protein rows only', lambda r: (none(r, 5, 41), none(r, n_a + 5, n_a + 41)),
                 'covers no protein residues'),
                ('one protein chain', lambda r: none(r, n_a, len(r)), 'one protein chain')):
            ui.source_rows = edited(change)
            d.source_changed()
            expect(f'region: {name}', d, True, says)
        ui.source_rows = real_rows
        d.source_changed()
        expect('region: the rows restored', d, False)
        saved = d.source_msa
        d.source_msa = dict(chains={'A': dict(unpaired=str(OUT / 'A.a3m'), paired=None)}, where='msas',
                            origin='test', label='AlphaFold Server')
        d.reuse.setEnabled(True)
        d.reuse.setChecked(True)
        expect('region: the source MSA has no chain B of the box', d, True, 'No MSA of chain B in msas')
        d.source_msa = saved
        d._show_source_msa()
        d.search.click()
        d.executable.setText('')
        expect('region: no engine given', d, True, 'Enter the LocalFold engine')
        d.executable.setText(engine)
        real_platform = ui.native_platform
        ui.native_platform = lambda: None
        d.needs_setup = lambda: True
        expect('region: a platform LocalFold does not run on', d, True, 'cannot be installed here')
        ui.native_platform = real_platform
        del d.needs_setup
        expect('region: ready at the end', d, False)
        d.close()
        pump(0.3)

        # Sequence input
        d = ui.show_prediction_dialog(session=session)
        pump(2, until=lambda: d.isVisible() and d.windowHandle() is not None and d.windowHandle().isExposed())
        d.output.setText(str(OUT / 'jobs'))
        d.backend.setCurrentIndex(d.backend.findData('af2'))
        expect('sequence: nothing pasted', d, True, 'Paste a sequence')
        d.input_text.setPlainText('>a\nMKVLAAGIVGLAAHE\n>b\nMSTNPKPQRKTKRNTNRRPQDVK')
        expect('sequence: pasted, not read', d, True, 'Click Load sequences')
        d.load_button.click()
        pump(10, until=lambda: not d.lookup_timer.isActive())
        for b in (d.reuse, d.features, d.use_a3m, d.search):
            b.setChecked(False)
        expect('sequence: read, no alignment', d, True, 'Choose an alignment')
        d.search.click()
        expect('sequence: ready', d, False)
        d.ranges[0].setText('9-3')
        expect('sequence: a range backwards', d, True, 'Chain A')
        d.ranges[0].setText('')
        d.input_text.setPlainText('>a\nMKVLAXGIVGLAAHE\n>b\nMSTNPKPQRKTKRNTNRRPQDVK')
        d.load_button.click()
        pump(10, until=lambda: not d.lookup_timer.isActive())
        expect('sequence: X is folded (UNK)', d, False)
        d.input_text.setPlainText('>a\nMKVLAUGIVGLAAHE\n>b\nMSTNPKPQRKTKRNTNRRPQDVK')
        d.load_button.click()
        pump(10, until=lambda: not d.lookup_timer.isActive())
        expect('sequence: U is named, with what to use instead', d, True, 'U at A:6 cannot be folded')
        d.input_text.setPlainText('TP53')            # A gene symbol is not looked up
        d.load_button.click()
        pump(10, until=lambda: not d.lookup_timer.isActive())
        expect('sequence: unreadable input says why', d, True, 'could not be read')
        d.close()
        code = 0 if all(r['ok'] for r in results) else 1
    except Exception:
        results.append(dict(name='exception', ok=False, detail=traceback.format_exc()))
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    for r in results:
        session.logger.info(('PASS ' if r['ok'] else 'FAIL ') + r['name'] + ('' if r['ok'] else f"  {r['detail']}"))
    os._exit(code)


main()
