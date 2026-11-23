"""Check AlphaLens's menus, help and zoom in a real ChimeraX window, with two open predictions.

    ALPHALENS_GUI_DIRS="/path/to/prediction_1;/path/to/prediction_2" \\
        ChimeraX --exit --cmd "open tests/chimerax_gui_check.py"

Covers what a GUI user meets first: Tools → AlphaLens with several predictions open, the Model
menu listing the other open predictions and switching to one, the Help buttons and context
menus, zoom keeping its point after a pan, the pLDDT slider's value staying off the plot,
and the pLDDT track kept in view as a numbered footer when zoomed in.
Two predictions of the same complex (AF2 and AF3, the same number of residues) are the hard case
for keeping their data apart.

Environment:
  ALPHALENS_GUI_DIRS   two prediction folders separated by ';' (required)
  ALPHALENS_SOURCE     'checkout' (default): src/ of this checkout without installing it;
                       'installed': the installed bundle, whose help pages are then opened too
  ALPHALENS_GUI_OUT    folder for results.txt and images (default: a new temporary folder)

Results: one PASS/FAIL line per check in results.txt and the log, then a SUMMARY line. ChimeraX
exits with status 1 if any check failed, else 0.
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from chimerax.core.commands import quote_path_if_necessary as quote, run

INSTALLED = os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed'


def _package(session):
    if INSTALLED:
        return 'chimerax.alphalens'
    src = Path(__file__).resolve().parents[1] / 'src'
    name = 'chimerax.alphalens_gui'
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
        self.failed += not ok
        self.log(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')

    def pump(self, seconds=0.3):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.session.ui.processEvents()
            time.sleep(0.02)

    def wait_for(self, cond, what, timeout=900):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout:
                self.check(f'{what} within {timeout} s', False, 'timed out')
                return False
            self.session.ui.processEvents()
            time.sleep(0.05)
        return True


def plots(session, tool):
    return [t for t in session.tools.list() if isinstance(t, tool.AlphaLensPlot) and not t.closed()]


def entries(plot, kind):
    return [(k, e[1]) for k, e in enumerate(plot._menu_entries) if e is not None and e[0] == kind]


def check_tools_and_model_menu(c, pkg, dirs):
    session = c.session
    cmd = importlib.import_module(pkg + '.cmd')
    tool = importlib.import_module(pkg + '.tool')
    data = importlib.import_module(pkg + '.data')
    structures = []
    for d in dirs:
        spath = data.structure_path_for_directory(d, 0)
        structures.append(run(session, f'open {quote(spath)}')[0])
    s1, s2 = structures
    c.pump(0.5)
    c.check('two predictions open, no AlphaLens window', not plots(session, tool), f'{s1}, {s2}')

    # Tools → Structure Prediction → AlphaLens, three times
    p1 = cmd.start_tool_from_menu(session)
    c.pump(0.5)
    c.check('Tools menu: a window for the first prediction', p1 is not None and p1.pae.structure is s1
            and len(plots(session, tool)) == 1, str(p1 and p1.pae.structure))
    p2 = cmd.start_tool_from_menu(session)
    c.pump(0.5)
    c.check('Tools menu again: a second window for the second prediction', p2 is not None
            and p2 is not p1 and p2.pae.structure is s2 and len(plots(session, tool)) == 2,
            str(p2 and p2.pae.structure))
    p3 = cmd.start_tool_from_menu(session)
    c.pump(0.3)
    c.check('Tools menu with both shown: no third window', p3 in (p1, p2) and len(plots(session, tool)) == 2)
    for p in (p1, p2):
        c.wait_for(lambda p=p: p.pae.has_contact, f'{p.pae.structure} contacts read')

    # The Model menu lists the other open prediction and a folder to open
    c.pump(0.3)
    others = entries(p1, 'structure')
    menu = p1._model_combo
    c.check('Model menu enabled, lists the other open prediction', menu.isEnabled()
            and [s for _, s in others] == [s2] and 'another AlphaLens window' in menu.itemText(others[0][0]),
            menu.itemText(others[0][0]) if others else 'no entry')
    c.check('Model menu ends with "Open a prediction folder…"', p1._menu_entries[-1] == ('open', None)
            and menu.itemText(menu.count() - 1).startswith('Open a prediction folder'))

    # Choosing a prediction that has its own window brings that window forward
    menu.activated.emit(others[0][0])
    c.wait_for(lambda: not p1.switch_pending, 'files read')
    c.pump(0.3)
    c.check('Model menu: a prediction with its own window stays there', p1.pae.structure is s1
            and p2.pae.structure is s2 and menu.currentIndex() != others[0][0])

    # Without its window, choosing it shows it in this one
    p2.delete()
    c.pump(0.3)
    others = entries(p1, 'structure')
    c.check('menu refreshed when the other window closed', others
            and 'another AlphaLens window' not in menu.itemText(others[0][0]), menu.itemText(others[0][0]) if others else '')
    agreement_before = p1.pae.agreement
    menu.activated.emit(others[0][0])
    c.wait_for(lambda: not p1.switch_pending, 'files read')
    c.pump(0.5)
    c.wait_for(lambda: p1.pae.has_contact, 'the second prediction read in the first window')
    c.pump(0.5)
    c.check('Model menu: another prediction shown in this window', p1.pae.structure is s2
            and s2._alphalens_plot is p1 and getattr(s1, '_alphalens_plot', None) is None
            and s1.display and s2.display, f'{p1.pae.structure}')
    c.check('its data are its own', p1.pae is s2.alphafold_pae and (p1.pae.agreement is None
            or p1.pae.agreement is not agreement_before) and p1._view_mode in p1.available_modes())
    c.check('the first prediction is listed for going back', [s for _, s in entries(p1, 'structure')] == [s1])
    menu.activated.emit(entries(p1, 'structure')[0][0])
    c.wait_for(lambda: not p1.switch_pending, 'files read')
    c.pump(0.5)
    c.wait_for(lambda: p1.pae.has_contact, 'back to the first prediction')
    c.check('and back again', p1.pae.structure is s1 and s1._alphalens_plot is p1)
    return p1


def check_zoom(c, plot):
    from Qt.QtCore import QPointF
    v = plot._view
    v.reset_zoom()
    c.pump(0.3)
    vr = v.viewport().rect()

    def drift(anchor, zoom):
        before = v.scene_point(anchor)
        zoom()
        c.pump(0.1)
        after = v.scene_point(anchor)
        return ((after.x() - before.x()) ** 2 + (after.y() - before.y()) ** 2) ** 0.5 * v.transform().m11()

    a = QPointF(vr.width() * 0.3, vr.height() * 0.4)
    d = drift(a, lambda: v.zoom_by(4, anchor=a))
    c.check('zoom keeps the point under the pointer', v.zoomed and d < 1.5, f'{d:.2f} px')
    hs, vs = v.horizontalScrollBar(), v.verticalScrollBar()
    hs.setValue(hs.value() + 60)
    vs.setValue(vs.value() + 40)            # A pan, as Shift-drag does it
    c.pump(0.2)
    b = QPointF(vr.width() * 0.6, vr.height() * 0.55)
    d = drift(b, lambda: v.zoom_by(1.5, anchor=b))
    c.check('after a pan, zoom keeps the point under the pointer', d < 1.5, f'{d:.2f} px')
    centre = QPointF(vr.center())
    d = drift(centre, lambda: v.zoom_by(1.3))
    c.check('+ / − zoom about the middle of the view', d < 1.5, f'{d:.2f} px')

    hint = v.hint_text() or ''
    c.check('zoomed in, the corner says how to pan', 'Shift-drag' in hint, hint)
    path = c.out / 'zoomed.png'
    plot.save_image(str(path))
    c.check('saving an image leaves the hint on screen', path.exists() and v.hint_visible)
    v.reset_zoom()
    v._hovering = True
    c.check('zoomed out, the corner says how to zoom', 'zoom' in (v.hint_text() or ''), v.hint_text() or '')
    v._hovering = False
    c.check('no hint when the pointer is elsewhere', v.hint_text() is None)


def check_plddt_track(c, plot):
    '''
    The pLDDT track and its hide slider: the slider's value ("hide <70") covers nothing of the
    matrix or the track; zoomed in, the track stays in view as a numbered footer, which answers
    hover as the track does, until panning brings the track itself into view.
    '''
    from Qt.QtCore import QPointF, QRect, QPoint, Qt, QEvent
    from Qt.QtGui import QMouseEvent
    v = plot._view
    w = plot.tool_window.ui_area.window()
    size = w.size()
    plot._plddt_slider.setValue(70)

    def covers_nothing(what):
        c.pump(0.6)
        label = plot._plddt_value
        if not label.isVisible():
            c.log(f'INFO  pLDDT slider value hidden ({what})')
            return
        matrix = QRect(v.mapFromScene(QPointF(0, 0)), v.mapFromScene(QPointF(v.n, v.n)))
        track = QRect(v.mapFromScene(QPointF(0, v.track[0])), v.mapFromScene(QPointF(v.n, v.track[1])))
        g = label.geometry()
        over = [name for name, r in (('matrix', matrix), ('track', track)) if g.intersects(r)]
        c.check(f'pLDDT slider value "{label.text()}" covers nothing of the plot ({what})',
                not over and v.viewport().rect().contains(g), f'{g.getRect()} over {over or "nothing"}')

    covers_nothing('window as opened')
    c.check('not zoomed in: no footer', v.footer_rect() is None)
    w.resize(size.width(), 1300)            # Taller than wide: no room right of the slider
    covers_nothing('tall window')
    w.resize(size)
    c.pump(0.5)
    v.reset_zoom()
    c.pump(0.4)
    moved_late = []                         # Slider or its value drawn a moment where they were
    for step in range(6):
        v.zoom_by(1.25, anchor=QPointF(v.viewport().width() * 0.45, v.viewport().height() * 0.4))
        now = (plot._plddt_slider.geometry().getRect(), plot._plddt_slider.isVisible(),
               plot._plddt_value.geometry().getRect(), plot._plddt_value.isVisible())
        c.pump(0.3)
        later = (plot._plddt_slider.geometry().getRect(), plot._plddt_slider.isVisible(),
                 plot._plddt_value.geometry().getRect(), plot._plddt_value.isVisible())
        if now != later:
            moved_late.append((step, now, later))
    c.check('zooming in: the hide slider and its value are in place in the same frame', not moved_late,
            str(moved_late[:1]))
    v.reset_zoom()
    v.zoom_by(3)                            # About the middle: the track is below the view
    c.pump(0.6)
    footer = v.footer_rect()
    slider = plot._plddt_slider.geometry()
    c.check('zoomed in: the track stays in view as a footer', footer is not None
            and footer.bottom() == v.viewport().rect().height(), str(footer))
    if footer is not None:
        c.check('the hide slider sits on the footer, its value drawn there', footer.toRect().intersects(slider)
                and not plot._plddt_value.isVisible() and v.footer_note == 'hide <70', v.footer_note)
        vis = v.mapToScene(v.viewport().rect()).boundingRect()
        rows = v.tick_rows(max(0, int(vis.left())), min(v.n, int(vis.right()) + 1))
        c.check('the footer numbers the visible residues', len(rows) >= 2, str(rows[:6]))
        bars = v.footer_bars(footer)
        pos = QPointF(footer.center().x(), bars.center().y())
        x, y = v._scene_position(QMouseEvent(QEvent.Type.MouseMove, pos, v.viewport().mapToGlobal(pos.toPoint()).toPointF(),
                                            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
        plot._report(int(x), int(y))
        text = plot._info.text()
        c.check('hovering the footer gives the residue under the pointer and its pLDDT', 'pLDDT' in text
                and abs(x - vis.center().x()) < 2, text)
        plot._set_info('')
        path = c.out / 'footer.png'
        plot.save_image(str(path))
    hs, vs = v.horizontalScrollBar(), v.verticalScrollBar()
    vs.setValue(vs.maximum())               # Panned down to the track itself
    c.pump(0.6)
    c.check('panned down to the track: no footer, the track in place', v.footer_rect() is None
            and v.natural_track_rect().bottom() <= v.viewport().rect().height())
    hs.setValue(hs.maximum())
    covers_nothing('zoomed in, bottom right')
    hs.setValue((hs.minimum() + hs.maximum()) // 2)
    covers_nothing('zoomed in, matrix wider than the view')
    v.reset_zoom()
    plot._plddt_slider.setValue(0)
    c.pump(0.3)


def check_help(c, pkg, plot):
    session = c.session
    helpdocs = importlib.import_module(pkg + '.helpdocs')
    c.check('plot: Help button', plot._help_button.text() == 'Help')
    c.check('plot: Help in the context menu (tool help set)', bool(plot.help), plot.help)
    from chimerax.core.commands import cli
    url = cli.command_url('alphalens')
    c.check('"help alphalens" points at the command reference', bool(url) and 'commands' in url, str(url))
    if INSTALLED:
        from chimerax.core import toolshed
        installed = [d for d in toolshed.get_help_directories()
                     if os.path.exists(os.path.join(d, 'user', 'tools', helpdocs.PLOT))]
        c.check('help pages installed with the bundle', bool(installed) and plot.help.startswith('help:'),
                f'{installed[:1]} {plot.help}')
        images = [os.path.exists(os.path.join(installed[0], 'user', 'tools', 'alphalens_images', n))
                  for n in ('views.png', 'contacts.png')] if installed else []
        c.check('help images installed', images and all(images))
        from chimerax.help_viewer.tool import HelpUI
        plot._help_button.click()
        c.pump(2.0)
        viewers = [t for t in session.tools.list() if isinstance(t, HelpUI)]
        c.check('Help opens the help viewer', bool(viewers))
        for t in viewers:
            t.delete()
    else:
        # The pages of an installed AlphaLens have the same names, and ChimeraX finds those
        from chimerax.core import toolshed
        installed = any(os.path.exists(os.path.join(d, 'user', 'tools', helpdocs.PLOT))
                        for d in toolshed.get_help_directories())
        c.check('checkout run: help points at the installed pages, else at GitHub',
                plot.help.startswith('help:' if installed else helpdocs.ONLINE), plot.help)

    predict_ui = importlib.import_module(pkg + '.predict_ui')
    from Qt.QtWidgets import QLabel
    d = predict_ui.show_prediction_dialog(session=session)
    c.pump(0.5)
    try:
        c.check('prediction dialog: Help button', d.help_button.text() == 'Help')
        actions = [a.text() for a in d.context_menu().actions()]
        c.check('prediction dialog: context menu with Help', 'Help' in actions, str(actions))
        credit = [w.text() for w in d.findChildren(QLabel) if 'Ovchinnikov' in w.text()]
        c.check('prediction dialog credits LocalFold', bool(credit), credit[0] if credit else '')
    finally:
        d.close()
        c.pump(0.3)


def main(session):
    dirs = [d.strip() for d in os.environ.get('ALPHALENS_GUI_DIRS', '').split(';') if d.strip()]
    out = Path(os.environ.get('ALPHALENS_GUI_OUT') or tempfile.mkdtemp(prefix='alphalens-gui-'))
    out.mkdir(parents=True, exist_ok=True)
    c = Checks(session, out)
    if not session.ui.is_gui:
        c.check('window available (run without --nogui)', False)
    elif len(dirs) != 2:
        c.check('ALPHALENS_GUI_DIRS lists two prediction folders', False)
    else:
        pkg = _package(session)
        c.log(f'INFO  testing {pkg}; output in {out}')
        for step in (lambda: check_tools_and_model_menu(c, pkg, dirs),):
            try:
                plot = step()
                check_zoom(c, plot)
                check_plddt_track(c, plot)
                check_help(c, pkg, plot)
            except Exception:
                c.check('completed without an exception', False, traceback.format_exc()[-800:])
        try:
            run(session, 'close')
        except Exception:
            pass
    total = sum(1 for line in c.results.read_text().splitlines() if line.startswith(('PASS', 'FAIL')))
    c.log(f'SUMMARY {total - c.failed} passed, {c.failed} failed; results in {c.results}')
    os._exit(1 if c.failed else 0)


main(session)
