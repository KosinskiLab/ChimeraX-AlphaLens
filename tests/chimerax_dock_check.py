"""Real-window check: plots docked into the main window keep a visible matrix.

ALPHALENS_DOCK_DIRS="<prediction>;<another prediction>"; ALPHALENS_TEST_OUTPUT is a new output
directory; ALPHALENS_SOURCE=installed or checkout (default). Opens both plots, docks the first,
then the second under it, floats both and docks both again at once. Before 1.5.1 the second
docked plot squeezed the first one's matrix to about 50 px, and the second plot docked again
was left loose over the graphics. Exit status 0 when the checks pass.
"""
import importlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
from Qt.QtCore import QTimer
from chimerax.core.commands import run, quote_path_if_necessary as quote

out = Path(os.environ['ALPHALENS_TEST_OUTPUT']); out.mkdir(parents=True, exist_ok=False)
if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
    pkg = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1]/'src'; pkg = 'chimerax.alphalens_dock_check'
    spec = importlib.util.spec_from_file_location(pkg, src/'__init__.py', submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec); sys.modules[pkg] = module; spec.loader.exec_module(module)
importlib.import_module(pkg+'.cmd').register_commands(session.logger)
MIN_PLOT_PX = 200       # tool.MIN_PLOT_PX (not in 1.5.0, which this check also has to fail on)


def pump(seconds=.1):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        session.ui.processEvents(); time.sleep(.005)


def measure(plots):
    mw = session.ui.main_window
    rows = []
    for p in plots:
        dw, v = p.tool_window._dock_widget, p._view
        g = dw.geometry()
        rows.append(dict(floating=dw.isFloating(), dock=[dw.width(), dw.height()], at=[g.x(), g.y()],
                         area=mw.dockWidgetArea(dw).value, visible=dw.isVisible(),
                         over_graphics=not dw.isFloating() and g.intersects(mw.centralWidget().geometry()),
                         viewport=[v.viewport().width(), v.viewport().height()],
                         matrix_px=round(v.transform().m11() * p.pae.matrix_size, 1)))
    return dict(main=[mw.width(), mw.height()], plots=rows)


def check_docked(state, both):
    plots = state['plots'] if both else state['plots'][:1]
    for k, p in enumerate(plots):
        assert not p['floating'], f'plot {k} did not dock'
        assert not p['over_graphics'], f'plot {k} docked loose over the graphics at {p["at"]}'
        assert p['viewport'][1] >= MIN_PLOT_PX, f'plot {k} matrix view squeezed to {p["viewport"]}'
        assert p['matrix_px'] >= 0.75 * MIN_PLOT_PX, f'plot {k} matrix drawn at {p["matrix_px"]} px'
    if both:
        a, b = (p['dock'][1] for p in plots)
        assert abs(a - b) <= 2, f'docked plots unequal: {a} and {b} px'


def main():
    report = {}
    try:
        dirs = [d for d in os.environ['ALPHALENS_DOCK_DIRS'].split(';') if d]
        assert len(dirs) == 2, 'ALPHALENS_DOCK_DIRS needs two predictions'
        plots = [run(session, f'alphalens open {quote(d)} wait true').structure._alphalens_plot for d in dirs]
        docks = [p.tool_window._dock_widget for p in plots]
        pump(1.0)
        report['floating'] = start = measure(plots)
        docks[0].setFloating(False); pump(1.0)
        report['first_docked'] = s = measure(plots); check_docked(s, both=False)
        docks[1].setFloating(False); pump(1.0)
        report['second_docked'] = s = measure(plots); check_docked(s, both=True)
        session.ui.main_window.grab().save(str(out/'sequential.png'))
        for d in docks:
            d.setFloating(True)
        pump(1.0)
        for d in docks:
            d.setFloating(False)
        pump(1.0)
        report['docked_together'] = s = measure(plots); check_docked(s, both=True)
        session.ui.main_window.grab().save(str(out/'together.png'))
        assert s['main'] == start['main'], 'docking resized the main window'
        report['passed'] = True
    except Exception:
        report['passed'] = False
        report['error'] = traceback.format_exc()
        traceback.print_exc(file=sys.__stderr__)
    (out/'results.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    print('DOCK CHECK PASSED' if report['passed'] else 'DOCK CHECK FAILED', file=sys.__stdout__, flush=True)
    os._exit(0 if report['passed'] else 1)


QTimer.singleShot(1000, main)
