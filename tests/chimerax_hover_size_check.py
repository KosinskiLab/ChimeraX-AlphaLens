"""Real-window regression for mouse hover changing plot/window geometry.

ALPHALENS_SOURCE=installed or checkout (default); ALPHALENS_HOVER_INPUT is an AF2/AF3
prediction folder with contact probabilities; ALPHALENS_TEST_OUTPUT is a new output
directory. ALPHALENS_HOVER_PROBE=1 records an unfixed build without asserting stability.
Sends Qt mouse events to this disposable test window only; does not move the user's cursor.
"""
import importlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
from Qt.QtCore import QEvent, QPointF, QTimer, Qt
from Qt.QtGui import QMouseEvent
from Qt.QtWidgets import QApplication
from chimerax.core.commands import run, quote_path_if_necessary as quote

out = Path(os.environ['ALPHALENS_TEST_OUTPUT']); out.mkdir(parents=True, exist_ok=False)
if os.environ.get('ALPHALENS_SOURCE', 'checkout') == 'installed':
    pkg = 'chimerax.alphalens'
else:
    src = Path(__file__).resolve().parents[1]/'src'; pkg = 'chimerax.alphalens_hover_check'
    spec = importlib.util.spec_from_file_location(pkg, src/'__init__.py', submodule_search_locations=[str(src)])
    module = importlib.util.module_from_spec(spec); sys.modules[pkg] = module; spec.loader.exec_module(module)
importlib.import_module(pkg+'.cmd').register_commands(session.logger)


def pump(seconds=.08):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        session.ui.processEvents(); time.sleep(.005)


def measure(plot):
    top=plot.tool_window.ui_area.window(); view=plot._view
    return dict(window=[top.width(),top.height()],viewport=[view.viewport().width(),view.viewport().height()],
                scale=view.transform().m11(),info_hint=plot._info.minimumSizeHint().width())


def main():
    try:
        pae=run(session, f'alphalens open {quote(os.environ["ALPHALENS_HOVER_INPUT"])} wait true')
        plot=pae.structure._alphalens_plot; view=plot._view
        pump(.5)
        initial=measure(plot); observations=[]
        plot.tool_window.ui_area.grab().save(str(out/'before.png'))
        for i in range(100):
            # Vary residue numbers and statistics; also cross into/out of the pLDDT track.
            x=(i*17)%pae.matrix_size; y=(i*37+pae.matrix_size//2)%pae.matrix_size
            if i%9==0: y=plot._track_offset+1
            position=view.mapFromScene(QPointF(x+.5,y+.5))
            event=QMouseEvent(QEvent.Type.MouseMove,QPointF(position),QPointF(view.viewport().mapToGlobal(position)),
                              Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(view.viewport(),event)
            pump(.025)
            observations.append(measure(plot))
        assert plot._info.text(), 'Mouse move did not update residue information'
        plot._report(pae.matrix_size-1,pae.matrix_size//2)
        pump(.05)
        plot.tool_window.ui_area.grab().save(str(out/'after.png'))
        stable=all(row['window']==initial['window'] and row['viewport']==initial['viewport']
                   and abs(row['scale']-initial['scale'])<1e-9 for row in observations)
        report=dict(stable=stable,initial=initial,final=observations[-1],observations=observations)
        if os.environ.get('ALPHALENS_HOVER_PROBE')!='1':
            assert stable, 'Hover resized the plot'
            assert plot._info.toolTip() == plot._info.text(), 'Full hover text was lost'
            report['additional_phases'] = {}
            for phase in ('zoomed', 'user_resized'):
                if phase == 'zoomed':
                    view.zoom_by(1.4)
                    assert view.zoomed
                else:
                    top=plot.tool_window.ui_area.window()
                    top.resize(top.width()+100,top.height()-60)
                pump(.2)
                before=measure(plot)
                if phase == 'user_resized':
                    assert before['window'] != initial['window'], 'Manual resizing stopped working'
                for i in range(40):
                    plot._report((i*19)%pae.matrix_size,(i*29)%pae.matrix_size)
                    pump(.01)
                    after=measure(plot)
                    assert all(before[k]==after[k] for k in ('window','viewport','scale')), phase
                report['additional_phases'][phase] = before
        (out/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='observations'}),file=sys.__stdout__,flush=True)
        print('HOVER GEOMETRY CHECK PASSED' if stable else 'HOVER GROWTH REPRODUCED',file=sys.__stdout__,flush=True)
        os._exit(0)
    except Exception:
        traceback.print_exc(file=sys.__stderr__); os._exit(1)
QTimer.singleShot(1000,main)
