'''
Save and restore a session with an AlphaLens plot, in a real window (Qt offscreen crashes ChimeraX).

    ALPHALENS_SESSION_MODEL=<AF3 model.cif with a distogram> ALPHALENS_SESSION_OUT=<empty dir> \\
        ChimeraX --cmd "open tests/chimerax_session_check.py"

Uses the installed bundle. Exit status 0 when all checks pass; the log is <out>/session_check.log.
'''
import os
import shutil
import time
import traceback

import numpy as np
from Qt.QtCore import QTimer
from chimerax.atomic import AtomicStructure
from chimerax.core.commands import run

MODEL, OUT = os.environ['ALPHALENS_SESSION_MODEL'], os.environ['ALPHALENS_SESSION_OUT']
LIMIT = 900
failures = []


def log(*a):
    with open(f'{OUT}/session_check.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a) + '\n')


def check(name, ok, detail=''):
    log('ok  ' if ok else 'FAIL', name, detail)
    if not ok:
        failures.append(name)


def done():
    log('FAILED ' + ', '.join(failures) if failures else 'ALL PASSED')
    os._exit(1 if failures else 0)


def later(ms, f):
    def g():
        try:
            f()
        except Exception:
            log('EXCEPTION', traceback.format_exc())
            failures.append('exception')
            done()
    QTimer.singleShot(ms, g)


def loaded(s):
    p = getattr(s, '_alphalens_plot', None)
    return p is not None and not p.closed() and p._load_progress is None


def when_loaded(next_step, t0=None):
    t0 = t0 or time.time()
    ss = session.models.list(type=AtomicStructure)
    if ss and loaded(ss[0]):
        later(1500, lambda: next_step(ss[0]))
    elif time.time() - t0 > LIMIT:
        check('loaded in time', False)
        done()
    else:
        later(500, lambda: when_loaded(next_step, t0))


def lines(s):
    from chimerax.alphalens.contacts import GROUP_NAME
    g = s.pseudobond_group(GROUP_NAME, create_type=None)
    return None if g is None else (len(g.pseudobonds), int(g.pseudobonds.displays.sum()))


def shown(s):
    return int(s.atoms.displays.sum()), int(s.residues.ribbon_displays.sum())


def fields(p):
    return dict(mode=p._mode, spread=p._spread_measure, thr=round(p._threshold, 3), region=p._struct_region,
                contacts=p._struct_contacts, picked=p._picked, plddt=p._plddt_slider.value(), mask=p._mask_matrix,
                hidden=np.flatnonzero(p._selected_hidden).tolist(), hider=p._hider.state(), files=(p.pae._pae_path, p._distogram_file))


saved = {}
d = f'{OUT}/prediction'


def link_prediction():
    '''A copy made of symbolic links, so that files can be taken away for the failure cases.'''
    seed_dir = os.path.dirname(MODEL)
    fold = os.path.dirname(seed_dir)
    seed = os.path.basename(seed_dir).split('_sample')[0]
    os.makedirs(f'{d}/{os.path.basename(seed_dir)}')
    for f in os.listdir(seed_dir):
        os.symlink(f'{seed_dir}/{f}', f'{d}/{os.path.basename(seed_dir)}/{f}')
    os.symlink(f'{fold}/{seed}_distogram', f'{d}/{seed}_distogram')
    return f'{d}/{os.path.basename(seed_dir)}/model.cif'


def start():
    run(session, f'alphalens open {link_prediction()} mode contact contacts true minProb 0.5', log=False)
    when_loaded(configure)


def configure(s):
    p = s._alphalens_plot
    check('spread available before saving', 'spread' in p.available_modes())
    p.set_threshold(0.6)
    n = p.pae.matrix_size
    p._struct_region = (0, n // 2, n // 2, n - 1)
    p.show_structure_contacts(True)
    p._mask_box.setChecked(True)
    p._plddt_slider.setValue(30)
    p._apply_hidden()
    p._selected_hidden[:5] = True
    p._apply_hidden()
    p.set_spread_measure('entropy') if hasattr(p, 'set_spread_measure') else None
    p.set_mode('spread')
    from chimerax.alphalens.contacts import GROUP_NAME
    bonds = s.pseudobond_group(GROUP_NAME, create_type=None).pseudobonds
    bonds[:5].displays = False          # Hidden by the user: must stay hidden
    later(1000, lambda: save(s))


def save(s):
    p = s._alphalens_plot
    saved.update(fields=fields(p), lines=lines(s), shown=shown(s))
    path = f'{OUT}/saved.cxs'
    run(session, f'save {path}', log=False)
    size = os.path.getsize(path)
    disto = sum(os.path.getsize(os.path.join(os.path.realpath(f'{d}/{x}'), y)) for x in os.listdir(d) if x.endswith('_distogram') for y in os.listdir(os.path.realpath(f'{d}/{x}')))
    check('session is small', size < min(40e6, disto / 4), f'session {size / 1e6:.1f} MB, distogram {disto / 1e6:.0f} MB')
    saved['path'] = path
    run(session, 'close session', log=False)
    later(1500, reopen)


def reopen():
    run(session, f'open {saved["path"]}', log=False)
    when_loaded(verify)


def verify(s):
    p = s._alphalens_plot
    now = fields(p)
    for k, v in saved['fields'].items():
        check(f'restored {k}', now[k] == v, f'{now[k]!r:.90} vs {v!r:.90}')
    check('spread still available', 'spread' in p.available_modes())
    check('has distogram', p.pae.has_distogram)
    check('contact lines kept, hidden ones stay hidden', lines(s) == saved['lines'], f'{lines(s)} vs {saved["lines"]}')
    check('atoms and ribbons shown unchanged', shown(s) == saved['shown'], f'{shown(s)} vs {saved["shown"]}')
    p._selected_hidden[:] = False
    p._plddt_slider.setValue(0)
    p._apply_hidden()
    check('Show hidden brings back what was hidden before saving', shown(s)[1] > saved['shown'][1], f'{shown(s)} vs {saved["shown"]}')
    run(session, 'close session', log=False)
    for x in os.listdir(d):
        if x.endswith('_distogram'):
            os.remove(f'{d}/{x}')
    later(1000, no_distogram)


def no_distogram():
    run(session, f'open {saved["path"]}', log=False)
    when_loaded(no_distogram_check)


def no_distogram_check(s):
    p = s._alphalens_plot
    check('distogram gone: plot opens, Spread off', not p.pae.has_distogram and 'spread' not in p.available_modes())
    check('distogram gone: contact view works', 'contact' in p.available_modes())
    run(session, 'close session', log=False)
    shutil.rmtree(d)
    later(1000, no_files)


def no_files():
    run(session, f'open {saved["path"]}', log=False)
    later(4000, no_files_check)


def no_files_check():
    ss = session.models.list(type=AtomicStructure)
    check('files gone: session still opens', len(ss) == 1)
    check('files gone: no plot', ss and getattr(ss[0], '_alphalens_plot', None) is None)
    check('files gone: contact lines still there', ss and lines(ss[0]) == saved['lines'])
    done()


log('start', MODEL)
later(1500, start)
