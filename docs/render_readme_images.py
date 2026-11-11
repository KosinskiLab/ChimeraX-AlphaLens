# Render the README images of AlphaLens in a real ChimeraX window, with the installed bundle.
# Offscreen Qt is not enough: the plot is grabbed from its window.
#
#   OUT=<dir> DATA=<dir> [PARTS=contacts,views,agreement] ChimeraX --script render_readme_images.py
#
# DATA holds two public E. coli AlphaFold2 predictions whose result pickles kept the distogram,
# P0A836+P0AGE9 (SucC-SucD) and P0ABD8+P24182 (AccB-AccC); in the Kosinski group they are under
# /g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/predictions/full_unfiltered_v3/
# ecoli/af2/pos_pairs/predictions/ (copy them with scripts/fetch_pred.sh). views.png is always
# rendered; PARTS can leave out contacts.png or agreement.png.
# contacts.png  SucC-SucD, contact-probability view, P >= 0.5, /A D219 - /B P101 clicked
# views.png     SucC-SucD, the five views and the contacts on the structure, 2 x 3 panels
# agreement.png AccB-AccC, agreement view, P >= 0.5, /A F148 - /B G52 hovered
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
OUT, DATA = os.environ['OUT'], os.environ['DATA']
PARTS = os.environ.get('PARTS', 'contacts,views,agreement').split(',')
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, 'render_log.txt'), 'w')
from chimerax.core.commands import run      # noqa: E402
from chimerax.atomic import AtomicStructure  # noqa: E402

# The README images are light, whatever the system appearance (Qt 6.8 and newer can switch it)
try:
    from Qt.QtCore import Qt as _Qt
    from Qt.QtGui import QGuiApplication
    QGuiApplication.styleHints().setColorScheme(_Qt.ColorScheme.Light)
except (AttributeError, ImportError):
    pass


def note(msg):
    log.write(msg + '\n')
    log.flush()


def flush(n=30):
    for _ in range(n):
        session.ui.processEvents()
        time.sleep(0.05)


def structure():
    return [m for m in session.models.list(type=AtomicStructure)][0]


def window(plot, w, h):
    plot.tool_window.ui_area.window().resize(w, h)
    flush()


def rows(plot):
    return {(r.chain_id, r.number): k for k, r in enumerate(plot.pae.row_residues_or_atoms())}


try:
    # ---- contacts.png
    run(session, f'alphalens open {DATA}/P0A836+P0AGE9 mode contact wait true')
    s = structure()
    plot = s._alphalens_plot
    plot.set_threshold(0.5)
    if 'contacts' in PARTS:
        window(plot, 760, 860)
        idx = rows(plot)
        plot._pick_contact(idx[('A', 219)], idx[('B', 101)])
        flush()
        note('contacts info: ' + plot._info.text())
        plot.save_image(os.path.join(OUT, 'contacts.png'))

    # ---- views.png
    plot.clear_picked_contact()
    run(session, f'alphalens #{s.id_string} mode agreement wait true')
    plot.set_threshold(0.5)
    window(plot, 720, 780)
    plot.set_mode('model')          # Let the window settle before grabbing
    flush(60)
    from Qt.QtGui import QImage, QPainter, QColor, QFont
    from Qt.QtCore import Qt, QRectF
    panels = []
    for mode, title in (('pae', 'PAE'), ('contact', 'Contact probability (dots: P ≥ 0.5)'),
                        ('model', 'Model contacts (P ≥ 0.5)'), ('std', 'Spread σ'),
                        ('agreement', 'Agreement (models with P ≥ 0.5)')):
        plot.set_mode(mode)
        flush()
        flush()
        panels.append((title, plot._view.grab(), plot._cbar.grab()))
        note(f'grabbed {mode}: view {plot._view.width()}x{plot._view.height()}')
    sizes = {(v.width(), v.height()) for _, v, _ in panels}
    assert len(sizes) == 1, f'view sizes differ: {sizes}'
    bg = panels[0][1].toImage().pixelColor(3, 3)       # Widget background, for the composite
    # Contacts on the structure, P >= 0.5, as drawn by Show Contacts
    plot._contacts_button_clicked(True)
    run(session, f'color #{s.id_string}/A #c9c9c9 target c; color #{s.id_string}/B #e9d6ad target c')
    run(session, 'set bgColor white; lighting soft; graphics silhouettes true; view; '
                 'turn y 90; view pad 0.12')
    dpr = panels[0][1].devicePixelRatio()
    box_w = 0.84 * panels[0][1].width() / dpr           # Logical size of the 3D box below
    box_h = 0.9 * panels[0][1].height() / dpr
    run(session, f'windowsize {int(box_w)} {int(box_h)}')
    flush()
    run(session, 'view pad 0.12')
    img3d = os.path.join(OUT, '_structure.png')
    run(session, f'save {img3d} width {int(box_w * dpr)} height {int(box_h * dpr)} '
                 f'supersample 3 transparentBackground true')
    from Qt.QtGui import QPixmap
    panels.append(('Contacts on the structure (P ≥ 0.5)', QPixmap(img3d), None))

    pw = max(p[1].width() / p[1].devicePixelRatio() for p in panels[:5])
    vh_l = max(p[1].height() / p[1].devicePixelRatio() for p in panels[:5])
    cb_h = max(p[2].height() / p[2].devicePixelRatio() for p in panels[:5])
    title_h, gap = 30, 16
    ph = title_h + vh_l + cb_h + gap
    cols, rws = 2, 3
    W, H = int(cols * pw + gap), int(rws * ph)
    img = QImage(int(W * dpr), int(H * dpr), QImage.Format.Format_RGB32)
    img.setDevicePixelRatio(dpr)
    img.fill(bg)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    font = QFont(p.font())
    font.setPointSizeF(13)
    font.setBold(True)
    p.setFont(font)
    from Qt.QtWidgets import QApplication
    p.setPen(QApplication.palette().windowText().color())     # Readable on the window's background
    for k, (title, view, cbar) in enumerate(panels):
        x0, y0 = (k % cols) * (pw + gap), (k // cols) * ph
        p.drawText(QRectF(x0, y0, pw, title_h), Qt.AlignmentFlag.AlignCenter, title)
        if cbar is None:            # 3D image: same box as a matrix panel, no colour bar
            bw, bh = view.width() / dpr, view.height() / dpr       # Same aspect as saved
            box = QRectF(x0 + (pw - bw) / 2, y0 + title_h + (vh_l - bh) / 2, bw, bh)
            p.drawPixmap(box, view, QRectF(view.rect()))
            continue
        vwl, vhl = view.width() / view.devicePixelRatio(), view.height() / view.devicePixelRatio()
        p.drawPixmap(QRectF(x0 + (pw - vwl) / 2, y0 + title_h, vwl, vhl), view, QRectF(view.rect()))
        cwl, chl = cbar.width() / cbar.devicePixelRatio(), cbar.height() / cbar.devicePixelRatio()
        p.drawPixmap(QRectF(x0 + (pw - cwl) / 2, y0 + title_h + vh_l, cwl, chl), cbar, QRectF(cbar.rect()))
    p.end()
    img = img.scaledToWidth(img.width() // 2, Qt.TransformationMode.SmoothTransformation)
    img.save(os.path.join(OUT, 'views.png'))
    note(f'views.png {img.width()}x{img.height()}')
    run(session, 'close')
    if 'agreement' not in PARTS:
        raise SystemExit

    # ---- agreement.png
    run(session, f'alphalens open {DATA}/P0ABD8+P24182 mode agreement wait true')
    s = structure()
    plot = s._alphalens_plot
    plot.set_threshold(0.5)
    window(plot, 760, 860)
    plot.set_mode('agreement')
    idx = rows(plot)
    plot._report(idx[('B', 52)], idx[('A', 148)])
    flush()
    note('agreement info: ' + plot._info.text())
    plot.save_image(os.path.join(OUT, 'agreement.png'))
    note('DONE')
except SystemExit:
    note('DONE (parts: ' + ','.join(PARTS) + ')')
except Exception:
    note('FAIL\n' + traceback.format_exc())
finally:
    log.close()
    run(session, 'exit')
