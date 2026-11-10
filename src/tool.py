"""
AlphaLens plot panel: PAE matrix with a pLDDT track, plus contact-probability, model-distance,
distogram-spread and seed / model agreement views reachable with the arrow buttons / tabs
when the data exist.

All matrices use the AlphaFold-DB green ramp (dark green = confident: low PAE, high contact
probability, sharp distance distribution, contact in many seeds or models). Predicted
contacts drawn on top of the matrix (dots) and on the structure (lines) are blue.

Interaction: scroll wheel / pinch zooms at the cursor, Shift-drag or middle-drag pans,
"Fit" resets. When zoomed into a large matrix the
visible region is redrawn at full resolution.
"""

from html import escape

import numpy as np
from chimerax.core.tools import ToolInstance
from chimerax.alphafold.pae import PAEView, pae_pixmap

from .contacts import BAND_EDGES, DEFAULT_MIN_P, DISTANCE_FAR, PICK_RGB, contact_rgb
from .hide import RowHider, true_runs

# pLDDT colors used by AlphaFold / the "alphafold" palette.
PLDDT_BINS = ((90, (0, 83, 214)), (70, (101, 203, 243)), (50, (255, 219, 19)),
              (-1e9, (255, 125, 69)))

MAX_DISPLAY = 900          # Matrices larger than this are pooled for the overview image
MAX_DETAIL = 1200          # Max pixels per side when redrawing a zoomed region
MAX_PIXELS_PER_RESIDUE = 40
MAX_DOTS = 40000
DOT_LEVELS = 20            # Color steps of the contact dots
SLIDER_MIN = 0.05          # Lowest contact threshold offered by the slider
UNDEFINED_GRAY = (190, 190, 190)   # Spread view: pairs whose distance is mostly in the last bin
DISTANCE_NEAR_GREY = 0.42          # Model distances view: grey at the contact cutoff (white at far)


def green_colormap(lo, hi, dark_at_high=False):
    '''AlphaFold-DB style ramp: dark green -> white (paegreen colors).'''
    from chimerax.core.colors import BuiltinColormaps, Colormap
    g = BuiltinColormaps['paegreen']
    colors = [tuple(c) for c in g.colors]
    if dark_at_high:
        colors = colors[::-1]
    return Colormap(tuple(np.linspace(lo, hi, len(colors))), colors)


CONTACT_BAND_EDGES = (0.0, 0.05) + BAND_EDGES + (1.0,)


def distance_colormap(contact_cmap):
    '''
    Colormap of the Model distances view (values from contacts.distance_view_values): grey
    from -2 (white, far) to -1 (dark, just beyond the cutoff), then contact_cmap from 0 to 1.
    '''
    from chimerax.core.colors import Colormap
    g = DISTANCE_NEAR_GREY
    vals = [-2.0, -1.0] + [float(v) for v in contact_cmap.data_values]
    cols = [(1.0, 1.0, 1.0, 1.0), (g, g, g, 1.0)] + [tuple(float(x) for x in c)
                                                    for c in contact_cmap.colors]
    return Colormap(vals, cols)


def banded_contact_colormap():
    '''Step colormap: < 0.05 white, then 0.05-0.2, 0.2-0.5, 0.5-0.8, >= 0.8.'''
    from chimerax.core.colors import Colormap
    g = green_colormap(0, 1, dark_at_high=True)
    e = CONTACT_BAND_EDGES
    levels = np.linspace(0, 1, len(e) - 1)
    vals, cols = [], []
    eps = 1e-6
    for b in range(len(e) - 1):
        c = tuple(g.interpolated_rgba8([levels[b]])[0] / 255.0)
        vals += [e[b] + (eps if b else 0), e[b + 1] - (eps if b < len(e) - 2 else 0)]
        cols += [c, c]
    return Colormap(vals, cols)


class Mode:
    def __init__(self, key, tab, title, unit, fmt, pool, dark_at_high):
        self.key, self.tab, self.title, self.unit, self.fmt = key, tab, title, unit, fmt
        self.pool = pool                    # How to downsample large matrices
        self.dark_at_high = dark_at_high


def make_modes(cutoff=8.0):
    c = f'{cutoff:g}'
    return {
        'pae': Mode('pae', 'PAE', 'Predicted aligned error', 'Å', '%.1f', 'mean', False),
        'contact': Mode('contact', 'Contact probability',
                        f'Predicted contact probability P(d < {c} Å)', '', '%.2f', 'max', True),
        'model': Mode('model', 'Model distances',
                      f'Model distances (CB–CB); pairs closer than {c} Å colored by predicted P',
                      'Å', '%.1f', 'max', True),
        'std': Mode('std', 'Spread σ', 'Distogram spread (standard deviation of distance)',
                    'Å', '%.2f', 'mean', False),
        'entropy': Mode('entropy', 'Spread H', 'Distogram spread (entropy of distance bins)',
                        'bits', '%.2f', 'mean', False),
        'agreement': Mode('agreement', 'Agreement', 'Agreement between seeds or models',
                          '', '%.0f', 'max', True),
    }


def pool2d(m, f, how):
    '''Downsample a 2D array by an integer factor (edge padded).'''
    if f <= 1:
        return m
    h, w = m.shape
    kh, kw = -(-h // f), -(-w // f)
    if kh * f != h or kw * f != w:
        m = np.pad(m, ((0, kh * f - h), (0, kw * f - w)), mode='edge')
    b = m.reshape(kh, f, kw, f)
    if how == 'max':
        return b.max(axis=(1, 3))
    if how == 'min':
        return b.min(axis=(1, 3))
    return b.mean(axis=(1, 3))


def pool_vector(v, f):
    if f <= 1:
        return v
    n = len(v)
    k = -(-n // f)
    v = np.pad(v, (0, k * f - n), mode='edge')
    return np.nanmean(v.reshape(k, f), axis=1)


def plddt_track_rgb(p, h):
    k = len(p)
    p = np.nan_to_num(p, nan=0.0)
    track = np.full((h, k, 3), 248, np.uint8)
    colors = np.empty((k, 3), np.uint8)
    for thresh, rgb in reversed(PLDDT_BINS):
        colors[p > thresh] = rgb
    heights = np.clip(np.round(p / 100 * h), 0, h).astype(int)
    mask = np.arange(h)[:, None] >= (h - heights)[None, :]
    track[mask] = np.broadcast_to(colors[None, :, :], (h, k, 3))[mask]
    return track


# -----------------------------------------------------------------------------
#
class AlphaLensView(PAEView):
    '''
    Graphics view in residue-index scene coordinates: overview image (+ zoom detail
    images added by the plot), with chain labels, dividers, pLDDT guides and contact
    dots drawn on top. Supports zoom and pan.
    '''

    def __init__(self, parent, select_cb, clear_cb, report_cb, key_cb, view_changed_cb):
        PAEView.__init__(self, parent, select_cb, clear_cb, report_cb)
        self._key_callback = key_cb
        self._view_changed_cb = view_changed_cb
        self.chains = []            # [(chain_id, start_row, end_row_exclusive)]
        self.n = 0                  # Matrix size (rows)
        self.track = None           # (y0, y1) of pLDDT track in scene coords
        self.show_dividers = True
        self.dots = []              # [(QColor, QPolygonF in scene coords)]
        self.dots_visible = False
        self.dot_min_px = 5.0
        self.dot_xy = None          # (N, 2) scene coords of dots, for clicking
        self.dot_pairs = None       # (N, 2) matrix rows (i, j) of each dot
        self.picked_xy = []         # Scene points of the picked contact (ring drawn)
        self.dot_click_cb = None
        self.track_hidden = []      # [(start, end)] rows hidden in the 3D model, faded on the track
        self.plddt_line = 0         # Value of the pLDDT slider, drawn across the track
        self._over_dot = False
        self.zoomed = False
        self._pan_last = None
        from Qt.QtCore import Qt, QTimer
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        from Qt.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setBackgroundBrush(self.palette().window())
        from Qt.QtWidgets import QGraphicsView
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._timer = t = QTimer(self)
        t.setSingleShot(True)
        t.setInterval(60)
        t.timeout.connect(lambda: self._view_changed_cb and self._view_changed_cb())

    def sizeHint(self):
        from Qt.QtCore import QSize
        return QSize(560, 620)

    # -- Image ------------------------------------------------------------------
    def set_image(self, rgb, scale, n, track, keep_zoom=False):
        scene = self.scene()
        if self._pixmap_item is not None:
            scene.removeItem(self._pixmap_item)
        pixmap = pae_pixmap(np.ascontiguousarray(rgb))
        item = scene.addPixmap(pixmap)
        item.setScale(scale)
        from Qt.QtCore import Qt
        item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._pixmap_item = item
        self.n = n
        self.track = track
        h = track[1] if track else n
        m = 0.07 * n                        # Margin for chain labels
        w = n + (2 if track else 1.3) * m   # With room on the right for the pLDDT slider
        if keep_zoom and self.zoomed:
            center = self.mapToScene(self.viewport().rect().center())
            scene.setSceneRect(-m, -m, w, h + 1.3 * m)
            self.centerOn(center)
        else:
            scene.setSceneRect(-m, -m, w, h + 1.3 * m)
            self.reset_zoom()
        self._timer.start()
        self.viewport().update()

    # -- Zoom / pan ---------------------------------------------------------------
    def _fit_scale(self):
        r = self.sceneRect()
        vp = self.viewport().rect()
        if r.width() <= 0 or r.height() <= 0:
            return 1.0
        return min(vp.width() / r.width(), vp.height() / r.height())

    def reset_zoom(self):
        self.zoomed = False
        from Qt.QtCore import Qt
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._timer.start()

    def zoom_by(self, factor):
        cur = self.transform().m11()
        fit = self._fit_scale()
        target = min(max(cur * factor, fit), MAX_PIXELS_PER_RESIDUE)
        if target <= fit * 1.001:
            if self.zoomed:
                self.reset_zoom()
            return
        self.scale(target / cur, target / cur)
        self.zoomed = True
        self._timer.start()

    def wheelEvent(self, event):
        # Any scroll (mouse wheel, smooth-scrolling mouse, trackpad) zooms at the cursor.
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.zoom_by(1.0015 ** max(-480, min(480, delta)))
        event.accept()

    def viewportEvent(self, event):
        from Qt.QtCore import QEvent, Qt
        if event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(1.0 + event.value())
                return True
        return PAEView.viewportEvent(self, event)

    def resizeEvent(self, event):
        from Qt.QtWidgets import QGraphicsView
        QGraphicsView.resizeEvent(self, event)
        if not self.zoomed:
            self.reset_zoom()
        self._timer.start()

    def scrollContentsBy(self, dx, dy):
        PAEView.scrollContentsBy(self, dx, dy)
        self._timer.start()

    def mousePressEvent(self, event):
        from Qt.QtCore import Qt
        if event.button() == Qt.MouseButton.RightButton:
            return                  # The context menu; keep the selection and the drag box
        pan = (event.button() == Qt.MouseButton.MiddleButton or
               (event.button() == Qt.MouseButton.LeftButton and
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        if pan:
            self._pan_last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        PAEView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            d = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if not self._mouse_down:
            from Qt.QtCore import Qt
            over = self.dot_at(*self._scene_position(event)) is not None
            if over != self._over_dot:
                self._over_dot = over
                if over:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.unsetCursor()
        PAEView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self._pan_last is not None:
            self._pan_last = None
            self.unsetCursor()
            return
        if self._mouse_down and self._down_xy is not None and self.dot_click_cb:
            # A click (no drag) on a contact dot picks that contact
            x, y = self._scene_position(event)
            x0, y0 = self._down_xy
            moved_px = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 * self.transform().m11()
            hit = self.dot_at(x, y) if moved_px < 4 else None
            if hit is not None:
                self._mouse_down = False
                self._clear_drag_box()
                self.dot_click_cb(*hit)
                return
        PAEView.mouseReleaseEvent(self, event)

    def _dot_width_px(self):
        return max(self.dot_min_px, 0.9 * self.transform().m11())

    def dot_at(self, x, y):
        '''(i, j) of the contact dot under scene point x, y, or None.'''
        if not self.dots_visible or self.dot_xy is None or len(self.dot_xy) == 0:
            return None
        s = self.transform().m11()
        d2 = ((self.dot_xy[:, 0] - x) ** 2 + (self.dot_xy[:, 1] - y) ** 2) * s * s
        k = int(np.argmin(d2))
        r = self._dot_width_px() / 2 + 3
        if d2[k] > r * r:
            return None
        i, j = self.dot_pairs[k]
        return int(i), int(j)

    def keyPressEvent(self, event):
        from Qt.QtCore import Qt
        k = event.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Right) and self._key_callback:
            self._key_callback(-1 if k == Qt.Key.Key_Left else 1)
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.5)
        elif k == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.5)
        elif k in (Qt.Key.Key_0, Qt.Key.Key_Escape):
            self.reset_zoom()
        else:
            PAEView.keyPressEvent(self, event)

    def _draw_drag_box(self, event):
        PAEView._draw_drag_box(self, event)
        from Qt.QtGui import QPen, QColor
        pen = QPen(QColor(0, 0, 0))
        pen.setCosmetic(True)       # 1 px at any zoom
        pen.setWidthF(1.5)
        self._drag_box.setPen(pen)
        self._drag_box.setZValue(3)

    # -- Overlay ------------------------------------------------------------------
    def drawForeground(self, painter, rect):
        if self.n == 0:
            return
        from Qt.QtCore import QPointF
        painter.save()
        try:
            if self.dots_visible and self.dots:
                self._draw_dots(painter)
            if self.picked_xy:
                self._draw_picked(painter)
            painter.resetTransform()
            vp = lambda x, y: QPointF(self.mapFromScene(QPointF(x, y)))
            self._draw_overlay(painter, vp, self.n, self.track[1] if self.track else self.n)
        finally:
            painter.restore()

    def _draw_picked(self, painter):
        from Qt.QtGui import QPen, QColor, QPainter
        from Qt.QtCore import Qt, QPointF
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r_px = self._dot_width_px() / 2 + 5
        s = self.transform().m11()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for color, width in ((QColor(0, 0, 0), 4.5), (QColor(*PICK_RGB), 2.5)):
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidthF(width)
            painter.setPen(pen)
            for x, y in self.picked_xy:
                painter.drawEllipse(QPointF(x, y), r_px / s, r_px / s)

    def _draw_dots(self, painter):
        '''Contact dots in scene coordinates with a cosmetic (pixel-sized) round pen.'''
        from Qt.QtGui import QPen, QColor, QPainter
        from Qt.QtCore import Qt
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = max(self.dot_min_px, 0.9 * self.transform().m11())
        rim = QPen(QColor(25, 25, 25))
        rim.setCosmetic(True)
        rim.setCapStyle(Qt.PenCapStyle.RoundCap)
        rim.setWidthF(w + 2)
        painter.setPen(rim)
        for _, poly in self.dots:
            painter.drawPoints(poly)
        for color, poly in self.dots:
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setWidthF(w)
            painter.setPen(pen)
            painter.drawPoints(poly)

    def _draw_overlay(self, painter, vp, n, bottom):
        from Qt.QtGui import QPen, QColor, QFont
        from Qt.QtCore import QPointF, QRectF, Qt
        frame = QPen(QColor(90, 90, 90))
        frame.setWidthF(1.0)
        painter.setPen(frame)
        painter.drawRect(QRectF(vp(0, 0), vp(n, n)))
        if self.track:
            painter.drawRect(QRectF(vp(0, self.track[0]), vp(n, self.track[1])))
            guide = QPen(QColor(150, 150, 150))
            guide.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(guide)
            for frac in (0.7, 0.9):     # pLDDT 70 / 90 guide lines
                y = self.track[1] - frac * (self.track[1] - self.track[0])
                painter.drawLine(vp(0, y), vp(n, y))
            for start, end in self.track_hidden:
                painter.fillRect(QRectF(vp(start, self.track[0]), vp(end, self.track[1])),
                                 QColor(255, 255, 255, 190))
            if self.plddt_line > 0:
                y = self.track[1] - self.plddt_line / 100 * (self.track[1] - self.track[0])
                line = QPen(QColor(20, 20, 20))
                line.setWidthF(1.5)
                painter.setPen(line)
                painter.drawLine(vp(0, y), vp(n, y))
        if self.show_dividers and len(self.chains) > 1:
            pen = QPen(QColor(40, 40, 40))
            pen.setWidthF(1.2)
            painter.setPen(pen)
            for _, start, _ in self.chains[1:]:
                painter.drawLine(vp(start, 0), vp(start, bottom))
                painter.drawLine(vp(0, start), vp(n, start))
        if not self.chains:
            return
        # Chain labels: above / left of the matrix, kept inside the viewport when zoomed.
        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        vr = QRectF(self.viewport().rect())
        top_y = max(vp(0, 0).y() - 4, fm.ascent() + 2)
        left_edge = max(vp(0, 0).x(), 0)
        for cid, start, end in self.chains:
            a, b = vp(start, start), vp(end, end)
            w = fm.horizontalAdvance(cid)
            # Visible part of this chain's span
            x0, x1 = max(a.x(), vr.left()), min(b.x(), vr.right())
            y0, y1 = max(a.y(), vr.top()), min(b.y(), vr.bottom())
            for (lo, hi, horizontal) in ((x0, x1, True), (y0, y1, False)):
                if hi - lo < w + 4:
                    continue
                c = (lo + hi) / 2
                if horizontal:
                    pos = QPointF(c - w / 2, top_y)
                else:
                    lx = left_edge - w - 4 if vp(0, 0).x() > w + 6 else 3
                    pos = QPointF(lx, c + fm.ascent() / 2 - 1)
                if self.zoomed:      # Readable on top of the matrix
                    painter.fillRect(QRectF(pos.x() - 2, pos.y() - fm.ascent() - 1, w + 4,
                                            fm.height() + 1), QColor(255, 255, 255, 200))
                painter.setPen(QColor(40, 40, 40))
                painter.drawText(pos, cid)
        if self.track and vp(0, 0).x() > 30:
            ty = (vp(0, self.track[0]).y() + vp(0, self.track[1]).y()) / 2
            small = QFont(font)
            small.setBold(False)
            small.setPointSizeF(max(7.0, font.pointSizeF() - 2))
            painter.setFont(small)
            lbl = 'pLDDT'
            painter.drawText(QPointF(vp(0, 0).x() - painter.fontMetrics().horizontalAdvance(lbl) - 3,
                                     ty + painter.fontMetrics().ascent() / 2 - 1), lbl)


# -----------------------------------------------------------------------------
#
class _PlotBeingRestored:
    '''What a session gets for a saved plot: the plot itself is built when the restore has ended.'''


# -----------------------------------------------------------------------------
#
class AlphaLensPlot(ToolInstance):

    name = 'AlphaLens'
    SESSION_ENDURING = False
    SESSION_SAVE = True                 # Settings and file paths only, see take_snapshot()
    SNAPSHOT_VERSION = 1

    def __init__(self, session, pae, cutoff=8.0, pae_colormap=None, divider_lines=True):
        self._pae = pae                     # AlphaLensPAE instance
        self.cutoff = cutoff
        self._modes = make_modes(cutoff)
        self._mode_cycle = ['pae', 'contact', 'model', 'spread', 'agreement']
        self._mode = 'pae'
        self._spread_measure = 'std'
        self._contact_scale = 'sqrt'        # 'sqrt', 'linear' or 'banded'
        self._far_threshold = 0.5           # Grey out spread where P(d > last bin edge) exceeds this
        self._threshold = DEFAULT_MIN_P     # Contacts (dots, 3D links, agreement) for P >= this
        self._agreement_runs = []           # Seeds / models of the prediction (data.agreement_runs)
        self._agreement_status = None       # Message while their contact probabilities load
        self._agreement_error = None        # Why the agreement view could not be made
        self._dots_on = True
        self._dots_all_views = True         # Contact dots on every matrix, not only the contact view
        self._load_progress = None          # data.LoadProgress while a distogram is being read
        self._distogram_file = None         # Distogram of the shown model, saved in sessions
        self._agreement_progress = None     # The same while the seeds or models are being read
        self._mask_matrix = False           # Also blank matrix pixels below the threshold
        self._pair_cache = None             # (pairs, probs) with P >= SLIDER_MIN, strongest first
        self._struct_contacts = False       # Contacts drawn on the structure
        self._struct_region = None
        self._picked = None                 # (i, j) contact picked by clicking a dot
        self._zoom_to_pick = False
        self._pae_colormap = pae_colormap
        self._drag_colors_structure = True
        self._showing_chain_dividers = divider_lines
        self._show_plddt = True
        self._color_blocks = None
        self._block_rgb = None
        self._detail_items = []
        self._render = None                 # Current render state (values, colormap, ...)
        self._data_status = 'No distogram found for this prediction'
        self._track_offset = None
        self._plddt = pae.row_plddt()
        self._hider = RowHider.for_pae(pae)
        self._selected_hidden = np.zeros(len(self._plddt), bool)    # Rows hidden by "Hide selected"
        self._models = []

        ToolInstance.__init__(self, session, 'AlphaLens')

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins=(6, 4, 6, 4), spacing=4)

        from Qt.QtWidgets import (QLabel, QWidget, QHBoxLayout, QToolButton, QButtonGroup,
                                  QComboBox, QPushButton, QSizePolicy)
        from Qt.QtCore import Qt

        self._heading = QLabel(parent)
        self._heading.setWordWrap(True)
        layout.addWidget(self._heading)

        # Model switcher
        mw = QWidget(parent)
        ml = QHBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(QLabel('Model:', mw))
        self._model_combo = mc = QComboBox(mw)
        mc.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mc.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mc.activated.connect(self._model_chosen)
        ml.addWidget(mc, stretch=1)
        layout.addWidget(mw)

        # Mode switcher:  [<] [PAE][Contact probability][Model distances][Spread][Agreement] [>]  [Fit]
        sw = QWidget(parent)
        sl = QHBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        self._left = lb = QToolButton(sw)
        lb.setArrowType(Qt.ArrowType.LeftArrow)
        lb.clicked.connect(lambda: self.step_mode(-1))
        self._right = rb = QToolButton(sw)
        rb.setArrowType(Qt.ArrowType.RightArrow)
        rb.clicked.connect(lambda: self.step_mode(1))
        sl.addWidget(lb)
        sl.addStretch(1)
        self._tabs = {}
        group = QButtonGroup(sw)
        group.setExclusive(True)
        for key in self._mode_cycle:
            b = QToolButton(sw)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setMinimumWidth(64)
            b.clicked.connect(lambda checked, k=key: self.set_mode(k))
            group.addButton(b)
            sl.addWidget(b)
            self._tabs[key] = b
        self._tab_group = group
        sl.addStretch(1)
        sl.addWidget(rb)
        fit = QToolButton(sw)
        fit.setText('Fit')
        fit.setToolTip('Reset zoom (0 / Esc).  Zoom: scroll wheel, trackpad pinch or +/−.  '
                       'Pan: Shift-drag or middle-drag.')
        fit.clicked.connect(lambda: self._view.reset_zoom())
        sl.addWidget(fit)
        layout.addWidget(sw)

        # Shown only while a distogram is being read. The text is a separate label because
        # the macOS style does not draw text inside a progress bar.
        from Qt.QtWidgets import QProgressBar
        self._progress_row = pw = QWidget(parent)
        pl = QHBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        self._progress_label = QLabel(pw)
        pl.addWidget(self._progress_label)
        self._progress_bar = pb = QProgressBar(pw)
        pb.setRange(0, 1000)
        pb.setTextVisible(False)
        pl.addWidget(pb, stretch=1)
        pw.hide()
        layout.addWidget(pw)

        self._view = gv = AlphaLensView(parent, self._rectangle_select, lambda: None,
                                       self._report, self.step_mode, self._view_changed)
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setFrameShape(gv.Shape.NoFrame)
        gv.dot_click_cb = self._pick_contact
        layout.addWidget(gv, stretch=1)
        from Qt.QtWidgets import QGraphicsScene
        gs = QGraphicsScene(gv)
        gv.setScene(gs)

        self._cbar = QLabel(parent)
        self._cbar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._cbar)

        # Contact threshold slider
        from Qt.QtWidgets import QSlider, QCheckBox
        from Qt.QtCore import QTimer
        tw_ = QWidget(parent)
        tl = QHBoxLayout(tw_)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel('Predicted contacts  P ≥', tw_))
        self._slider = sl_ = QSlider(Qt.Orientation.Horizontal, tw_)
        sl_.setRange(int(SLIDER_MIN * 100), 100)
        sl_.setValue(int(round(self._threshold * 100)))
        sl_.setToolTip('Show predicted contacts (dots, lines on the structure, agreement, '
                       'CSV export) with contact probability at least this')
        sl_.valueChanged.connect(self._slider_moved)
        tl.addWidget(sl_, stretch=1)
        self._slider_label = QLabel(f'{self._threshold:.2f}', tw_)
        self._slider_label.setMinimumWidth(self._slider_label.fontMetrics().horizontalAdvance('0.00') + 4)
        tl.addWidget(self._slider_label)
        self._mask_box = mb = QCheckBox('hide in matrix too', tw_)
        mb.setToolTip('Also blank contact-matrix pixels below the threshold')
        mb.toggled.connect(self._set_mask_matrix)
        tl.addWidget(mb)
        self._threshold_row = tw_
        layout.addWidget(tw_)
        self._slider_timer = st = QTimer(parent)
        st.setSingleShot(True)
        st.setInterval(120)
        st.timeout.connect(self._apply_threshold)
        self._progress_timer = pt = QTimer(parent)
        pt.setInterval(250)
        pt.timeout.connect(self._show_progress)

        # On the right of the pLDDT track, with the same scale: hides the model below its value
        self._plddt_slider = ps = QSlider(Qt.Orientation.Vertical, gv.viewport())
        ps.setRange(0, 100)
        ps.setToolTip('Hide the parts of the model with pLDDT below this value')
        ps.valueChanged.connect(self._plddt_slider_moved)
        self._plddt_value = QLabel('0', gv.viewport())
        self._plddt_timer = pdt = QTimer(parent)
        pdt.setSingleShot(True)
        pdt.setInterval(100)
        pdt.timeout.connect(self._apply_plddt_threshold)

        self._info = QLabel(parent)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._info.setMinimumHeight(self._info.fontMetrics().height() + 2)
        layout.addWidget(self._info)

        bw = QWidget(parent)
        bl = QHBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        for text, cb in (('Color PAE Domains', self._color_domains),
                         ('Color pLDDT', self._color_plddt)):
            b = QPushButton(text, bw)
            b.clicked.connect(cb)
            bl.addWidget(b)
        self._contacts_button = cbn = QPushButton('Show Contacts', bw)
        cbn.setCheckable(True)
        cbn.clicked.connect(self._contacts_button_clicked)
        bl.addWidget(cbn)
        b = QPushButton('Save Image', bw)
        b.clicked.connect(self._save_image)
        bl.addWidget(b)
        self._export_button = eb = QPushButton('Export CSV', bw)
        eb.clicked.connect(self._export_csv)
        bl.addWidget(eb)
        bl.addStretch(1)
        layout.addWidget(bw)

        self._attach_structure()
        self._set_chains()
        self._populate_models()
        self._update_controls()
        self._update_dots()
        self._redraw()
        tw.manage(placement=None)

    # ---------------------------------------------------------------------------
    # Data / structure
    #
    @property
    def pae(self):
        return self._pae

    def _attach_structure(self):
        s = self._pae.structure
        self._structure_delete_handler = s.triggers.add_handler('deleted', self._structure_deleted)
        self._structure_changes_handler = s.triggers.add_handler('changes', self._structure_changed)

    def _detach_structure(self):
        h = getattr(self, '_structure_changes_handler', None)
        if h is not None:
            h.remove()
            self._structure_changes_handler = None
        h = getattr(self, '_structure_delete_handler', None)
        if h is not None:
            try:
                self._pae.structure.triggers.remove_handler(h)
            except Exception:
                pass
            self._structure_delete_handler = None

    def _structure_changed(self, trigger_name, data):
        structure, changes = data
        if self.closed() or structure is not self._pae.structure:
            return
        if ('coord changed' in changes.atom_reasons()
                or 'active_coordset changed' in changes.structure_reasons()
                or 'coordset changed' in changes.coordset_reasons()
                or changes.num_deleted_atoms() or len(changes.created_atoms())):
            # The cache checks current coordinates too, for commands before this trigger fires.
            self._pae._model_distances = self._pae._model_contacts = None
            if self._mode == 'model':
                self._block_rgb = None
                self._redraw(keep_zoom=True)

    def set_pae(self, pae):
        '''Show another model (same prediction) in this window.'''
        same_size = pae.matrix_size == self._pae.matrix_size
        self._end_progress()                # A read for the previous model is no longer shown
        self._distogram_file = None
        self._detach_structure()
        self._restore_hidden()
        from . import contacts
        contacts.clear_picked_contact(self._pae.structure)
        if same_size and pae.agreement is None:
            pae.agreement = self._pae.agreement     # Seeds / models are shared by all models
        self._pae = pae
        self._attach_structure()
        self._pair_cache = None
        if not same_size:
            self._picked = None
        self._plddt = pae.row_plddt()
        self._hider = RowHider.for_pae(pae)
        if not same_size:
            self._selected_hidden = np.zeros(len(self._plddt), bool)
        self._apply_hidden()
        self._color_blocks = None
        self._block_rgb = None
        self._data_status = ('contact probabilities loaded' if pae.has_distogram or pae.has_contact
                             else 'No distogram found for this prediction')
        if not same_size:
            self._struct_region = None
        self._set_chains()
        self._populate_models()
        self._update_controls()
        self._update_dots()
        self._redraw(keep_zoom=same_size)

    def set_data_status(self, status):
        self._data_status = status
        self._update_controls()

    def data_changed(self):
        self._pair_cache = None
        self._update_controls()
        self._update_dots()
        self._redraw(keep_zoom=True)

    def _populate_models(self):
        from . import data
        mc = self._model_combo
        mc.blockSignals(True)
        mc.clear()
        s = self._pae.structure
        path = getattr(s, 'filename', None)
        try:
            self._models = data.prediction_models(path) if path else []
        except Exception:
            self._models = []
        try:
            self._agreement_runs = data.agreement_runs(path) if path else []
        except Exception:
            self._agreement_runs = []
        import os
        cur = os.path.realpath(path) if path else None
        current = -1
        from Qt.QtCore import Qt
        for i, m in enumerate(self._models):
            label = m['label'] + ('' if m['path'] else '   (not available locally)')
            mc.addItem(label)
            if m['path'] is None:
                item = mc.model().item(i)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            elif current < 0 and cur and os.path.realpath(m['path']) == cur:
                current = i
        if current < 0:
            mc.addItem(str(s))
            current = mc.count() - 1
        mc.setCurrentIndex(current)
        n_avail = sum(1 for m in self._models if m['path'])
        mc.setEnabled(n_avail > 1)
        mc.setToolTip('Show another model of this prediction in this window'
                      if n_avail > 1 else 'No other models of this prediction found locally')
        mc.blockSignals(False)

    def _model_chosen(self, index):
        if index >= len(self._models):
            return
        path = self._models[index]['path']
        if not path:
            return
        import os
        cur = getattr(self._pae.structure, 'filename', None)
        if cur and os.path.realpath(cur) == os.path.realpath(path):
            return
        from .cmd import switch_model
        try:
            switch_model(self.session, self, path)
        except Exception as e:
            self.session.logger.error(f'AlphaLens: could not switch model: {e}')
            self._populate_models()

    # ---------------------------------------------------------------------------
    # Modes
    #
    @property
    def _view_mode(self):
        '''Requested view if its data are available, else PAE (choice is remembered).'''
        m = self._mode
        if m not in self.available_modes() or (m == 'agreement' and self._pae.agreement is None):
            return 'pae'
        return m

    @property
    def mode(self):
        m = self._view_mode
        return self._modes[self._spread_measure if m == 'spread' else m]

    @property
    def threshold(self):
        return self._threshold

    def available_modes(self):
        pae = self._pae
        modes = ['pae']
        if pae.has_contact:
            modes.append('contact')
        modes.append('model')               # Contacts colored by P when there is one
        if pae.has_distogram:
            modes.append('spread')
        if pae.agreement is not None or (pae.has_contact and len(self._agreement_runs) >= 2
                                         and self._agreement_error is None
                                         and self._load_progress is None):
            modes.append('agreement')       # Loaded when first shown
        return modes

    def step_mode(self, step):
        modes = self.available_modes()
        if len(modes) < 2:
            return
        i = modes.index(self._view_mode)
        m = modes[(i + step) % len(modes)]
        if m == 'agreement' and self._pae.agreement is None:
            if self._agreement_status is None:
                self._load_agreement()      # Shown when loaded
                return
            m = modes[(i + 2 * step) % len(modes)]      # Still loading: skip it
        self._mode = m
        self._redraw(keep_zoom=True)

    def set_mode(self, name):
        if name in ('std', 'entropy'):
            self._spread_measure = name
            name = 'spread'
        if name not in self.available_modes():
            from chimerax.core.errors import UserError
            need = {'spread': 'a distogram',
                    'agreement': 'contact probabilities for at least two seeds or models'
                    }.get(name, 'a distogram or AF3 contact_probs')
            why = self._agreement_error if name == 'agreement' and self._agreement_error \
                else self._data_status
            raise UserError(f'View "{name}" needs {need}; {self._pae.structure}: {why}')
        if name == 'agreement' and self._pae.agreement is None:
            self._load_agreement()
            for key, b in self._tabs.items():   # Keep the shown view's tab checked meanwhile
                b.setChecked(key == self._view_mode)
            return
        self._mode = name
        self._redraw(keep_zoom=True)

    # ---------------------------------------------------------------------------
    # Agreement between seeds / models
    #
    def _load_agreement(self):
        if self._agreement_status is not None:
            return                          # Already loading
        from .cmd import start_agreement
        try:
            start_agreement(self.session, self._pae, self, self.cutoff, show=True)
        except Exception as e:
            self.agreement_failed(str(e))

    def agreement_loading(self, n, kind, progress):
        self._agreement_status = f'Loading the contact probabilities of {n} {kind}s ...'
        self._agreement_progress = progress
        self._sync_progress()
        self._info.setText(self._agreement_status)
        self._update_controls()

    def agreement_loaded(self, show=False):
        self._agreement_status = None
        self._agreement_progress = None
        self._sync_progress()
        self._info.setText('')
        if show:
            self._mode = 'agreement'
        self._update_controls()
        self._redraw(keep_zoom=True)

    def agreement_failed(self, reason):
        self._agreement_status = None
        self._agreement_progress = None
        self._sync_progress()
        self._agreement_error = reason
        self._info.setText(f'No agreement view: {reason}')
        self._update_controls()

    def _agreement_tooltip(self, enabled):
        if self._agreement_status is not None:
            return self._agreement_status
        agr = self._pae.agreement
        if agr is not None:
            return (f'Number of {agr.kind}s (of {agr.n}) in which each residue pair has '
                    f'P ≥ the slider threshold')
        if enabled:
            runs = self._agreement_runs
            return (f'Number of {runs[0]["kind"]}s (of {len(runs)}) in which each residue pair '
                    f'has P ≥ the slider threshold; read when first shown')
        runs = self._agreement_runs
        if self._agreement_error:
            why = self._agreement_error
        elif self._load_progress is not None:
            why = 'the distogram is still loading'
        elif not self._pae.has_contact:
            if runs and runs[0]['kind'] == 'model':
                why = ('the result pickles have no distogram (AlphaPulldown removes it unless '
                       'run with --noremove_keys_from_pickles)')
            else:
                why = 'the shown model has no contact probabilities'
        else:
            from .data import agreement_shortfall
            why = agreement_shortfall(getattr(self._pae.structure, 'filename', None), runs)
        return ('Needs contact probabilities for at least two seeds (AlphaFold3) or two models '
                f'with distograms (AlphaFold2); {why}')

    def _log_agreement(self):
        from .cmd import log_agreement
        log_agreement(self.session, self._pae, self._threshold)

    def set_spread_measure(self, measure):
        self._spread_measure = measure
        self._redraw(keep_zoom=True)

    def set_far_threshold(self, threshold):
        self._far_threshold = threshold
        self._redraw(keep_zoom=True)

    def _far_mask(self):
        mt = self._pae.metrics
        if mt is None or 'far' not in mt or self._far_threshold >= 1:
            return None
        return mt['far'] > self._far_threshold

    def set_contact_scale(self, scale):
        self._contact_scale = scale
        self._redraw(keep_zoom=True)

    def set_dots_shown(self, show):
        self._dots_on = show
        self._update_dots()
        self._redraw(keep_zoom=True)

    def _slider_moved(self, value):
        self._slider_label.setText(f'{value / 100:.2f}')
        self._slider_timer.start()          # Debounce while dragging

    def _apply_threshold(self):
        self.set_threshold(self._slider.value() / 100, from_slider=True)

    def set_threshold(self, threshold, from_slider=False):
        '''Show contacts (dots, 3D links, optionally matrix) with P >= threshold.'''
        t = float(min(1.0, max(SLIDER_MIN, threshold)))
        self._threshold = t
        if not from_slider:
            self._slider.blockSignals(True)
            self._slider.setValue(int(round(t * 100)))
            self._slider.blockSignals(False)
            self._slider_label.setText(f'{t:.2f}')
        self._update_dots()
        self._redraw(keep_zoom=True)
        if self._struct_contacts:
            self.show_structure_contacts(True)
        self._update_controls()

    def _set_mask_matrix(self, on):
        self._mask_matrix = bool(on)
        self._redraw(keep_zoom=True)

    def set_pae_colormap(self, colormap):
        self._pae_colormap = colormap
        self._redraw(keep_zoom=True)

    # ---------------------------------------------------------------------------
    # Data loading status
    #
    def distogram_loading(self, path, progress):
        self._distogram_file = path
        self._data_status = f'Loading distogram {path} ...'
        self._load_progress = progress
        self._sync_progress()
        self._update_controls()
        if self._pae.has_contact:
            self._redraw(keep_zoom=True)    # The heading says where the contacts come from

    def distogram_loaded(self):
        self._data_status = 'ok'
        self._end_progress()
        self.data_changed()

    def distogram_failed(self, message):
        self._data_status = f'Distogram could not be read: {message}'
        self._end_progress()
        self._update_controls()

    def _sync_progress(self):
        '''Show the progress row while something is being read, else hide it.'''
        if self._load_progress is None and self._agreement_progress is None:
            self._progress_timer.stop()
            self._progress_row.hide()
        else:
            self._show_progress()
            self._progress_row.show()
            self._progress_timer.start()

    def _show_progress(self):
        p = self._load_progress or self._agreement_progress
        if p is None:
            return
        fraction = p.fraction()
        bar = self._progress_bar
        if fraction is None:
            bar.setRange(0, 0)              # Busy indicator until the file size is known
        else:
            bar.setRange(0, 1000)
            bar.setValue(int(1000 * fraction))
        self._progress_label.setText(p.message())

    def _end_progress(self):
        self._load_progress = None
        self._sync_progress()

    def _update_controls(self):
        avail = self.available_modes()
        on = len(avail) > 1
        tip = 'Switch view (or use ←/→ keys in the plot)' if on else self._data_status
        for b in (self._left, self._right):
            b.setEnabled(on)
            b.setToolTip(tip)
        reasons = {'contact': 'Needs a distogram or AlphaFold3 contact_probs',
                   'spread': 'Needs a distogram'}
        c = f'{self._contact_cutoff():g}'
        about = {
            'pae': 'Predicted aligned error',
            'contact': f'Contact probability predicted by AlphaFold, P(d < {c} Å), from the '
                       'distogram or AF3 contact_probs; dots mark the pairs above the slider',
            'model': f'Distances in the shown model (CB–CB, CA for glycine): grey from {c} Å '
                     f'(dark) to {DISTANCE_FAR:g} Å and beyond (white); pairs closer than {c} Å '
                     'are colored by the predicted contact probability'
                     + ('' if self._pae.has_contact else ' (none here: one color)'),
            'spread': 'How spread the predicted distance distribution of each pair is',
        }
        for key, b in self._tabs.items():
            enabled = key in avail
            if key == 'agreement':
                b.setToolTip(self._agreement_tooltip(enabled))
                enabled = enabled and self._agreement_status is None
            else:
                b.setToolTip(about.get(key, '') if enabled
                             else f'{reasons.get(key, "")} ({self._data_status})')
            b.setEnabled(enabled)
        eb = self._export_button
        eb.setEnabled(self._pae.has_contact and self._load_progress is None)
        if self._load_progress is not None:
            eb.setToolTip('Available when the distogram has loaded')
        elif self._pae.has_contact:
            agr = self._pae.agreement
            where = ' inside the dragged box' if self._struct_region is not None else ''
            per_run = f', with P in each of the {agr.n} {agr.kind}s' if agr is not None else ''
            eb.setToolTip(f'Save the contacts with P ≥ {self._threshold:.2f}{where} to a CSV file: '
                          f'residues, P, PAE, expected and model distances{per_run}')
        else:
            eb.setToolTip('Needs contact probabilities')
        cb = self._contacts_button
        cb.setEnabled(self._pae.has_contact)
        cb.setChecked(self._struct_contacts and self._pae.has_contact)
        cb.setText('Hide Contacts' if cb.isChecked() else 'Show Contacts')
        self._threshold_row.setEnabled(self._pae.has_contact)
        cb.setToolTip(f'Draw all predicted contacts with P ≥ {self._threshold:.2f} on the structure '
                      'as lines, darker blue and thicker for higher P; dragging a box in the plot '
                      'limits them to that region' if self._pae.has_contact
                      else 'Needs contact probabilities')

    # ---------------------------------------------------------------------------
    # Drawing
    #
    def _colormap_and_values(self, mode, m):
        '''Colormap, values to color and colorbar spec (lo, hi, gamma, ticks).'''
        if mode.key == 'pae':
            if self._pae_colormap is not None:
                cmap = self._pae_colormap
                v = cmap.data_values
                return cmap, m, (float(v[0]), float(v[-1]), None, None)
            return green_colormap(0, 30), m, (0, 30, None, None)
        if mode.key == 'contact':
            if self._mask_matrix:
                m = np.where(m >= self._threshold, m, 0)
            if self._contact_scale == 'banded':
                return banded_contact_colormap(), m, (0, 1, None, CONTACT_BAND_EDGES)
            cmap = green_colormap(0, 1, dark_at_high=True)
            if self._contact_scale == 'sqrt':
                return cmap, np.sqrt(np.clip(m, 0, 1)), (0, 1, 0.5, None)
            return cmap, m, (0, 1, None, None)
        if mode.key == 'model':
            return self._model_colormap_and_values(m)
        if mode.key == 'agreement':
            n = self._pae.agreement.n
            ticks = np.arange(n + 1) if n <= 10 else np.unique(np.round(np.linspace(0, n, 6)))
            return green_colormap(0, n, dark_at_high=True), m, (0, n, None, ticks)
        # Spread: range normalized to this prediction (99th percentile of defined pairs)
        far = self._far_mask()
        defined = m[~far] if far is not None else m
        hi = float(np.percentile(defined, 99)) if defined.size else 1.0
        hi = max(0.5, np.ceil(hi * 2) / 2)
        return green_colormap(0, hi), m, (0, hi, None, None)

    def _contact_cutoff(self):
        mt = self._pae.metrics
        return mt.get('cutoff', self.cutoff) if mt else self.cutoff

    def _model_colormap_and_values(self, P):
        '''
        Distances in the model in grey, with the pairs closer than the cutoff (the contacts in
        the model) colored like the contact-probability view. A contact with P near 0 still
        gets the lightest visible color, so that no model contact disappears.
        '''
        from .contacts import model_distances, distance_view_values
        dist = model_distances(self._pae)[0]
        c = self._contact_cutoff()
        if P is None:
            cmap = green_colormap(0, 1, dark_at_high=True)
            return distance_colormap(cmap), distance_view_values(dist, c), (0, 1, None, ())
        mode = self._modes['contact']
        saved, self._mask_matrix = self._mask_matrix, False
        try:
            cmap, vals, cbar = self._colormap_and_values(mode, P)
        finally:
            self._mask_matrix = saved
        floor = 0.06 if self._contact_scale == 'banded' else 1 / 6
        return distance_colormap(cmap), distance_view_values(dist, c, vals, floor), cbar

    def _render_region(self, y0, y1, x0, x1, f):
        '''RGB image of matrix rows y0:y1, columns x0:x1, pooled by f.'''
        r = self._render
        if self._block_rgb is not None:
            return np.ascontiguousarray(self._block_rgb[y0:y1:f, x0:x1:f])
        pooled = pool2d(r['vals'][y0:y1, x0:x1], f, r['pool'])
        rgb = r['cmap'].interpolated_rgba8(pooled.ravel())[:, :3].reshape(pooled.shape + (3,))
        if r['grey'] is not None:
            g = pool2d(r['grey'][y0:y1, x0:x1].astype(np.float32), f, 'mean') >= 0.5
            rgb[g] = UNDEFINED_GRAY
        return rgb

    def _matrix(self, key):
        if key == 'agreement':
            return self._pae.agreement.counts(self._threshold)
        if key == 'model':          # Colored by P; the contacts themselves come from the model
            return self._pae.matrix('contact')
        return self._pae.matrix(key)

    def _redraw(self, keep_zoom=False):
        vm = self._view_mode
        mode = self.mode
        m = self._matrix(mode.key)
        n = self._pae.matrix_size if m is None else m.shape[0]
        f = max(1, -(-n // MAX_DISPLAY))
        cmap, vals, cbar = self._colormap_and_values(mode, m)
        far = self._far_mask() if vm == 'spread' else None
        self._render = {'vals': vals, 'cmap': cmap, 'pool': mode.pool, 'grey': far, 'f': f, 'n': n}
        self._block_rgb = (self._block_colored_rgb(vals, mode, cbar)
                           if self._color_blocks is not None else None)

        rgb = self._render_region(0, n, 0, n, f)
        k = rgb.shape[0]
        track = None
        if self._show_plddt:
            gap = max(2, k // 120)
            th = max(14, k // 9)
            sep = np.full((gap, rgb.shape[1], 3), 255, np.uint8)
            trk = plddt_track_rgb(pool_vector(self._plddt, f), th)
            rgb = np.concatenate([rgb, sep, trk[:, :rgb.shape[1]]], axis=0)
            track = ((k + gap) * f, (k + gap + th) * f)
            self._render['track_px'] = th
        self._track_offset = track[0] if track else None
        self._clear_detail()
        v = self._view
        v.show_dividers = self._showing_chain_dividers
        v.dots_visible = bool(v.dots) and (vm == 'contact' or self._dots_all_views)
        v.set_image(rgb, f, n, track, keep_zoom=keep_zoom)

        nav = '  ←/→ view, wheel/pinch zoom.' if len(self.available_modes()) > 1 else \
              '  Wheel/pinch to zoom.'
        src = ''
        if mode.key in ('contact', 'model') and self._pae.has_contact \
                and self._pae.metrics.get('contact_only'):
            note = 'distogram loading' if self._load_progress is not None else 'no distogram'
            src = f' <span style="color:gray">(AF3 contact_probs; {note})</span>'
        if mode.key == 'model' and not self._pae.has_contact:
            src = ' <span style="color:gray">(no contact probabilities: contacts in one color)</span>'
        pool = (f' <span style="color:gray">[overview {f}×{f} {mode.pool}-pooled; '
                f'zoom in for full resolution]</span>') if f > 1 else ''
        if far is not None:
            edge = self._pae.metrics.get('far_edge', 21.7)
            src = (f' <span style="color:gray">grey: P(d &gt; {edge:.1f} Å) &gt; '
                   f'{self._far_threshold:g}, spread undefined</span>')
        title = mode.title
        if vm == 'agreement':
            agr = self._pae.agreement
            title = (f'Agreement: {agr.kind}s (of {agr.n}) with P(d < {agr.cutoff:g} Å) ≥ '
                     f'{self._threshold:.2f}')
        self._heading.setText(f'<html><b>{escape(title)}</b>{src}{pool}<br>'
                              f'{escape(str(self._pae.structure))} · {n} residues. '
                              f'Drag a box to color residues.{nav}</html>')
        for key, b in self._tabs.items():
            label = self._modes[key].tab if key != 'spread' else self._modes[self._spread_measure].tab
            b.setText(label)
            b.setChecked(key == vm)
        self._colorbar_args = (mode, cmap, cbar)
        self._draw_colorbar(mode, cmap, cbar)

    def _refresh_colorbar(self):
        if getattr(self, '_colorbar_args', None) is not None:
            self._draw_colorbar(*self._colorbar_args)

    # -- Hiding parts of the model -----------------------------------------------------
    def _view_changed(self):
        self._place_plddt_slider()
        self._update_detail()

    def _place_plddt_slider(self):
        '''Put the slider right of the track, so that its handle follows the track's scale.'''
        v, slider, value = self._view, self._plddt_slider, self._plddt_value
        if v.track is None:
            slider.hide()
            value.hide()
            return
        from Qt.QtCore import QPointF
        from Qt.QtWidgets import QStyle
        top_right = v.mapFromScene(QPointF(v.n, v.track[0]))
        bottom = v.mapFromScene(QPointF(v.n, v.track[1])).y()
        handle = slider.style().pixelMetric(QStyle.PixelMetric.PM_SliderLength, None, slider)
        width = slider.sizeHint().width()
        x = min(top_right.x() + 4, v.viewport().width() - width)
        y = top_right.y() - handle // 2
        slider.setGeometry(x, y, width, bottom - top_right.y() + handle)
        value.adjustSize()
        value.move(x + (width - value.width()) // 2, y - value.height())
        shown = bottom > 0 and top_right.y() < v.viewport().height()
        slider.setVisible(shown)
        value.setVisible(shown)

    def _plddt_slider_moved(self, value):
        self._view.plddt_line = value
        self._plddt_value.setText(str(value))
        self._place_plddt_slider()
        self._view.viewport().update()
        self._plddt_timer.start()           # Debounce while dragging

    def _apply_plddt_threshold(self):
        self._apply_hidden()
        t = self._plddt_slider.value()
        n = int((self._plddt < t).sum())
        self.session.logger.status(f'AlphaLens: {n} residues with pLDDT < {t} hidden' if t
                                   else 'AlphaLens: pLDDT slider off')

    def _apply_hidden(self):
        self._hider.apply(plddt=self._plddt < self._plddt_slider.value(),
                          selected=self._selected_hidden)
        self._view.track_hidden = true_runs(self._hider.hidden_rows)
        self._view.plddt_line = self._plddt_slider.value()
        self._view.viewport().update()

    def _restore_hidden(self):
        if not self._pae.structure.deleted:
            self._hider.restore()

    def _hide_selected(self):
        rows = self._selected_to_hide()
        if not rows.any():
            return
        self._selected_hidden = self._selected_hidden | rows
        self._apply_hidden()                # The selection stays, so that it can be hidden again
        self.session.logger.info(f'AlphaLens {self._pae.structure}: hid {int(rows.sum())} selected '
                                 f'residues; right-click the plot and use "Show hidden" to restore them')

    def _selected_to_hide(self):
        '''Rows selected in ChimeraX and not yet hidden by "Hide selected".'''
        return self._hider.selected_rows() & ~self._selected_hidden

    def _show_hidden(self):
        self._selected_hidden = np.zeros_like(self._selected_hidden)
        self._apply_hidden()

    # -- Zoom detail ----------------------------------------------------------------
    def _clear_detail(self):
        scene = self._view.scene()
        for it in self._detail_items:
            scene.removeItem(it)
        self._detail_items = []

    def _update_detail(self):
        '''Redraw the visible region at (near) full resolution when zoomed in.'''
        self._clear_detail()
        r = self._render
        v = self._view
        if r is None or not v.zoomed or r['f'] <= 1:
            return
        n = r['n']
        vis = v.mapToScene(v.viewport().rect()).boundingRect()
        x0, x1 = max(0, int(vis.left())), min(n, int(np.ceil(vis.right())) + 1)
        y0, y1 = max(0, int(vis.top())), min(n, int(np.ceil(vis.bottom())) + 1)
        if x1 <= x0:
            return
        fd = max(1, -(-max(x1 - x0, max(y1 - y0, 1)) // MAX_DETAIL))
        if fd >= r['f']:
            return
        # Align to the detail grid so neighbouring redraws match
        x0, y0 = (x0 // fd) * fd, (y0 // fd) * fd
        scene = v.scene()
        from Qt.QtCore import Qt
        if y1 > y0:
            rgb = self._render_region(y0, y1, x0, x1, fd)
            item = scene.addPixmap(pae_pixmap(np.ascontiguousarray(rgb)))
            item.setPos(x0, y0)
            item.setScale(fd)
            item.setZValue(0.5)
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._detail_items.append(item)
        if v.track and vis.bottom() > v.track[0]:
            th = r.get('track_px', 14)
            trk = plddt_track_rgb(pool_vector(self._plddt[x0:x1], fd), th)
            item = scene.addPixmap(pae_pixmap(np.ascontiguousarray(trk)))
            from Qt.QtGui import QTransform
            item.setTransform(QTransform.fromScale(fd, (v.track[1] - v.track[0]) / th))
            item.setPos(x0, v.track[0])
            item.setZValue(0.5)
            self._detail_items.append(item)

    # -- Contact dots ----------------------------------------------------------------
    def _candidates(self):
        '''All candidate contacts with P >= SLIDER_MIN (computed once per data set).'''
        if self._pair_cache is None:
            from .contacts import contact_pairs
            self._pair_cache = contact_pairs(self._pae, SLIDER_MIN, max_pairs=MAX_DOTS)
        return self._pair_cache

    def _update_dots(self):
        v = self._view
        v.dots = []
        v.dot_xy = v.dot_pairs = None
        if not self._dots_on or not self._pae.has_contact:
            self._update_picked_marker()
            return
        from .contacts import filter_pairs
        pairs, probs = filter_pairs(*self._candidates(), self._threshold)
        if len(pairs) == 0:
            self._update_picked_marker()
            return
        from Qt.QtGui import QPolygonF, QColor
        from Qt.QtCore import QPointF
        i, j = pairs[:, 0] + 0.5, pairs[:, 1] + 0.5
        v.dot_xy = np.concatenate([np.stack([j, i], 1), np.stack([i, j], 1)])
        v.dot_pairs = np.concatenate([pairs, pairs])
        # Blue by P, as the lines on the structure; drawn in DOT_LEVELS color steps
        levels = np.minimum((probs * DOT_LEVELS).astype(int), DOT_LEVELS - 1)
        for b in np.unique(levels):
            sel = pairs[levels == b]
            bi, bj = sel[:, 0] + 0.5, sel[:, 1] + 0.5
            pts = [QPointF(x, y) for x, y in zip(np.concatenate([bj, bi]), np.concatenate([bi, bj]))]
            rgb = contact_rgb((b + 0.5) / DOT_LEVELS)
            v.dots.append((QColor(*(int(c) for c in rgb)), QPolygonF(pts)))
        self._update_picked_marker()

    # -- Picked contact (click on a dot) --------------------------------------------------
    def _pick_contact(self, i, j):
        from . import contacts
        self._picked = (i, j)
        contacts.show_picked_contact(self._pae, i, j, zoom=self._zoom_to_pick)
        self._update_picked_marker()
        self._report(j, i)

    def clear_picked_contact(self):
        from . import contacts
        self._picked = None
        contacts.clear_picked_contact(self._pae.structure)
        self._update_picked_marker()

    def _update_picked_marker(self):
        v = self._view
        if self._picked is None:
            v.picked_xy = []
        else:
            i, j = self._picked
            v.picked_xy = [(j + 0.5, i + 0.5), (i + 0.5, j + 0.5)]
        v.viewport().update()

    def refresh_picked_contact(self):
        '''Re-draw the picked contact on the (new) structure after a model switch.'''
        if self._picked is not None and self._pae.has_contact:
            from . import contacts
            contacts.show_picked_contact(self._pae, *self._picked, log=False)

    # -- Structure contacts -------------------------------------------------------------
    def show_structure_contacts(self, show=True, min_prob=None):
        from . import contacts
        self._struct_contacts = bool(show) and self._pae.has_contact
        if min_prob is not None and abs(min_prob - self._threshold) > 1e-9:
            self.set_threshold(min_prob)    # Redraws the 3D contacts if they are shown
            if self._struct_contacts:
                return
        if self._struct_contacts:
            contacts.show_contacts(self._pae, min_p=self._threshold, region=self._struct_region,
                                   candidates=(self._candidates() if self._struct_region is None
                                               and self._threshold >= SLIDER_MIN else None))
        else:
            contacts.hide_contacts(self._pae.structure)
        self._hider.apply()                 # Drawing lines can switch the model to cartoon
        self._update_controls()
        self._refresh_colorbar()

    def _contacts_button_clicked(self, checked):
        '''Show Contacts draws all contacts above the slider, also after a box was dragged.'''
        if checked:
            self._struct_region = None
        self.show_structure_contacts(checked)

    def refresh_structure_contacts(self):
        if self._struct_contacts:
            self.show_structure_contacts(True)
        else:
            self._update_controls()
        try:
            self.refresh_picked_contact()
        except Exception as e:
            self.session.logger.warning(f'AlphaLens: could not re-draw clicked contact: {e}')

    # -- Colorbar ------------------------------------------------------------------
    def _draw_colorbar(self, mode, cmap, cbar):
        lo, hi, gamma, band_ticks = cbar
        from Qt.QtGui import QPixmap, QPainter, QColor, QImage, QPen
        from Qt.QtCore import Qt, QRectF, QPointF
        dpr = self._cbar.devicePixelRatioF()
        w, gh, H = 260, 12, 30
        x0, x1 = 20, 20 + w
        # Legend of the blue contact colors when dots or lines are shown
        dots = self._view.dots_visible or (self._struct_contacts and self._pae.has_contact)
        model_plain = mode.key == 'model' and not self._pae.has_contact
        extra = 170 + (200 if dots else 0) + (250 if mode.key == 'model' else 0)
        pm = QPixmap(int((w + extra) * dpr), int(H * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        t = np.linspace(0, 1, w)
        vals = lo + t * (hi - lo) if gamma is None else t      # gamma: bar is in display space
        row = np.ascontiguousarray(cmap.interpolated_rgba8(vals)[:, :3][None, :, :].repeat(gh, 0))
        img = QImage(row.data, w, gh, 3 * w, QImage.Format.Format_RGB888)
        if model_plain:         # Model distances without P: contact swatch and distance ramp
            p.setPen(QColor(90, 90, 90))
            fm = p.fontMetrics()
            rgb = tuple(int(v) for v in cmap.interpolated_rgba8([1.0])[0][:3])
            text = f'< {self._contact_cutoff():g} Å in model'
            p.fillRect(QRectF(x0, 2, 14, gh), QColor(*rgb))
            p.drawRect(QRectF(x0, 2, 14, gh))
            p.drawText(int(x0 + 18), 2 + gh, text)
            self._draw_distance_ramp(p, x0 + 18 + fm.horizontalAdvance(text) + 14, gh)
            p.end()
            self._cbar.setPixmap(pm)
            return
        p.drawImage(QRectF(x0, 2, w, gh), img)
        p.setPen(QColor(90, 90, 90))
        p.drawRect(QRectF(x0, 2, w, gh))
        if band_ticks is not None:
            ticks = np.array(band_ticks)
            pos = (ticks - lo) / (hi - lo if hi > lo else 1)
        elif gamma is None:
            if mode.key == 'pae':
                ticks = np.linspace(lo, hi, 7)
            else:       # Spread: round steps, not thirds of the range
                step = next((s for s in (0.5, 1, 2, 5, 10) if (hi - lo) / s <= 5), (hi - lo) / 4)
                ticks = np.arange(lo, hi + 1e-9, step)
            pos = (ticks - lo) / (hi - lo if hi > lo else 1)
        else:
            ticks = np.array([0, 0.05, 0.25, 0.5, 1.0])
            pos = ticks ** gamma
        fm = p.fontMetrics()
        for tv, tp in zip(ticks, pos):
            x = x0 + tp * w
            p.drawLine(int(x), 2 + gh, int(x), 5 + gh)
            s = f'{tv:g}'
            p.drawText(int(x - fm.horizontalAdvance(s) / 2), 5 + gh + fm.ascent(), s)
        unit = mode.unit
        if mode.key in ('contact', 'model'):
            unit = {'sqrt': 'P (√ scale)', 'linear': 'P', 'banded': 'P (bands)'}[self._contact_scale]
            if mode.key == 'model':
                unit += f', pairs < {self._contact_cutoff():g} Å'
        elif mode.key == 'agreement':
            unit = f'{self._pae.agreement.kind}s'
        p.drawText(x1 + 8, 2 + gh, unit)
        sx = x1 + 8 + fm.horizontalAdvance(unit) + 14
        if self._view_mode == 'spread' and self._far_mask() is not None:
            p.fillRect(QRectF(sx, 2, 14, gh), QColor(*UNDEFINED_GRAY))
            p.drawRect(QRectF(sx, 2, 14, gh))
            p.drawText(int(sx + 18), 2 + gh, 'far (>21.7 Å)')
            sx += 18 + fm.horizontalAdvance('far (>21.7 Å)') + 14
        elif mode.key == 'model':
            sx = self._draw_distance_ramp(p, sx, gh)
        if dots:
            # Predicted contacts (dots and lines): blue ramp from the threshold to 1
            label = 'contacts:'
            p.drawText(int(sx), 2 + gh, label)
            sx += fm.horizontalAdvance(label) + 6
            t = self._threshold
            lw = 70
            ramp = np.ascontiguousarray(contact_rgb(np.linspace(t, 1, lw))[None, :, :].repeat(gh, 0))
            p.drawImage(QRectF(sx, 2, lw, gh),
                        QImage(ramp.data, lw, gh, 3 * lw, QImage.Format.Format_RGB888))
            p.drawRect(QRectF(sx, 2, lw, gh))
            for tv, tx in ((f'{t:.2f}', sx), ('1', sx + lw)):
                p.drawText(int(tx - fm.horizontalAdvance(tv) / 2), 5 + gh + fm.ascent(), tv)
        p.end()
        self._cbar.setPixmap(pm)

    def _draw_distance_ramp(self, p, sx, gh):
        '''Grey ramp of the Model distances view (cutoff to DISTANCE_FAR Å); returns the next x.'''
        from Qt.QtGui import QImage
        from Qt.QtCore import QRectF
        fm = p.fontMetrics()
        label = 'distance:'
        p.drawText(int(sx), 2 + gh, label)
        sx += fm.horizontalAdvance(label) + 6
        lw = 70
        g = np.round(255 * np.linspace(DISTANCE_NEAR_GREY, 1.0, lw)).astype(np.uint8)
        ramp = np.ascontiguousarray(np.repeat(np.stack([g, g, g], -1)[None], gh, 0))
        p.drawImage(QRectF(sx, 2, lw, gh), QImage(ramp.data, lw, gh, 3 * lw, QImage.Format.Format_RGB888))
        p.drawRect(QRectF(sx, 2, lw, gh))
        for tv, tx in ((f'{self._contact_cutoff():g}', sx), (f'{DISTANCE_FAR:g}+ Å', sx + lw)):
            p.drawText(int(tx - fm.horizontalAdvance(tv) / 2), 5 + gh + fm.ascent(), tv)
        return sx + lw + 30

    def _set_chains(self):
        from .contacts import row_chain_ids
        chains = []
        for i, cid in enumerate(row_chain_ids(self._pae)):
            if chains and chains[-1][0] == cid:
                chains[-1][2] = i + 1
            else:
                chains.append([cid, i, i + 1])
        self._view.chains = [tuple(c) for c in chains]

    def show_chain_dividers(self, show=True, thickness=None):
        self._showing_chain_dividers = show
        self._view.show_dividers = show
        self._view.viewport().update()

    # ---------------------------------------------------------------------------
    # Color plot by structure colors
    #
    def _color_from_structure(self):
        rra = self._pae.row_residues_or_atoms()
        from chimerax.atomic import Residue
        blocks = {}
        for i, ra in enumerate(rra):
            if ra is None:
                continue
            c = tuple(ra.ribbon_color if isinstance(ra, Residue) else ra.color)
            blocks.setdefault(c, []).append(i)
        self._color_blocks = tuple(blocks.items())
        self._redraw(keep_zoom=True)

    def _clear_color_blocks(self):
        self._color_blocks = None
        self._redraw(keep_zoom=True)

    def _block_colored_rgb(self, vals, mode, cbar):
        lo, hi = cbar[0], cbar[1]
        if cbar[2] is not None:
            lo, hi = 0, 1

        def colored(sub, color):
            c = self._block_colormap(color, lo, hi, mode.dark_at_high)
            return c.interpolated_rgba8(sub.ravel())[:, :3].reshape(sub.shape + (3,))

        rgb = colored(vals, (0, 0, 0, 255))     # Gray outside same-color blocks
        for color, idx in self._color_blocks:
            sub = np.ix_(idx, idx)
            rgb[sub] = colored(vals[sub], color)
        return rgb

    def _block_colormap(self, color, lo, hi, dark_at_high):
        from chimerax.core.colors import Colormap
        fr = np.sqrt(np.linspace(0, 1, 7))
        if dark_at_high:
            fr = fr[::-1]
        fc = np.array(color, np.float32) / 255
        bg = np.ones(4, np.float32)
        return Colormap(tuple(np.linspace(lo, hi, 7)), [tuple((1 - f) * fc + f * bg) for f in fr])

    # ---------------------------------------------------------------------------
    # Mouse hover / drag
    #
    def _report(self, x, y):
        pae = self._pae
        n = pae.matrix_size
        msg = ''
        if 0 <= x < n and 0 <= y < n:
            cname, rname = pae.row_residue_or_atom_name(x), pae.row_residue_or_atom_name(y)
            vals = [f'PAE {pae.matrix("pae")[y, x]:.1f} Å']
            mt = pae.metrics
            if pae.has_contact:
                vals.append(f'P(d<{mt["cutoff"]:g}Å) {mt["contact"][y, x]:.2f}')
            if pae.has_distogram:
                sp = self._modes[self._spread_measure]
                pfar = mt['far'][y, x]
                if pfar > self._far_threshold:
                    vals.append(f'{sp.tab.split()[-1]} undefined')
                else:
                    vals.append(f'{sp.tab.split()[-1]} {sp.fmt % mt[sp.key][y, x]} {sp.unit}')
                vals += [f'P(d>{mt["far_edge"]:.1f}Å) {pfar:.2f}', f'E[d] {mt["mean"][y, x]:.1f} Å']
            from .contacts import model_distance
            md = model_distance(pae, y, x)
            if md is not None:
                vals.append(f'model {md[1]} {md[0]:.1f} Å')
            agr = pae.agreement
            if agr is not None:
                k = int(agr.counts(self._threshold)[y, x])
                vals.append(f'{k} of {agr.n} {agr.kind}s')
                if self._view_mode == 'agreement':
                    vals[-1] += ' (P ' + ' '.join(f'{v:.2f}' for v in agr.pair_values(y, x)) + ')'
            msg = f'{rname} ↔ {cname}:  ' + ',  '.join(vals)
        elif self._track_offset is not None and y >= self._track_offset and 0 <= x < n:
            msg = f'{pae.row_residue_or_atom_name(x)}:  pLDDT {self._plddt[x]:.1f}'
        self._info.setText(msg)

    def _rectangle_select(self, xy1, xy2):
        n = self._pae.matrix_size
        (x1, y1), (x2, y2) = xy1, xy2
        on_track = self._track_offset is not None and min(y1, y2) >= n
        if on_track:
            y1, y2 = x1, x2          # Drag inside the pLDDT track selects columns only
        clamp = lambda v: int(min(max(v, 0), n - 1))
        r1, r2 = clamp(min(x1, x2)), clamp(max(x1, x2))
        r3, r4 = clamp(min(y1, y2)), clamp(max(y1, y2))
        off_diagonal = (r2 < r3 or r4 < r1)
        if self._drag_colors_structure:
            if off_diagonal:
                self._color_rows(r1, r2, 'lime', 'gray')
                self._color_rows(r3, r4, 'magenta')
            else:
                self._color_rows(min(r1, r3), max(r2, r4), 'lime', 'gray')
        else:
            ranges = [(r1, r2), (r3, r4)] if off_diagonal else [(min(r1, r3), max(r2, r4))]
            self._select_rows(ranges)
        tiny = (r2 - r1) < 3 and (r4 - r3) < 3
        if on_track and tiny:
            self.session.selection.clear()      # A click on the track deselects
        elif on_track:
            self._select_rows([(r1, r2)])       # For "Hide selected" in the context menu
        if not tiny and self._pae.has_contact:
            # A dragged box draws the contacts inside it as lines on the structure
            self._struct_region = (r3, r4, r1, r2)
            self.show_structure_contacts(True)
        elif tiny and self._struct_contacts:
            # A click restores all contacts
            self._struct_region = None
            self.show_structure_contacts(True)

    def _color_rows(self, row1, row2, colorname, other_colorname=None):
        from chimerax.core.colors import BuiltinColors
        from chimerax.atomic import Residue, Atom
        from chimerax.alphafold.pae import _residue_and_atom_spec
        m = self._pae.structure
        if other_colorname:
            oc = BuiltinColors[other_colorname].uint8x4()
            m.residues.ribbon_colors = oc
            m.residues.atoms.colors = oc
        color = BuiltinColors[colorname].uint8x4()
        ras = [ra for ra in self._pae.row_residues_or_atoms()[row1:row2 + 1] if ra is not None]
        for ra in ras:
            if isinstance(ra, Residue):
                ra.ribbon_color = color
                ra.atoms.colors = color
            elif isinstance(ra, Atom):
                ra.color = color
                ra.residue.ribbon_color = color
        if ras:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, f'color {_residue_and_atom_spec(ras)} {colorname}')

    def _select_rows(self, ranges):
        from chimerax.alphafold.pae import _residue_and_atom_spec, _residues_and_atoms_to_atoms
        self.session.selection.clear()
        specs = []
        rra = self._pae.row_residues_or_atoms()
        for r1, r2 in ranges:
            ras = [ra for ra in rra[r1:r2 + 1] if ra is not None]
            _residues_and_atoms_to_atoms(ras).selected = True
            if ras:
                specs.append(_residue_and_atom_spec(ras))
        if specs:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, 'select ' + ' '.join(specs))

    # ---------------------------------------------------------------------------
    # Buttons / menu
    #
    def _color_domains(self):
        self._pae.color_domains(log_command=True)

    def _color_plddt(self):
        self._pae.color_plddt(log_command=True)

    def _fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction

        def toggle(text, checked, cb, target=menu):
            a = QAction(text, target)
            a.setCheckable(True)
            a.setChecked(checked)
            a.triggered.connect(cb)
            target.addAction(a)

        def set_drag(checked):
            self._drag_colors_structure = checked

        def set_plddt(checked):
            self._show_plddt = checked
            self._redraw(keep_zoom=True)

        def set_dots_all(checked):
            self._dots_all_views = checked
            self._redraw(keep_zoom=True)

        hide = menu.addAction('Hide selected', self._hide_selected)
        hide.setEnabled(bool(self._selected_to_hide().any()))
        show = menu.addAction('Show hidden', self._show_hidden)
        show.setEnabled(bool(self._selected_hidden.any()))
        menu.addSeparator()
        menu.addAction('Fit plot (reset zoom)', self._view.reset_zoom)
        toggle('Dragging box colors structure', self._drag_colors_structure, set_drag)
        menu.addAction('Color plot from structure', self._color_from_structure)
        menu.addAction('Color plot by value', self._clear_color_blocks)
        toggle('Show chain divider lines', self._showing_chain_dividers, self.show_chain_dividers)
        toggle('Show pLDDT track', self._show_plddt, set_plddt)
        if self._pae.has_contact:
            menu.addSeparator()
            sub = menu.addMenu('Contact color scale')
            for key, label in (('sqrt', 'Enhanced (√P)'), ('linear', 'Linear'),
                               ('banded', 'Bands (0.05 / 0.2 / 0.5 / 0.8)')):
                toggle(label, self._contact_scale == key,
                       lambda checked, k=key: self.set_contact_scale(k), sub)
            toggle('Show contact dots', self._dots_on, self.set_dots_shown)
            toggle('Show contact dots in all views', self._dots_all_views, set_dots_all)
            toggle('Zoom 3D view to clicked contact', self._zoom_to_pick,
                   lambda checked: setattr(self, '_zoom_to_pick', checked))
            if self._picked is not None:
                menu.addAction('Clear clicked contact', self.clear_picked_contact)
            if self._struct_region is not None:
                menu.addAction('Contacts on structure: whole matrix', self._reset_contact_region)
        if self._pae.has_distogram:
            menu.addSeparator()
            toggle('Spread as standard deviation (Å)', self._spread_measure == 'std',
                   lambda checked: self.set_spread_measure('std'))
            toggle('Spread as entropy (bits)', self._spread_measure == 'entropy',
                   lambda checked: self.set_spread_measure('entropy'))
            sub = menu.addMenu('Grey out spread where P(d > 21.7 Å) >')
            for t in (0.2, 0.5, 0.8, 1.0):
                toggle(f'{t:g}' if t < 1 else 'never', self._far_threshold == t,
                       lambda checked, t=t: self.set_far_threshold(t), sub)
        if self._pae.agreement is not None:
            menu.addSeparator()
            menu.addAction('Log agreement table', self._log_agreement)
        menu.addSeparator()
        menu.addAction('Save image', self._save_image)

    def _reset_contact_region(self):
        self._struct_region = None
        self.show_structure_contacts(self._struct_contacts)

    def _save_image(self):
        import os
        s = self._pae.structure
        base = os.path.splitext(os.path.basename(getattr(s, 'filename', None) or s.name))[0]
        suggested = os.path.join(os.getcwd(), f'{base}_{self.mode.key}.png')
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self.tool_window.ui_area, 'Save plot image', suggested)
        if path:
            self.save_image(path)

    def save_image(self, path):
        self.tool_window.ui_area.grab().save(path)

    def _export_csv(self):
        import os
        import re
        name = re.sub(r'[^\w.+-]+', '_', self._pae.structure.name).strip('_') or 'alphalens'
        suggested = os.path.join(os.getcwd(), f'{name}_contacts.csv')
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self.tool_window.ui_area, 'Save contacts as CSV',
                                              suggested, 'CSV files (*.csv)')
        if path:
            self.save_contacts_csv(path)

    def save_contacts_csv(self, path):
        '''Contacts with P >= the slider threshold, inside the dragged box if there is one.'''
        from .cmd import save_contacts_csv
        return save_contacts_csv(self.session, self._pae, path, self._threshold,
                                 region=self._struct_region)

    # ---------------------------------------------------------------------------
    # Sessions store the settings and the paths of the prediction files, never the matrices:
    # a distogram is gigabytes. The plot is rebuilt from those files when the session opens.
    #
    def take_snapshot(self, session, flags):
        pae = self._pae
        s = pae.structure
        cmap = self._pae_colormap
        return {
            'version': self.SNAPSHOT_VERSION,
            'structure': s,
            'model_file': getattr(s, 'filename', None),
            'pae_file': pae._pae_path,
            'distogram_file': self._distogram_file,
            'cutoff': self.cutoff,
            'mode': self._mode,
            'spread_measure': self._spread_measure,
            'contact_scale': self._contact_scale,
            'far_threshold': self._far_threshold,
            'threshold': self._threshold,
            'dots_on': self._dots_on,
            'dots_all_views': self._dots_all_views,
            'mask_matrix': self._mask_matrix,
            'struct_contacts': self._struct_contacts,
            'struct_region': self._struct_region,
            'picked': None if self._picked is None else tuple(int(i) for i in self._picked),
            'zoom_to_pick': self._zoom_to_pick,
            'drag_colors_structure': self._drag_colors_structure,
            'divider_lines': self._showing_chain_dividers,
            'show_plddt': self._show_plddt,
            'color_blocks': self._color_blocks is not None,
            'plddt_slider': self._plddt_slider.value(),
            'selected_hidden': np.flatnonzero(self._selected_hidden).tolist(),
            'hider': self._hider.state(),
            'pae_colormap': None if cmap is None else (cmap.data_values.tolist(), cmap.colors.tolist()),
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        def restore(*_):
            from chimerax.core.triggerset import DEREGISTER
            from .cmd import restore_plot
            try:
                restore_plot(session, data)
            except Exception as e:      # A plot that cannot be rebuilt must not lose the session
                session.logger.warning(f'AlphaLens: could not restore the plot of '
                                       f'{data.get("structure")}: {type(e).__name__}: {e}')
            return DEREGISTER
        session.triggers.add_handler('end restore session', restore)
        return _PlotBeingRestored()

    def restore_settings(self, st):
        '''Apply saved settings. What the session already restored (contact lines) is not redrawn.'''
        self._spread_measure = st.get('spread_measure', self._spread_measure)
        self._contact_scale = st.get('contact_scale', self._contact_scale)
        self._far_threshold = st.get('far_threshold', self._far_threshold)
        self._dots_on = st.get('dots_on', self._dots_on)
        self._dots_all_views = st.get('dots_all_views', self._dots_all_views)
        self._zoom_to_pick = st.get('zoom_to_pick', self._zoom_to_pick)
        self._drag_colors_structure = st.get('drag_colors_structure', self._drag_colors_structure)
        self._show_plddt = st.get('show_plddt', self._show_plddt)
        self._mask_box.blockSignals(True)
        self._mask_box.setChecked(bool(st.get('mask_matrix', False)))
        self._mask_box.blockSignals(False)
        self._mask_matrix = self._mask_box.isChecked()
        self.set_threshold(st.get('threshold', self._threshold))    # Before _struct_contacts is set
        n = len(self._plddt)
        region = st.get('struct_region')
        self._struct_region = None if region is None or max(region) >= n else tuple(region)
        picked = st.get('picked')
        self._picked = None if picked is None or max(picked) >= n else tuple(picked)
        self._update_picked_marker()
        # Preserve the requested state while AF2 contact probabilities are still loading.
        self._struct_contacts = bool(st.get('struct_contacts'))
        self._hider.set_state(st.get('hider', {}))
        hidden = np.asarray(st.get('selected_hidden', ()), int)
        self._selected_hidden[hidden[(hidden >= 0) & (hidden < n)]] = True
        self._plddt_slider.blockSignals(True)
        self._plddt_slider.setValue(int(st.get('plddt_slider', 0)))
        self._plddt_slider.blockSignals(False)
        self._plddt_value.setText(str(self._plddt_slider.value()))
        self._apply_hidden()
        if st.get('color_blocks'):
            self._color_from_structure()
        self.restore_mode(st.get('mode'))
        self._update_controls()
        self._update_dots()
        self._redraw(keep_zoom=False)

    def restore_mode(self, mode):
        '''Show the saved view when its data are there. Agreement is read only when asked for.'''
        if mode in self.available_modes() and mode != 'agreement':
            self.set_mode(mode)

    def restore_loaded(self, st):
        '''The data have been read: go to a view that had to wait for them.'''
        mode = st.get('mode')
        if mode == 'agreement' and 'agreement' in self.available_modes():
            self.set_mode('agreement')
        else:
            self.restore_mode(mode)
        if self._struct_contacts and not self._pae.has_contact:
            self._struct_contacts = False
        self._update_controls()

    # ---------------------------------------------------------------------------
    #
    def closed(self):
        return self.tool_window.tool_instance is None

    def _structure_deleted(self, *args):
        if not self.closed():
            self.delete()

    def delete(self):
        self._detach_structure()
        self._restore_hidden()              # Without the plot, nothing could show them again
        v = self._view
        try:                       # Nothing may fire into a half-deleted tool
            self._plddt_timer.stop()
            self._plddt_timer.timeout.disconnect()
            v._timer.stop()
            v._timer.timeout.disconnect()
            self._slider_timer.stop()
            self._slider_timer.timeout.disconnect()
            self._progress_timer.stop()
            self._progress_timer.timeout.disconnect()
        except Exception:
            pass
        v.dot_click_cb = None
        v._view_changed_cb = v._key_callback = None
        v.dots = []
        self._render = self._block_rgb = None
        ToolInstance.delete(self)
