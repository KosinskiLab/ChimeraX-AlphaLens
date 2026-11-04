"""
DistoPAE plot panel: PAE matrix with a pLDDT track, plus contact-probability and
distogram-spread views reachable with the arrow buttons / tabs when the data exist.

All three views use the AlphaFold-DB green ramp (dark green = confident: low PAE,
high contact probability, sharp distance distribution).

Interaction: scroll wheel / pinch zooms at the cursor, Shift-drag or middle-drag pans,
"Fit" resets. When zoomed into a large matrix the
visible region is redrawn at full resolution.
"""

from html import escape

import numpy as np
from chimerax.core.tools import ToolInstance
from chimerax.alphafold.pae import PAEView, pae_pixmap

from .contacts import BAND_EDGES, BAND_RGB

# pLDDT colors used by AlphaFold / the "alphafold" palette.
PLDDT_BINS = ((90, (0, 83, 214)), (70, (101, 203, 243)), (50, (255, 219, 19)),
              (-1e9, (255, 125, 69)))

MAX_DISPLAY = 900          # Matrices larger than this are pooled for the overview image
MAX_DETAIL = 1200          # Max pixels per side when redrawing a zoomed region
MAX_PIXELS_PER_RESIDUE = 40
MAX_DOTS = 40000
SLIDER_MIN = 0.05          # Lowest contact threshold offered by the slider
UNDEFINED_GRAY = (190, 190, 190)   # Spread view: pairs whose distance is mostly in the last bin


def green_colormap(lo, hi, dark_at_high=False):
    '''AlphaFold-DB style ramp: dark green -> white (paegreen colors).'''
    from chimerax.core.colors import BuiltinColormaps, Colormap
    g = BuiltinColormaps['paegreen']
    colors = [tuple(c) for c in g.colors]
    if dark_at_high:
        colors = colors[::-1]
    return Colormap(tuple(np.linspace(lo, hi, len(colors))), colors)


CONTACT_BAND_EDGES = (0.0, 0.05) + BAND_EDGES + (1.0,)


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
        'contact': Mode('contact', f'Contacts < {c} Å', f'Contact probability P(d < {c} Å)',
                        '', '%.2f', 'max', True),
        'std': Mode('std', 'Spread σ', 'Distogram spread (standard deviation of distance)',
                    'Å', '%.2f', 'mean', False),
        'entropy': Mode('entropy', 'Spread H', 'Distogram spread (entropy of distance bins)',
                        'bits', '%.2f', 'mean', False),
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
class DistoPAEView(PAEView):
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
        if keep_zoom and self.zoomed:
            center = self.mapToScene(self.viewport().rect().center())
            scene.setSceneRect(-m, -m, n + 1.3 * m, h + 1.3 * m)
            self.centerOn(center)
        else:
            scene.setSceneRect(-m, -m, n + 1.3 * m, h + 1.3 * m)
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
        for color, width in ((QColor(0, 0, 0), 4.5), (QColor(255, 200, 0), 2.5)):
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
class DistoPAEPlot(ToolInstance):

    name = 'DistoPAE Plot'
    SESSION_ENDURING = False
    SESSION_SAVE = False

    def __init__(self, session, pae, cutoff=8.0, pae_colormap=None, divider_lines=True):
        self._pae = pae                     # DistoPAE instance
        self.cutoff = cutoff
        self._modes = make_modes(cutoff)
        self._mode_cycle = ['pae', 'contact', 'spread']
        self._mode = 'pae'
        self._spread_measure = 'std'
        self._contact_scale = 'sqrt'        # 'sqrt', 'linear' or 'banded'
        self._far_threshold = 0.5           # Grey out spread where P(d > last bin edge) exceeds this
        self._threshold = 0.2               # Contacts (dots, 3D links) shown for P >= this
        self._dots_on = True
        self._dots_all_views = False
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
        self._models = []

        ToolInstance.__init__(self, session, 'DistoPAE Plot')

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

        # Mode switcher:  [<] [PAE][Contacts][Spread] [>]   [Fit]
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
            b.setMinimumWidth(110)
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

        self._view = gv = DistoPAEView(parent, self._rectangle_select, lambda: None,
                                       self._report, self.step_mode, self._update_detail)
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
        tl.addWidget(QLabel('Contacts  P ≥', tw_))
        self._slider = sl_ = QSlider(Qt.Orientation.Horizontal, tw_)
        sl_.setRange(int(SLIDER_MIN * 100), 100)
        sl_.setValue(int(round(self._threshold * 100)))
        sl_.setToolTip('Hide contacts (dots and 3D links) with probability below this')
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
        cbn.clicked.connect(lambda checked: self.show_structure_contacts(checked))
        bl.addWidget(cbn)
        b = QPushButton('Save Image', bw)
        b.clicked.connect(self._save_image)
        bl.addWidget(b)
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

    def _detach_structure(self):
        h = getattr(self, '_structure_delete_handler', None)
        if h is not None:
            try:
                self._pae.structure.triggers.remove_handler(h)
            except Exception:
                pass
            self._structure_delete_handler = None

    def set_pae(self, pae):
        '''Show another model (same prediction) in this window.'''
        same_size = pae.matrix_size == self._pae.matrix_size
        self._detach_structure()
        from . import contacts
        contacts.clear_picked_contact(self._pae.structure)
        self._pae = pae
        self._attach_structure()
        self._pair_cache = None
        if not same_size:
            self._picked = None
        self._plddt = pae.row_plddt()
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
            self.session.logger.error(f'DistoPAE: could not switch model: {e}')
            self._populate_models()

    # ---------------------------------------------------------------------------
    # Modes
    #
    @property
    def _view_mode(self):
        '''Requested view if its data are available, else PAE (choice is remembered).'''
        return self._mode if self._mode in self.available_modes() else 'pae'

    @property
    def mode(self):
        m = self._view_mode
        return self._modes[self._spread_measure if m == 'spread' else m]

    def available_modes(self):
        pae = self._pae
        modes = ['pae']
        if pae.has_contact:
            modes.append('contact')
        if pae.has_distogram:
            modes.append('spread')
        return modes

    def step_mode(self, step):
        modes = self.available_modes()
        if len(modes) < 2:
            return
        i = modes.index(self._view_mode)
        self._mode = modes[(i + step) % len(modes)]
        self._redraw(keep_zoom=True)

    def set_mode(self, name):
        if name in ('std', 'entropy'):
            self._spread_measure = name
            name = 'spread'
        if name not in self.available_modes():
            from chimerax.core.errors import UserError
            need = 'a distogram' if name == 'spread' else 'a distogram or AF3 contact_probs'
            raise UserError(f'View "{name}" needs {need}; {self._pae.structure}: {self._data_status}')
        self._mode = name
        self._redraw(keep_zoom=True)

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
    def distogram_loading(self, path):
        self._data_status = f'Loading distogram {path} ...'
        self._update_controls()

    def distogram_loaded(self):
        self._data_status = 'ok'
        self.data_changed()

    def distogram_failed(self, message):
        self._data_status = f'Distogram could not be read: {message}'
        self._update_controls()

    def _update_controls(self):
        avail = self.available_modes()
        on = len(avail) > 1
        tip = 'Switch view (or use ←/→ keys in the plot)' if on else self._data_status
        for b in (self._left, self._right):
            b.setEnabled(on)
            b.setToolTip(tip)
        reasons = {'contact': 'Needs a distogram or AlphaFold3 contact_probs',
                   'spread': 'Needs a distogram'}
        for key, b in self._tabs.items():
            enabled = key in avail
            b.setEnabled(enabled)
            b.setToolTip('' if enabled else f'{reasons.get(key, "")} ({self._data_status})')
        cb = self._contacts_button
        cb.setEnabled(self._pae.has_contact)
        cb.setChecked(self._struct_contacts and self._pae.has_contact)
        cb.setText('Hide Contacts' if cb.isChecked() else 'Show Contacts')
        self._threshold_row.setEnabled(self._pae.has_contact)
        cb.setToolTip(f'Draw predicted contacts (P ≥ {self._threshold:.2f}) on the structure; '
                      'while shown, dragging a box limits them to that region'
                      if self._pae.has_contact else 'Needs contact probabilities')

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
        # Spread: range normalized to this prediction (99th percentile of defined pairs)
        far = self._far_mask()
        defined = m[~far] if far is not None else m
        hi = float(np.percentile(defined, 99)) if defined.size else 1.0
        hi = max(0.5, np.ceil(hi * 2) / 2)
        return green_colormap(0, hi), m, (0, hi, None, None)

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

    def _redraw(self, keep_zoom=False):
        vm = self._view_mode
        mode = self.mode
        m = self._pae.matrix(mode.key)
        n = m.shape[0]
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
        if mode.key == 'contact' and self._pae.metrics.get('contact_only'):
            src = ' <span style="color:gray">(AF3 contact_probs; no distogram)</span>'
        pool = (f' <span style="color:gray">[overview {f}×{f} {mode.pool}-pooled; '
                f'zoom in for full resolution]</span>') if f > 1 else ''
        if far is not None:
            edge = self._pae.metrics.get('far_edge', 21.7)
            src = (f' <span style="color:gray">grey: P(d &gt; {edge:.1f} Å) &gt; '
                   f'{self._far_threshold:g}, spread undefined</span>')
        self._heading.setText(f'<html><b>{escape(mode.title)}</b>{src}{pool}<br>'
                              f'{escape(str(self._pae.structure))} · {n} residues. '
                              f'Drag a box to color residues.{nav}</html>')
        for key, b in self._tabs.items():
            label = self._modes[key].tab if key != 'spread' else self._modes[self._spread_measure].tab
            b.setText(label)
            b.setChecked(key == vm)
        self._draw_colorbar(mode, cmap, cbar)

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
        from .contacts import filter_pairs, band_index
        pairs, probs = filter_pairs(*self._candidates(), self._threshold)
        if len(pairs) == 0:
            self._update_picked_marker()
            return
        from Qt.QtGui import QPolygonF, QColor
        from Qt.QtCore import QPointF
        i, j = pairs[:, 0] + 0.5, pairs[:, 1] + 0.5
        v.dot_xy = np.concatenate([np.stack([j, i], 1), np.stack([i, j], 1)])
        v.dot_pairs = np.concatenate([pairs, pairs])
        bands = np.maximum(band_index(probs), 0)       # Below 0.2 drawn as the lightest band
        for b in range(len(BAND_RGB)):
            sel = pairs[bands == b]
            if len(sel) == 0:
                continue
            bi, bj = sel[:, 0] + 0.5, sel[:, 1] + 0.5
            pts = [QPointF(x, y) for x, y in zip(np.concatenate([bj, bi]), np.concatenate([bi, bj]))]
            v.dots.append((QColor(*BAND_RGB[b]), QPolygonF(pts)))
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
                                   candidates=self._candidates() if self._threshold >= SLIDER_MIN else None)
        else:
            contacts.hide_contacts(self._pae.structure)
        self._update_controls()

    def refresh_structure_contacts(self):
        if self._struct_contacts:
            self.show_structure_contacts(True)
        else:
            self._update_controls()
        try:
            self.refresh_picked_contact()
        except Exception as e:
            self.session.logger.warning(f'DistoPAE: could not re-draw clicked contact: {e}')

    # -- Colorbar ------------------------------------------------------------------
    def _draw_colorbar(self, mode, cmap, cbar):
        lo, hi, gamma, band_ticks = cbar
        from Qt.QtGui import QPixmap, QPainter, QColor, QImage, QPen
        from Qt.QtCore import Qt, QRectF, QPointF
        dpr = self._cbar.devicePixelRatioF()
        w, gh, H = 260, 12, 30
        x0, x1 = 20, 20 + w
        dots = self._view.dots_visible
        pm = QPixmap(int((w + (380 if dots else 170)) * dpr), int(H * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        t = np.linspace(0, 1, w)
        vals = lo + t * (hi - lo) if gamma is None else t      # gamma: bar is in display space
        row = np.ascontiguousarray(cmap.interpolated_rgba8(vals)[:, :3][None, :, :].repeat(gh, 0))
        img = QImage(row.data, w, gh, 3 * w, QImage.Format.Format_RGB888)
        p.drawImage(QRectF(x0, 2, w, gh), img)
        p.setPen(QColor(90, 90, 90))
        p.drawRect(QRectF(x0, 2, w, gh))
        if band_ticks is not None:
            ticks = np.array(band_ticks)
            pos = ticks
        elif gamma is None:
            ticks = np.linspace(lo, hi, 4 if mode.key != 'pae' else 7)
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
        if mode.key == 'contact':
            unit = {'sqrt': 'P (√ scale)', 'linear': 'P', 'banded': 'P (bands)'}[self._contact_scale]
        p.drawText(x1 + 8, 2 + gh, unit)
        sx = x1 + 8 + fm.horizontalAdvance(unit) + 14
        if self._view_mode == 'spread' and self._far_mask() is not None:
            p.fillRect(QRectF(sx, 2, 14, gh), QColor(*UNDEFINED_GRAY))
            p.drawRect(QRectF(sx, 2, 14, gh))
            p.drawText(int(sx + 18), 2 + gh, 'far (>21.7 Å)')
        elif dots:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.drawText(int(sx), 2 + gh, 'dots:')
            sx += fm.horizontalAdvance('dots:') + 8
            t = self._threshold
            edges = list(BAND_EDGES[1:]) + [None]
            for b, rgb in enumerate(BAND_RGB):
                hi_edge = edges[b]
                if hi_edge is not None and hi_edge <= t + 1e-9:
                    continue
                lo_edge = BAND_EDGES[b] if b > 0 else t
                lo_edge = max(lo_edge, t)
                p.setPen(QPen(QColor(25, 25, 25), 1))
                p.setBrush(QColor(*rgb))
                p.drawEllipse(QPointF(sx + 5, 2 + gh / 2), 4.5, 4.5)
                p.setPen(QColor(90, 90, 90))
                label = f'≥ {lo_edge:g}' if hi_edge is None else f'{lo_edge:g}–{hi_edge:g}'
                p.drawText(int(sx + 12), 2 + gh, label)
                sx += 12 + fm.horizontalAdvance(label) + 8
        p.end()
        self._cbar.setPixmap(pm)

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
            msg = f'{rname} ↔ {cname}:  ' + ',  '.join(vals)
        elif self._track_offset is not None and y >= self._track_offset and 0 <= x < n:
            msg = f'{pae.row_residue_or_atom_name(x)}:  pLDDT {self._plddt[x]:.1f}'
        self._info.setText(msg)

    def _rectangle_select(self, xy1, xy2):
        n = self._pae.matrix_size
        (x1, y1), (x2, y2) = xy1, xy2
        if self._track_offset is not None and min(y1, y2) >= n:
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
        if not tiny and self._pae.has_contact:
            # A dragged box draws the contacts inside it as red lines on the structure
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

    # ---------------------------------------------------------------------------
    #
    def closed(self):
        return self.tool_window.tool_instance is None

    def _structure_deleted(self, *args):
        if not self.closed():
            self.delete()

    def delete(self):
        self._detach_structure()
        v = self._view
        try:                       # Nothing may fire into a half-deleted tool
            v._timer.stop()
            v._timer.timeout.disconnect()
            self._slider_timer.stop()
            self._slider_timer.timeout.disconnect()
        except Exception:
            pass
        v.dot_click_cb = None
        v._view_changed_cb = v._key_callback = None
        v.dots = []
        self._render = self._block_rgb = None
        ToolInstance.delete(self)
