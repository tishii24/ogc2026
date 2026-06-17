"""
Bay layout visualization widget (redesigned)

Layout:
  [Layers: L0 L1 ...]  [Scale - +]  [Reset]       <- top controls
  +--------------+--------------------------------+
  |  Block Panel |  Bay 0   Bay 1   Bay 2  ...      |
  |  (vertical scroll) |  (horizontal scroll)                  |
  +--------------+--------------------------------+

- Block panel: blocks listed vertically by ID. Click=rotate, drag->move to bay.
- Bay panel: bays listed horizontally. Block drag (bay<->bay, bay<->panel).
- Drag: centralized mouse event management in BayLayoutTab + semi-transparent ghost overlay.
"""

import math
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QScrollArea, QCheckBox,
    QSplitter, QToolTip, QComboBox, QApplication,
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF,
    QFont, QFontMetricsF, QCursor, QPainterPath,
)
from shapely.geometry import (Point as ShapelyPoint, Polygon as ShapelyPolygon,
                               LineString as ShapelyLine)
from shapely.ops import unary_union
from utils import (
    Bay as UtilsBay,
    Block as UtilsBlock,
    CollisionResult,
    check_collisions,
    _resolve_layers,
    _translate_verts,
    _anchor_verts,
    _bounding_box,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_LAYER_GRAY_BASE = 80
_LAYER_GRAY_STEP = 28
_LAYER_GRAY_MAX  = 210

PANEL_SCALE   = 5.0   # Block panel initial scale (auto-recalculated on instance load)
BAY_SCALE_DEF = 7.0   # Bay initial scale (auto-recalculated on instance load)
BAY_PAD       = 10
PANEL_PAD     = 8
PANEL_INFO_W  = 70   # Block info text column width (px)
PANEL_SHAPE_W = 120   # Block shape column width (px)
PANEL_W       = PANEL_INFO_W + PANEL_SHAPE_W + 8
BLOCK_MARGIN  = 10
XCUT_LAYER_W  = 12   # X cross-section layer column width (px)
YCUT_LAYER_H  = 12   # Y cross-section layer row height (px)

# Auto-scale calculation -- fit bay width to screen
_BAY_TARGET_PX_W = 800   # Target pixel width for a single bay
_BAY_TARGET_PX_H = 240   # Target pixel height for a single bay
_SCALE_MIN = 0.5
_SCALE_MAX = 30.0
_SCALE_STEP = 0.5        # Increment per -/+ button click


def _auto_scale(bay_w: float, bay_h: float) -> float:
    """Determine initial scale as the minimum of both constraints given bay dimensions."""
    s_w = _BAY_TARGET_PX_W / max(bay_w, 1)
    s_h = _BAY_TARGET_PX_H / max(bay_h, 1)
    raw = min(s_w, s_h)
    # Round to nearest 0.5
    return max(_SCALE_MIN, min(_SCALE_MAX, round(raw / _SCALE_STEP) * _SCALE_STEP))


def _layer_gray(layer_idx: int) -> int:
    return min(_LAYER_GRAY_MAX, _LAYER_GRAY_BASE + layer_idx * _LAYER_GRAY_STEP)


def _point_in_poly(px: float, py: float, verts: list) -> bool:
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _block_color(blk_id: int, li: int, alpha: int = 200) -> QColor:
    """Unique color per block ID * layer -- for block distinction in bay and cross-section views."""
    hue = (blk_id * 67 + 30) % 360
    sat = max(100, 200 - li * 25)
    val = max(140, 220 - li * 20)
    return QColor.fromHsv(hue, sat, val, alpha)


def _compute_cross_section(blocks, bay_w, bay_h, axis, pos, visible_layers):
    """Return list of cross-section segments for each block-layer.

    axis='x' -> vertical line X=pos, segment values are bay Y coordinates
    axis='y' -> horizontal line Y=pos, segment values are bay X coordinates
    Returns: [(blk_id, layer_idx, seg_start, seg_end), ...]
    """
    BIG = max(bay_w, bay_h) * 2 + 1
    if axis == 'x':
        cut = ShapelyLine([(pos, -BIG), (pos, BIG)])
    else:
        cut = ShapelyLine([(-BIG, pos), (BIG, pos)])
    segments = []
    for blk in blocks:
        for li, layer_verts in enumerate(blk.layers_at_pos()):
            if visible_layers is not None and li not in visible_layers:
                continue
            if len(layer_verts) < 3:
                continue
            poly = ShapelyPolygon(layer_verts)
            if not poly.intersects(cut):
                continue
            inter = poly.intersection(cut)
            lines = list(inter.geoms) if hasattr(inter, 'geoms') else [inter]
            for line in lines:
                try:
                    coords = list(line.coords)
                except Exception:
                    continue
                vals = [c[1] if axis == 'x' else c[0] for c in coords]
                if len(vals) >= 2:
                    segments.append((blk.id, li, min(vals), max(vals)))
    return segments


def _compute_collision_cross_section(collision_results, bay_w, bay_h, axis, pos):
    """Return cross-section segments of the collision intersection polygon with the cut line.

    Returns: [(layer_idx, seg_start, seg_end), ...]
    """
    BIG = max(bay_w, bay_h) * 2 + 1
    if axis == 'x':
        cut = ShapelyLine([(pos, -BIG), (pos, BIG)])
    else:
        cut = ShapelyLine([(-BIG, pos), (BIG, pos)])
    segments = []
    for r in collision_results:
        inter_poly = r.intersection
        if inter_poly is None or inter_poly.is_empty:
            continue
        geoms = list(inter_poly.geoms) if hasattr(inter_poly, 'geoms') else [inter_poly]
        for geom in geoms:
            try:
                poly = ShapelyPolygon(list(geom.exterior.coords))
            except Exception:
                continue
            if not poly.intersects(cut):
                continue
            inter = poly.intersection(cut)
            lines = list(inter.geoms) if hasattr(inter, 'geoms') else [inter]
            for line in lines:
                try:
                    coords = list(line.coords)
                except Exception:
                    continue
                vals = [c[1] if axis == 'x' else c[0] for c in coords]
                if len(vals) >= 2:
                    segments.append((r.layer_index, min(vals), max(vals)))
    return segments


def _draw_hatch(painter, rect: QRectF, step: int = 4):
    """Draw hatch lines for cross-section emphasis -- 45° diagonal lines clipped to rect."""
    painter.save()
    painter.setClipRect(rect)
    painter.setPen(QPen(QColor(0, 0, 0, 65), 0.75))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
    i = step
    while i < rw + rh:
        painter.drawLine(QPointF(rx + i - rh, ry + rh), QPointF(rx + i, ry))
        i += step
    painter.restore()


# -----------------------------------------------------------------------------
# BlockItem
# -----------------------------------------------------------------------------
class BlockItem:
    def __init__(self, block_id: int, block_data: dict):
        self.id = block_id
        self.display_id = block_id
        self.data = block_data
        self.orient_idx = 0
        self.x: float = 0.0
        self.y: float = 0.0

    @property
    def orientations(self):
        return self.data["shape"]

    @property
    def current_orient(self):
        if not self.orientations:
            return {"orientation": 0, "layers": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
        return self.orientations[self.orient_idx]

    @property
    def orientation_index(self):
        return self.current_orient["orientation"]

    def resolved_layers(self):
        return _resolve_layers(self.current_orient["layers"])

    def anchored_layers(self):
        return self.resolved_layers()

    def layers_at_pos(self):
        anchored = self.anchored_layers()
        return [_translate_verts(l, self.x, self.y) for l in anchored]

    def bounding_rect_at_pos(self):
        layers = self.layers_at_pos()
        if not layers:
            return (self.x, self.y, self.x + 1, self.y + 1)
        all_v = [v for l in layers for v in l]
        return _bounding_box(all_v)

    def anchored_bb(self):
        layers = self.anchored_layers()
        if not layers:
            return (0, 0, 1, 1)
        all_v = [v for l in layers for v in l]
        return _bounding_box(all_v)

    def rotate_next(self):
        self.orient_idx = (self.orient_idx + 1) % len(self.orientations)


# -----------------------------------------------------------------------------
# Common block drawing helper
# -----------------------------------------------------------------------------
def _draw_block_at(painter, blk, origin_px, scale, visible_layers,
                   alpha=230, flip_y=True):
    all_layers = blk.anchored_layers()
    if not all_layers:
        return
    vis_indices = [i for i in range(len(all_layers))
                   if visible_layers is None or i in visible_layers]
    if not vis_indices:
        return

    def to_px(bx, by):
        if flip_y:
            return QPointF(origin_px.x() + bx * scale,
                           origin_px.y() - by * scale)
        return QPointF(origin_px.x() + bx * scale,
                       origin_px.y() + by * scale)

    def verts_to_poly(verts):
        poly = QPolygonF()
        for x, y in verts:
            poly.append(to_px(x, y))
        return poly

    top_idx = vis_indices[-1]
    vis_verts = []

    # 1) Fill pass
    for li in vis_indices:
        layer = all_layers[li]
        if not layer:
            continue
        gray = _layer_gray(li)
        painter.setBrush(QBrush(QColor(gray, gray, gray, alpha)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(verts_to_poly(layer))
        vis_verts.extend(layer)

    # 2) Dashed layer boundaries (all layers, drawn first so outer line covers them)
    pen_dash = QPen(QColor("#555555"), 1.0)
    pen_dash.setStyle(Qt.PenStyle.DashLine)
    pen_dash.setDashPattern([4, 3])
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(pen_dash)
    for li in vis_indices:
        layer = all_layers[li]
        if not layer:
            continue
        painter.drawPolygon(verts_to_poly(layer))

    # 3) Solid outer boundary = union of all visible layers
    pen_solid = QPen(QColor("#222222"), 2.0)
    pen_solid.setStyle(Qt.PenStyle.SolidLine)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(pen_solid)
    shapely_polys = [
        ShapelyPolygon(all_layers[li])
        for li in vis_indices
        if all_layers[li] and len(all_layers[li]) >= 3
    ]
    if shapely_polys:
        outer = unary_union(shapely_polys)
        geoms = list(outer.geoms) if hasattr(outer, 'geoms') else [outer]
        for geom in geoms:
            try:
                coords = list(geom.exterior.coords)
            except AttributeError:
                continue
            painter.drawPolygon(verts_to_poly(coords))

    if vis_verts:
        cx = sum(v[0] for v in vis_verts) / len(vis_verts)
        cy = sum(v[1] for v in vis_verts) / len(vis_verts)
        center = to_px(cx, cy)
        id_part     = f"B{blk.display_id}"
        orient_part = f"({blk.orientation_index})"
        _scr = QApplication.primaryScreen()
        _dpi_s = 72.0 / (_scr.logicalDotsPerInch() if _scr else 72.0)
        font = QFont("Arial")
        font.setWeight(QFont.Weight.Bold)
        font.setPointSizeF(max(8, int(scale * 1.4)) * _dpi_s)
        painter.setFont(font)
        fm = QFontMetricsF(font)
        id_w     = fm.horizontalAdvance(id_part)
        orient_w = fm.horizontalAdvance(orient_part)
        total_w  = id_w + orient_w
        lh  = fm.height()
        pad = 3.0
        bg_rect = QRectF(
            center.x() - total_w / 2 - pad,
            center.y() - lh / 2 + lh * 0.8 - lh - pad,
            total_w + pad * 2,
            lh + pad * 2,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 210)))
        painter.drawRoundedRect(bg_rect, 3.0, 3.0)
        ty  = center.y() - lh / 2 + lh * 0.8
        tx0 = center.x() - total_w / 2
        painter.setPen(QPen(QColor("#1e293b")))
        painter.drawText(QPointF(tx0, ty), id_part)
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(QPointF(tx0 + id_w, ty), orient_part)


# -----------------------------------------------------------------------------
# BlockPanelCanvas
# -----------------------------------------------------------------------------
class BlockPanelCanvas(QWidget):
    drag_started = pyqtSignal(object, QPoint, QPointF)

    PAD = PANEL_PAD
    W   = PANEL_W - 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocks: list[BlockItem] = []
        self.visible_layers: Optional[set] = None
        self.SCALE = PANEL_SCALE          # Can be changed externally via set_scale()
        self._block_rects: list = []      # [(y_top, y_bot, blk)]
        self._bay_area: float = 1.0       # Average bay area for area ratio calculation
        self._press_pos: Optional[QPoint] = None
        self._press_blk: Optional[BlockItem] = None
        self._press_grab_offset: QPointF = QPointF(0.0, 0.0)
        self._drag_started_flag = False
        self.setFixedWidth(self.W)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._rebuild()

    def set_scale(self, s: float):
        self.SCALE = max(_SCALE_MIN, min(_SCALE_MAX, s))
        self._rebuild()

    # Minimum height (px) sufficient to display 5 info rows (ID + 5 rows)
    _INFO_MIN_H = 110

    def _rebuild(self):
        self._block_rects.clear()
        y = self.PAD
        for blk in self.blocks:
            bb = blk.anchored_bb()
            bh = bb[3] - bb[1]
            draw_h = max(int(math.ceil(bh * self.SCALE)), self._INFO_MIN_H)
            y_bot = y + draw_h
            self._block_rects.append((y, y_bot, blk))
            y = y_bot + BLOCK_MARGIN
        self.setFixedHeight(max(y + self.PAD, 50))
        self.update()

    def rebuild(self):
        self._rebuild()

    def _draw_block_info(self, painter: QPainter, blk: BlockItem,
                         x: float, y_top: float, y_bot: float):
        """Render block info in the left info column."""
        d = blk.data
        r        = d.get("release_time", "--")
        due      = d.get("due_date", "--")
        pt       = d.get("processing_time", "--")
        wl       = d.get("workload", "--")
        prefs    = d.get("bay_preferences", [])
        shape    = d.get("shape", [])
        n_orient = len(shape)

        # Area ratio: orientation 0 base layer bounding box / bay area
        area_ratio = 0.0
        if shape:
            lyr0 = shape[0]["layers"][0] if shape[0]["layers"] else []
            if lyr0:
                x0b, y0b, x1b, y1b = _bounding_box(lyr0)
                area_ratio = (x1b - x0b) * (y1b - y0b) / max(self._bay_area, 1.0)

        # Bay preference: preferred bay index and ratio
        pref_str = "--"
        if prefs:
            best_i = max(range(len(prefs)), key=lambda i: prefs[i])
            pref_str = f"B{best_i}:{prefs[best_i]}%"

        # info background
        info_rect = QRectF(x, y_top - 2, PANEL_INFO_W - 2, y_bot - y_top + 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRect(info_rect)

        CLR_LABEL = QColor("#e3e3e3")   # slate-500
        CLR_ID    = QColor("#f1f5f9")   # slate-100
        CLR_VAL   = QColor("#e2e8f0")   # slate-200
        CLR_ACC   = QColor("#38bdf8")   # sky-400  (timing)
        CLR_GRN   = QColor("#4ade80")   # green-400 (area)
        CLR_AMB   = QColor("#fbbf24")   # amber-400 (rotation)
        CLR_PRP   = QColor("#c084fc")   # purple-400 (workload)
        CLR_ORG   = QColor("#fb923c")   # orange-400 (preference)

        # Normalize to 72 DPI (macOS baseline) -> renders at the same pixel size on Windows (96 DPI) etc.
        _screen = QApplication.primaryScreen()
        _dpi    = _screen.logicalDotsPerInch() if _screen else 72.0
        _s      = 72.0 / _dpi  # macOS:1.0  Windows 96dpi:0.75

        FONT_ID  = QFont("Arial")
        FONT_ID.setWeight(QFont.Weight.Bold)
        FONT_ID.setPointSizeF(10 * _s)
        FONT_LBL = QFont("Arial")
        FONT_LBL.setPointSizeF(9 * _s)
        FONT_VAL = QFont("Arial")
        FONT_VAL.setWeight(QFont.Weight.Bold)
        FONT_VAL.setPointSizeF(10 * _s)

        pad = 5.0
        iw  = PANEL_INFO_W - 2

        # -- Block ID header ------------------------------------------------
        fm_id = QFontMetricsF(FONT_ID)
        painter.setFont(FONT_ID)
        painter.setPen(QPen(CLR_ID))
        id_text = f"B{blk.display_id}"
        id_x = x + (iw - fm_id.horizontalAdvance(id_text)) / 2
        id_y = y_top + fm_id.ascent() + 2
        painter.drawText(QPointF(id_x, id_y), id_text)

        sep_y = id_y + fm_id.descent() + 2
        painter.setPen(QPen(QColor("#535557"), 1))
        painter.drawLine(QPointF(x + pad, sep_y), QPointF(x + iw - pad, sep_y))

        # -- Info rows: label*value left/right aligned on one line ----------
        rows = [
            ("Area", f"{area_ratio:.1%}", CLR_GRN),
            ("Rel",  str(r),             CLR_ACC),
            ("Due",  str(due),           CLR_ACC),
            ("Proc", str(pt),            CLR_VAL),
            ("WL",   str(wl),            CLR_PRP),
            ("Pref", pref_str,           CLR_ORG),
            ("Rots", str(n_orient),      CLR_AMB),
        ]

        fm_lbl = QFontMetricsF(FONT_LBL)
        fm_val = QFontMetricsF(FONT_VAL)
        row_h  = max(fm_lbl.height(), fm_val.height()) + 1
        baseline_off = max(fm_lbl.ascent(), fm_val.ascent())

        avail_h      = (y_bot - 2) - (sep_y + 2)
        total_rows_h = row_h * len(rows)
        start_y = sep_y + 2 + max(0.0, (avail_h - total_rows_h) / 2)

        for i, (lbl, val, val_clr) in enumerate(rows):
            ry = start_y + i * row_h
            if ry + row_h > y_bot + 2:
                break
            base = ry + baseline_off

            painter.setFont(FONT_LBL)
            painter.setPen(QPen(CLR_LABEL))
            painter.drawText(QPointF(x + pad, base), lbl)

            painter.setFont(FONT_VAL)
            painter.setPen(QPen(val_clr))
            val_x = x + iw - fm_val.horizontalAdvance(val) - pad
            painter.drawText(QPointF(val_x, base), val)

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f172a"))

        info_w = PANEL_INFO_W
        shape_x = float(info_w)

        for y_top, y_bot, blk in self._block_rects:
            # Shape column background (white)
            painter.fillRect(QRectF(shape_x, y_top - 2, self.W - info_w - 2, (y_bot - y_top) + 4),
                             QColor("#ffffff"))

            # Draw block shape centered inside the shape column.
            # Compute origin so the block's bounding-box centre aligns with the
            # centre of the available shape area (both horizontally and vertically).
            bb = blk.anchored_bb()          # (lx0, ly0, lx1, ly1) in local coords
            cx_local = (bb[0] + bb[2]) / 2.0
            cy_local = (bb[1] + bb[3]) / 2.0
            pad_px    = float(self.PAD)
            avail_w   = float(self.W) - shape_x - pad_px
            avail_h   = float(y_bot - y_top) - pad_px
            ox = shape_x + pad_px / 2.0 + avail_w / 2.0 - cx_local * self.SCALE
            oy = float(y_top) + pad_px / 2.0 + avail_h / 2.0 + cy_local * self.SCALE
            _draw_block_at(painter, blk, QPointF(ox, oy), self.SCALE,
                           self.visible_layers, alpha=220, flip_y=True)

            # Block info (left info column)
            self._draw_block_info(painter, blk, 0.0, y_top, y_bot)

        painter.end()

    def _blk_at_y(self, py: int) -> Optional[BlockItem]:
        for y_top, y_bot, blk in self._block_rects:
            if y_top - 2 <= py <= y_bot + 4:
                return blk
        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            blk = self._blk_at_y(int(ev.position().y()))
            if blk:
                self._press_blk = blk
                self._press_pos = ev.pos()
                self._drag_started_flag = False
                # Compute grab offset using the same origin as paintEvent drawing
                for y_top, y_bot, b in self._block_rects:
                    if b is blk:
                        bb = blk.anchored_bb()
                        cx_local = (bb[0] + bb[2]) / 2.0
                        cy_local = (bb[1] + bb[3]) / 2.0
                        shape_x  = float(PANEL_INFO_W)
                        pad_px   = float(self.PAD)
                        avail_w  = float(self.W) - shape_x - pad_px
                        avail_h  = float(y_bot - y_top) - pad_px
                        ox_draw  = shape_x + pad_px / 2.0 + avail_w / 2.0 - cx_local * self.SCALE
                        oy_draw  = float(y_top) + pad_px / 2.0 + avail_h / 2.0 + cy_local * self.SCALE
                        self._press_grab_offset = QPointF(
                            (ev.position().x() - ox_draw) / self.SCALE,
                            (oy_draw - ev.position().y()) / self.SCALE,
                        )
                        break
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_blk and (ev.buttons() & Qt.MouseButton.LeftButton):
            if (ev.pos() - self._press_pos).manhattanLength() > 6 and not self._drag_started_flag:
                self._drag_started_flag = True
                self.drag_started.emit(self._press_blk, ev.globalPosition().toPoint(),
                                       self._press_grab_offset)
        else:
            blk = self._blk_at_y(int(ev.position().y()))
            if blk is not None:
                d = blk.data
                r   = d.get("release_time", "--")
                due = d.get("due_date", "--")
                pt  = d.get("processing_time", "--")
                wl  = d.get("workload", "--")
                prefs = d.get("bay_preferences", [])
                pref_str = "  ".join(f"B{i}:{p}%" for i, p in enumerate(prefs))
                n_orient = len(d.get("shape", []))
                layers   = d.get("shape", [{}])[0].get("layers", []) if d.get("shape") else []
                n_layers = sum(1 for lyr in layers if lyr != [])
                tip = (
                    f"<b>Block {blk.display_id}</b><br>"
                    f"Release: <b>{r}</b> &nbsp; Due: <b>{due}</b> &nbsp; "
                    f"Proc: <b>{pt}</b><br>"
                    f"Workload: <b>{wl}</b><br>"
                    f"Orientations: <b>{n_orient}</b> &nbsp; "
                    f"Layers: <b>{n_layers}</b><br>"
                    f"Bay prefs: {pref_str}"
                )
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._press_blk and not self._drag_started_flag:
                self._press_blk.rotate_next()
                self._rebuild()
            self._press_blk = None
            self._press_pos = None
            self._press_grab_offset = QPointF(0.0, 0.0)
            self._drag_started_flag = False
        super().mouseReleaseEvent(ev)


# -----------------------------------------------------------------------------
# BayCanvas
# -----------------------------------------------------------------------------
class BayCanvas(QWidget):
    drag_started  = pyqtSignal(object, QPoint, QPointF)
    block_changed = pyqtSignal()   # On block state change (rotate/drop etc.)

    PAD = BAY_PAD

    def __init__(self, bay_w: float, bay_h: float, bay_idx: int,
                 bay_id: int = 0,
                 read_only: bool = False, parent=None):
        super().__init__(parent)
        self.bay_w     = bay_w
        self.bay_h     = bay_h
        self.bay_idx   = bay_idx
        self.bay_id    = bay_id if bay_id > 0 else bay_idx + 1
        self.read_only = read_only
        self.blocks: list[BlockItem] = []
        self._scale  = BAY_SCALE_DEF
        self.visible_layers: Optional[set] = None
        self._cached_collisions: list = []

        self._press_blk: Optional[BlockItem] = None
        self._press_pos: Optional[QPoint] = None
        self._press_grab_offset: QPointF = QPointF(0.0, 0.0)
        self._drag_started_flag = False
        self._cached_collision_results: list[CollisionResult] = []

        # solution viewer: set of block ids being exited (crane in progress)
        # and the id of the block just entered (-1 = none)
        # failed_op_block_id: block id whose check_entry/check_exit failed (-1 = none)
        # obstruction_block_ids: blocks that caused the check failure
        self.exiting_block_ids: set[int] = set()
        self.entering_block_id: int = -1
        self.failed_op_block_id: int = -1
        self.obstruction_block_ids: set[int] = set()

        # Cut lines -- always shown
        self._cut_x_pos: float = bay_w / 2   # Vertical cut line X coordinate
        self._cut_y_pos: float = bay_h / 2   # Horizontal cut line Y coordinate
        self._cut_dragging_axis: Optional[str] = None  # 'x' | 'y' | None

        # Cross-section cache
        self._n_layers: int = 1
        self._cached_x_segments: list = []
        self._cached_y_segments: list = []
        self._cached_x_collision_segs: list = []  # [(layer_idx, y_start, y_end)]
        self._cached_y_collision_segs: list = []  # [(layer_idx, x_start, x_end)]
        self._preview_x_segs: list = []           # Drag preview -- X cross-section
        self._preview_y_segs: list = []           # Drag preview -- Y cross-section
        self._bay_px_w: int = 0
        self._bay_px_h: int = 0

        self.setMouseTracking(True)
        self._update_size()

    def _update_size(self):
        self._bay_px_w = int(self.bay_w * self._scale + self.PAD * 2)
        self._bay_px_h = int(self.bay_h * self._scale + self.PAD * 2)
        total_w = self._bay_px_w + self._n_layers * XCUT_LAYER_W
        total_h = self._bay_px_h + self._n_layers * YCUT_LAYER_H
        self.setFixedSize(total_w, total_h)

    def set_scale(self, s: float):
        self._scale = max(1.0, s)
        self._update_size()
        self.update()

    def _to_pixel(self, bx: float, by: float) -> QPointF:
        return QPointF(self.PAD + bx * self._scale,
                       self._bay_px_h - self.PAD - by * self._scale)

    def _to_bay(self, px: float, py: float):
        return ((px - self.PAD) / self._scale,
                (self._bay_px_h - self.PAD - py) / self._scale)

    def _block_at(self, bx, by) -> Optional[BlockItem]:
        for blk in reversed(self.blocks):
            bb = blk.bounding_rect_at_pos()
            if not (bb[0] <= bx <= bb[2] and bb[1] <= by <= bb[3]):
                continue
            for layer in blk.layers_at_pos():
                if _point_in_poly(bx, by, layer):
                    return blk
        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            px, py = ev.position().x(), ev.position().y()
            if px < self._bay_px_w and py < self._bay_px_h:
                if self._near_cut_x(px, py):
                    self._cut_dragging_axis = 'x'
                elif self._near_cut_y(px, py):
                    self._cut_dragging_axis = 'y'
                elif not self.read_only:
                    bx, by = self._to_bay(px, py)
                    blk = self._block_at(bx, by)
                    if blk:
                        self._press_blk = blk
                        self._press_pos = ev.pos()
                        self._drag_started_flag = False
                        self._press_grab_offset = QPointF(bx - blk.x, by - blk.y)
                        self.blocks.remove(blk)
                        self.blocks.append(blk)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        px, py = ev.position().x(), ev.position().y()
        if self._cut_dragging_axis and (ev.buttons() & Qt.MouseButton.LeftButton):
            bx, by = self._to_bay(px, py)
            if self._cut_dragging_axis == 'x':
                self._cut_x_pos = max(0.0, min(bx, float(self.bay_w)))
            else:
                self._cut_y_pos = max(0.0, min(by, float(self.bay_h)))
            self._refresh_segments()
            super().mouseMoveEvent(ev)
            return
        if not self.read_only:
            if self._press_blk and (ev.buttons() & Qt.MouseButton.LeftButton):
                if (ev.pos() - self._press_pos).manhattanLength() > 6 and not self._drag_started_flag:
                    self._drag_started_flag = True
                    self.drag_started.emit(self._press_blk, ev.globalPosition().toPoint(),
                                           self._press_grab_offset)
        if not (ev.buttons() & Qt.MouseButton.LeftButton):
            if px < self._bay_px_w and py < self._bay_px_h:
                if self._near_cut_x(px, py):
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                elif self._near_cut_y(px, py):
                    self.setCursor(Qt.CursorShape.SplitVCursor)
                else:
                    self.unsetCursor()
                self._update_collision_tooltip(ev.position(), ev.globalPosition().toPoint())
            else:
                self.unsetCursor()
        super().mouseMoveEvent(ev)

    def _update_collision_tooltip(self, local_pos: QPointF, global_pos: QPoint):
        """Show collision tooltip or block info tooltip based on mouse position."""
        bx, by = self._to_bay(local_pos.x(), local_pos.y())
        pt = ShapelyPoint(bx, by)

        # 1) Check collision area first
        hit: list[CollisionResult] = []
        for r in self._cached_collision_results:
            try:
                if not r.intersection.is_empty and r.intersection.contains(pt):
                    hit.append(r)
            except Exception:
                pass
        if hit:
            lines = ["<b>Collision detected</b>"]
            for r in hit:
                lines.append(
                    f"Block <b>{r.block_a.block_id}</b> <-> "
                    f"Block <b>{r.block_b.block_id}</b> &nbsp;"
                    f"Layer <b>L{r.layer_index}</b> &nbsp;"
                    f"(area {r.area:.2f})"
                )
            QToolTip.showText(global_pos, "<br>".join(lines), self)
            return

        # 2) Non-collision block area -> show block position / orientation info
        blk = self._block_at(bx, by)
        if blk is not None:
            text = (
                f"<b>Block {blk.display_id}</b><br>"
                f"Position: ({blk.x:.1f}, {blk.y:.1f})<br>"
                f"Orientation index: {blk.orientation_index}"
            )
            QToolTip.showText(global_pos, text, self)
            return

        QToolTip.hideText()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._cut_dragging_axis:
                self._cut_dragging_axis = None
                super().mouseReleaseEvent(ev)
                return
            if self._press_blk and not self._drag_started_flag:
                self._press_blk.rotate_next()
                br = self._press_blk.bounding_rect_at_pos()
                bw = int(math.ceil(br[2] - br[0]))
                bh = int(math.ceil(br[3] - br[1]))
                # Keep position; only clamp minimally when going outside bay boundary
                cx = int(self._press_blk.x)
                cy = int(self._press_blk.y)
                self._press_blk.x = min(cx, max(0, int(self.bay_w) - bw))
                self._press_blk.y = min(cy, max(0, int(self.bay_h) - bh))
                self.mark_dirty_for_block(self._press_blk.id)
                self.block_changed.emit()
            self._press_blk = None
            self._press_pos = None
            self._press_grab_offset = QPointF(0.0, 0.0)
            self._drag_started_flag = False
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f0f4f8"))

        tl = self._to_pixel(0, self.bay_h)
        br_pt = self._to_pixel(self.bay_w, 0)
        bay_rect = QRectF(tl, br_pt)
        painter.fillRect(bay_rect, Qt.GlobalColor.white)
        painter.setPen(QPen(QColor("#334155"), 2))
        painter.drawRect(bay_rect)

        grid_pen = QPen(QColor("#e2e8f0"), 0.5)
        painter.setPen(grid_pen)
        for gx in range(0, int(self.bay_w) + 1, max(1, int(self.bay_w) // 20)):
            painter.drawLine(self._to_pixel(gx, 0), self._to_pixel(gx, self.bay_h))
        for gy in range(0, int(self.bay_h) + 1, max(1, int(self.bay_h) // 10)):
            painter.drawLine(self._to_pixel(0, gy), self._to_pixel(self.bay_w, gy))

        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(QPointF(self.PAD + 4, self.PAD - 3), f"Bay {self.bay_id}")

        # -- Display bay dimensions in center (two lines) --------------------
        _scr = QApplication.primaryScreen()
        _dpi_s = 72.0 / (_scr.logicalDotsPerInch() if _scr else 72.0)
        size_font = QFont("Arial")
        size_font.setPointSizeF(40 * _dpi_s)
        fm_size = QFontMetricsF(size_font)
        line1 = f"Bay {self.bay_id}"
        line2 = f"{int(self.bay_w)} x {int(self.bay_h)}"
        line_gap = fm_size.height() * 0.1
        total_h = fm_size.height() * 2 + line_gap
        cy1 = bay_rect.center().y() - total_h / 2 + fm_size.ascent()
        cy2 = cy1 + fm_size.height() + line_gap
        painter.setFont(size_font)
        painter.setPen(QPen(QColor(180, 180, 180, 90)))
        painter.drawText(QPointF(bay_rect.center().x() - fm_size.horizontalAdvance(line1) / 2, cy1), line1)
        painter.drawText(QPointF(bay_rect.center().x() - fm_size.horizontalAdvance(line2) / 2, cy2), line2)

        # -- Block rendering: draw in layer order across all blocks ------------
        # Step 1: pre-collect pixel transform data for each block
        blk_entries = []  # (blk, origin, vis_indices, all_layers)
        for blk in self.blocks:
            origin = self._to_pixel(blk.x, blk.y)
            all_layers = blk.anchored_layers()
            vis_indices = [i for i in range(len(all_layers))
                           if self.visible_layers is None or i in self.visible_layers]
            if vis_indices:
                blk_entries.append((blk, origin, vis_indices, all_layers))

        def _to_qpoly(origin, all_layers_li, scale):
            poly = QPolygonF()
            for vx, vy in all_layers_li:
                poly.append(QPointF(origin.x() + vx * scale,
                                    origin.y() - vy * scale))
            return poly

        # Step 2: collect layer indices across all blocks in ascending order and fill
        # Blocks being EXITed are drawn semi-transparent; fully ENTERed blocks are drawn normally
        all_li = sorted({li for _, _, vis, _ in blk_entries for li in vis})
        for li in all_li:
            gray = _layer_gray(li)
            painter.setPen(Qt.PenStyle.NoPen)
            for blk, origin, vis_indices, all_layers in blk_entries:
                if li not in vis_indices or not all_layers[li]:
                    continue
                if blk.id in self.exiting_block_ids:
                    # Exiting: semi-transparent fill
                    painter.setBrush(QBrush(QColor(gray, gray, gray, 80)))
                else:
                    painter.setBrush(QBrush(QColor(gray, gray, gray, 230)))
                painter.drawPolygon(_to_qpoly(origin, all_layers[li], self._scale))

        # Step 3: per-block outlines -- occluded parts (faint) vs. exposed parts (normal)
        # When drawing a block's layers, areas occluded by higher layers of other blocks
        # are drawn with a faint pen; the rest use a normal pen.
        # Exiting block: thick orange dashed line. Entered block: thick green solid line.
        pen_dash_normal = QPen(QColor("#555555"), 1.0)
        pen_dash_normal.setStyle(Qt.PenStyle.DashLine)
        pen_dash_normal.setDashPattern([4, 3])
        pen_dash_faded = QPen(QColor(160, 160, 160, 80), 0.8)
        pen_dash_faded.setStyle(Qt.PenStyle.DashLine)
        pen_dash_faded.setDashPattern([4, 3])
        pen_solid_normal = QPen(QColor("#222222"), 2.0)
        pen_solid_normal.setStyle(Qt.PenStyle.SolidLine)
        pen_solid_faded = QPen(QColor(160, 160, 160, 80), 1.2)
        pen_solid_faded.setStyle(Qt.PenStyle.SolidLine)
        # Exiting: thick orange dashed line (crane operation in progress)
        pen_exit_dash = QPen(QColor(220, 120, 0, 200), 1.2)
        pen_exit_dash.setStyle(Qt.PenStyle.DashLine)
        pen_exit_dash.setDashPattern([4, 3])
        pen_exit_solid = QPen(QColor(220, 120, 0, 220), 2.5)
        pen_exit_solid.setStyle(Qt.PenStyle.DashLine)
        pen_exit_solid.setDashPattern([6, 4])
        # Entered: yellow solid line highlight
        pen_entry_solid = QPen(QColor(234, 179, 8, 255), 3.0)
        pen_entry_solid.setStyle(Qt.PenStyle.SolidLine)
        # check_entry/check_exit failed: red thick solid line
        pen_failed_solid = QPen(QColor(220, 30, 30, 255), 3.0)
        pen_failed_solid.setStyle(Qt.PenStyle.SolidLine)
        pen_failed_dash = QPen(QColor(220, 30, 30, 200), 1.5)
        pen_failed_dash.setStyle(Qt.PenStyle.DashLine)
        pen_failed_dash.setDashPattern([4, 3])
        # Obstruction block: purple thick solid line
        pen_obstruct_solid = QPen(QColor(147, 51, 234, 255), 3.0)
        pen_obstruct_solid.setStyle(Qt.PenStyle.SolidLine)
        pen_obstruct_dash = QPen(QColor(147, 51, 234, 200), 1.5)
        pen_obstruct_dash.setStyle(Qt.PenStyle.DashLine)
        pen_obstruct_dash.setDashPattern([4, 3])

        def _covering_path_for(blk_id, li_threshold):
            """Return QPainterPath of pixel areas from other blocks (excluding blk_id)
            with layer index > li_threshold. This path represents the 'occluding area'."""
            cp = QPainterPath()
            for other_blk, other_origin, other_vis, other_layers in blk_entries:
                if other_blk.id == blk_id:
                    continue
                for j in other_vis:
                    if j > li_threshold and other_layers[j]:
                        p = QPainterPath()
                        p.addPolygon(_to_qpoly(other_origin, other_layers[j], self._scale))
                        cp = cp.united(p)
            return cp

        painter.setBrush(Qt.BrushStyle.NoBrush)
        for blk, origin, vis_indices, all_layers in blk_entries:
            is_exiting    = blk.id in self.exiting_block_ids
            is_entering   = blk.id == self.entering_block_id
            is_failed     = blk.id == self.failed_op_block_id
            is_obstruct   = blk.id in self.obstruction_block_ids

            # -- Per-layer boundary dashed lines -----------------------------
            for li in vis_indices:
                if not all_layers[li]:
                    continue
                poly = _to_qpoly(origin, all_layers[li], self._scale)

                if is_failed:
                    painter.setClipping(False)
                    painter.setPen(pen_failed_dash)
                    painter.drawPolygon(poly)
                    continue

                if is_obstruct:
                    painter.setClipping(False)
                    painter.setPen(pen_obstruct_dash)
                    painter.drawPolygon(poly)
                    continue

                if is_exiting:
                    painter.setClipping(False)
                    painter.setPen(pen_exit_dash)
                    painter.drawPolygon(poly)
                    continue

                covering = _covering_path_for(blk.id, li)
                if covering.isEmpty():
                    painter.setClipping(False)
                    painter.setPen(pen_dash_normal)
                    painter.drawPolygon(poly)
                else:
                    painter.setClipPath(covering)
                    painter.setPen(pen_dash_faded)
                    painter.drawPolygon(poly)
                    full = QPainterPath()
                    full.addRect(QRectF(self.rect()))
                    painter.setClipPath(full.subtracted(covering))
                    painter.setPen(pen_dash_normal)
                    painter.drawPolygon(poly)
                    painter.setClipping(False)

            # -- Solid outer union --------------------------------------------
            s_polys = [
                ShapelyPolygon(all_layers[li])
                for li in vis_indices
                if all_layers[li] and len(all_layers[li]) >= 3
            ]
            if not s_polys:
                continue
            outer = unary_union(s_polys)
            geoms = list(outer.geoms) if hasattr(outer, 'geoms') else [outer]
            outer_polys = []
            for geom in geoms:
                try:
                    coords = list(geom.exterior.coords)
                except AttributeError:
                    continue
                outer_polys.append(_to_qpoly(origin, coords, self._scale))

            if is_failed:
                # check_entry/check_exit failed: red thick solid outer line
                painter.setClipping(False)
                painter.setPen(pen_failed_solid)
                for op in outer_polys:
                    painter.drawPolygon(op)
                continue

            if is_obstruct:
                # Obstruction block: purple thick solid outer line
                painter.setClipping(False)
                painter.setPen(pen_obstruct_solid)
                for op in outer_polys:
                    painter.drawPolygon(op)
                continue

            if is_exiting:
                # Exiting: thick orange dashed outer line
                painter.setClipping(False)
                painter.setPen(pen_exit_solid)
                for op in outer_polys:
                    painter.drawPolygon(op)
                continue

            if is_entering:
                # Entered: green solid highlight outer line
                painter.setClipping(False)
                painter.setPen(pen_entry_solid)
                for op in outer_polys:
                    painter.drawPolygon(op)
                continue

            # Normal block: faint rendering of occluded area via clip separation
            top_li = vis_indices[-1]
            covering_outer = _covering_path_for(blk.id, top_li - 1)
            for op in outer_polys:
                if covering_outer.isEmpty():
                    painter.setClipping(False)
                    painter.setPen(pen_solid_normal)
                    painter.drawPolygon(op)
                else:
                    painter.setClipPath(covering_outer)
                    painter.setPen(pen_solid_faded)
                    painter.drawPolygon(op)
                    full = QPainterPath()
                    full.addRect(QRectF(self.rect()))
                    painter.setClipPath(full.subtracted(covering_outer))
                    painter.setPen(pen_solid_normal)
                    painter.drawPolygon(op)
                    painter.setClipping(False)

        # Step 4: diagonal hatch overlay for exiting blocks
        painter.setClipping(False)
        for blk, origin, vis_indices, all_layers in blk_entries:
            if blk.id not in self.exiting_block_ids:
                continue
            s_polys = [
                ShapelyPolygon(all_layers[li])
                for li in vis_indices
                if all_layers[li] and len(all_layers[li]) >= 3
            ]
            if not s_polys:
                continue
            outer = unary_union(s_polys)
            geoms = list(outer.geoms) if hasattr(outer, 'geoms') else [outer]
            for geom in geoms:
                try:
                    coords = list(geom.exterior.coords)
                except AttributeError:
                    continue
                qpoly = _to_qpoly(origin, coords, self._scale)
                path = QPainterPath()
                path.addPolygon(qpoly)
                bb = path.boundingRect()
                painter.save()
                painter.setClipPath(path)
                hatch_pen = QPen(QColor(220, 120, 0, 130), 1.0)
                painter.setPen(hatch_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                step = 8
                x0, y0 = bb.left(), bb.top()
                total = bb.width() + bb.height()
                i = 0
                while i < total:
                    painter.drawLine(
                        QPointF(x0 + i - bb.height(), y0 + bb.height()),
                        QPointF(x0 + i, y0),
                    )
                    i += step
                painter.restore()

        # -- Highlight area outside bay boundary ----------------------------
        bay_shapely = ShapelyPolygon([
            (0, 0), (self.bay_w, 0), (self.bay_w, self.bay_h), (0, self.bay_h)
        ])
        painter.setClipping(False)
        for blk, origin, vis_indices, all_layers in blk_entries:
            world_polys = []
            for li in vis_indices:
                if all_layers[li] and len(all_layers[li]) >= 3:
                    world_verts = [(blk.x + vx, blk.y + vy) for vx, vy in all_layers[li]]
                    try:
                        world_polys.append(ShapelyPolygon(world_verts))
                    except Exception:
                        pass
            if not world_polys:
                continue
            world_union = unary_union(world_polys)
            outside = world_union.difference(bay_shapely)
            if outside.is_empty:
                continue
            geoms = list(outside.geoms) if hasattr(outside, 'geoms') else [outside]
            for geom in geoms:
                try:
                    coords = list(geom.exterior.coords)
                except AttributeError:
                    continue
                qpoly = QPolygonF()
                for wx, wy in coords:
                    qpoly.append(self._to_pixel(wx, wy))
                painter.setBrush(QBrush(QColor(220, 30, 30, 110)))
                painter.setPen(QPen(QColor(180, 0, 0, 220), 2.0))
                painter.drawPolygon(qpoly)
                out_path = QPainterPath()
                out_path.addPolygon(qpoly)
                bb_out = out_path.boundingRect()
                painter.save()
                painter.setClipPath(out_path)
                painter.setPen(QPen(QColor(180, 0, 0, 130), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                step = 5
                x0, y0 = bb_out.left(), bb_out.top()
                total = bb_out.width() + bb_out.height()
                i = 0
                while i < total:
                    painter.drawLine(
                        QPointF(x0 + i - bb_out.height(), y0 + bb_out.height()),
                        QPointF(x0 + i, y0),
                    )
                    i += step
                painter.restore()

        # Step 5: labels (always on top)
        for blk, origin, vis_indices, all_layers in blk_entries:
            vis_verts = [v for li in vis_indices for v in (all_layers[li] or [])]
            if not vis_verts:
                continue
            cx = sum(v[0] for v in vis_verts) / len(vis_verts)
            cy = sum(v[1] for v in vis_verts) / len(vis_verts)
            center = QPointF(origin.x() + cx * self._scale,
                             origin.y() - cy * self._scale)
            is_exiting    = blk.id in self.exiting_block_ids
            is_entering   = blk.id == self.entering_block_id
            is_failed     = blk.id == self.failed_op_block_id
            is_obstruct   = blk.id in self.obstruction_block_ids
            orient_tag = ""
            if is_failed and is_exiting:
                label = f"B{blk.display_id}\n[EXIT ✗]"
                label_color = QColor(180, 0, 0)
                bg_color    = QColor(255, 180, 180, 230)
            elif is_failed and is_entering:
                label = f"B{blk.display_id}\n[ENTRY ✗]"
                label_color = QColor(180, 0, 0)
                bg_color    = QColor(255, 180, 180, 230)
            elif is_failed:
                label = f"B{blk.display_id}\n[✗]"
                label_color = QColor(180, 0, 0)
                bg_color    = QColor(255, 180, 180, 230)
            elif is_obstruct:
                label = f"B{blk.display_id}\n[Obstruct]"
                label_color = QColor(100, 20, 160)
                bg_color    = QColor(230, 200, 255, 230)
            elif is_exiting:
                label = f"B{blk.display_id}\n[EXIT]"
                label_color = QColor(180, 80, 0)
                bg_color    = QColor(255, 220, 180, 210)
            elif is_entering:
                label = f"B{blk.display_id}\n[ENTRY]"
                label_color = QColor(120, 80, 0)
                bg_color    = QColor(255, 245, 150, 210)
            else:
                label       = f"B{blk.display_id}"
                orient_tag  = f"({blk.orientation_index})"
                label_color = QColor("#1e293b")
                bg_color    = QColor(255, 255, 255, 190)
            _scr = QApplication.primaryScreen()
            _dpi_s = 72.0 / (_scr.logicalDotsPerInch() if _scr else 72.0)
            font = QFont("Arial")
            font.setWeight(QFont.Weight.Bold)
            font.setPointSizeF(max(7, int(self._scale * 1.4)) * _dpi_s)
            painter.setFont(font)
            fm = QFontMetricsF(font)
            lines = label.split("\n")
            lh = fm.height()
            th = lh * len(lines)
            orient_w = fm.horizontalAdvance(orient_tag) if orient_tag else 0.0
            max_lw = max(fm.horizontalAdvance(ln) for ln in lines)
            if orient_tag:   # single-line normal case: append orient width
                max_lw += orient_w
            pad = 3.0
            bg_rect = QRectF(
                center.x() - max_lw / 2 - pad,
                center.y() - th / 2 + lh * 0.8 - lh - pad,
                max_lw + pad * 2,
                th + pad * 2,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(bg_rect, 3.0, 3.0)
            for i, line in enumerate(lines):
                lw = fm.horizontalAdvance(line)
                if orient_tag:
                    tx = center.x() - max_lw / 2
                else:
                    tx = center.x() - lw / 2
                ty = center.y() - th / 2 + i * lh + lh * 0.8
                painter.setPen(QPen(label_color))
                painter.drawText(QPointF(tx, ty), line)
                if orient_tag:
                    painter.setPen(QPen(QColor("#f59e0b")))
                    painter.drawText(QPointF(tx + lw, ty), orient_tag)

        painter.setBrush(QBrush(QColor(255, 50, 50, 70)))
        painter.setPen(QPen(QColor(200, 0, 0, 160), 1.5))
        for inter_poly in self._cached_collisions:
            if inter_poly is None or inter_poly.is_empty:
                continue
            geoms = list(inter_poly.geoms) if hasattr(inter_poly, 'geoms') else [inter_poly]
            for geom in geoms:
                try:
                    coords = list(geom.exterior.coords)
                except AttributeError:
                    continue
                poly = QPolygonF()
                for x, y in coords:
                    poly.append(self._to_pixel(x, y))
                painter.drawPolygon(poly)

        # -- Cut lines within bay (X vertical, Y horizontal) -----------------
        painter.setClipping(False)
        pen_cut = QPen(QColor(220, 30, 30, 200), 1.5)
        pen_cut.setStyle(Qt.PenStyle.DashLine)
        pen_cut.setDashPattern([8, 4])
        painter.setPen(pen_cut)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # X cut line (vertical)
        cpx = self.PAD + self._cut_x_pos * self._scale
        painter.drawLine(QPointF(cpx, self.PAD),
                         QPointF(cpx, self._bay_px_h - self.PAD))
        painter.setBrush(QBrush(QColor(220, 30, 30, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        hx = QPainterPath()
        hx.moveTo(cpx, self.PAD)
        hx.lineTo(cpx - 5, self.PAD - 8)
        hx.lineTo(cpx + 5, self.PAD - 8)
        hx.closeSubpath()
        painter.drawPath(hx)
        # Y cut line (horizontal)
        cpy = self._bay_px_h - self.PAD - self._cut_y_pos * self._scale
        painter.setPen(pen_cut)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(self.PAD, cpy),
                         QPointF(self._bay_px_w - self.PAD, cpy))
        painter.setBrush(QBrush(QColor(220, 30, 30, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        hy = QPainterPath()
        hy.moveTo(self.PAD, cpy)
        hy.lineTo(self.PAD - 8, cpy - 5)
        hy.lineTo(self.PAD - 8, cpy + 5)
        hy.closeSubpath()
        painter.drawPath(hy)

        # -- X cross-section panel (right, vertical) -------------------------
        xp0 = self._bay_px_w
        painter.fillRect(QRectF(xp0, 0,
                                self._n_layers * XCUT_LAYER_W, self._bay_px_h),
                         QColor("#f8fafc"))
        # Divider between bay and panel
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawLine(QPointF(xp0, self.PAD),
                         QPointF(xp0, self._bay_px_h - self.PAD))
        for li in range(self._n_layers):
            col_x = xp0 + li * XCUT_LAYER_W
            gray = _layer_gray(li)
            painter.fillRect(
                QRectF(col_x, self.PAD, XCUT_LAYER_W, self._bay_px_h - 2 * self.PAD),
                QColor(gray, gray, gray, 20))
            painter.setPen(QPen(QColor("#e2e8f0"), 0.5))
            painter.drawLine(QPointF(col_x, self.PAD),
                             QPointF(col_x, self._bay_px_h - self.PAD))
            painter.setPen(QPen(QColor("#64748b")))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(QRectF(col_x, 2, XCUT_LAYER_W, self.PAD - 4),
                             Qt.AlignmentFlag.AlignCenter, f"L{li}")
        _x_blk_rects: dict = {}
        for blk_id, li, y_start, y_end in self._cached_x_segments:
            if li >= self._n_layers:
                continue
            col_x = xp0 + li * XCUT_LAYER_W
            py_top = self._bay_px_h - self.PAD - y_end * self._scale
            py_bot = self._bay_px_h - self.PAD - y_start * self._scale
            seg_h = max(py_bot - py_top, 1.0)
            gray = _layer_gray(li)
            rect = QRectF(col_x, py_top, XCUT_LAYER_W, seg_h)
            painter.fillRect(rect, QColor(gray, gray, gray, 210))
            _draw_hatch(painter, rect)
            _x_blk_rects.setdefault(blk_id, []).append(rect)
        # Per-block connected outline -- adjacent layer boundaries removed
        painter.setPen(QPen(QColor("#222222"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for rects in _x_blk_rects.values():
            path = QPainterPath()
            for r in rects:
                path.addRect(r)
            painter.drawPath(path.simplified())
        # X cross-section collision overlay
        for li, y_start, y_end in self._cached_x_collision_segs:
            if li >= self._n_layers:
                continue
            col_x = xp0 + li * XCUT_LAYER_W
            py_top = self._bay_px_h - self.PAD - y_end * self._scale
            py_bot = self._bay_px_h - self.PAD - y_start * self._scale
            seg_h = max(py_bot - py_top, 1.0)
            painter.fillRect(QRectF(col_x, py_top, XCUT_LAYER_W, seg_h),
                             QColor(255, 50, 50, 120))
            painter.setPen(QPen(QColor(200, 0, 0, 200), 0.7))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(col_x, py_top, XCUT_LAYER_W, seg_h))
        # X cross-section drag preview
        _pen_prev = QPen(QColor(37, 99, 235, 220), 1.5)
        _pen_prev.setStyle(Qt.PenStyle.DashLine)
        _pen_prev.setDashPattern([4, 3])
        for blk_id, li, y_start, y_end in self._preview_x_segs:
            if li >= self._n_layers:
                continue
            col_x = xp0 + li * XCUT_LAYER_W
            py_top = self._bay_px_h - self.PAD - y_end * self._scale
            py_bot = self._bay_px_h - self.PAD - y_start * self._scale
            seg_h = max(py_bot - py_top, 1.0)
            painter.fillRect(QRectF(col_x, py_top, XCUT_LAYER_W, seg_h),
                             QColor(37, 99, 235, 55))
            painter.setPen(_pen_prev)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(col_x, py_top, XCUT_LAYER_W, seg_h))
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(xp0, self.PAD,
                                self._n_layers * XCUT_LAYER_W,
                                self._bay_px_h - 2 * self.PAD))

        # -- Y cross-section panel (bottom, horizontal) -----------------------
        yp0 = self._bay_px_h
        painter.fillRect(QRectF(0, yp0,
                                self._bay_px_w, self._n_layers * YCUT_LAYER_H),
                         QColor("#f8fafc"))
        # Divider between bay and panel
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawLine(QPointF(self.PAD, yp0),
                         QPointF(self._bay_px_w - self.PAD, yp0))
        for li in range(self._n_layers):
            row_y = yp0 + (self._n_layers - 1 - li) * YCUT_LAYER_H
            gray = _layer_gray(li)
            painter.fillRect(
                QRectF(self.PAD, row_y, self._bay_px_w - 2 * self.PAD, YCUT_LAYER_H),
                QColor(gray, gray, gray, 20))
            painter.setPen(QPen(QColor("#e2e8f0"), 0.5))
            painter.drawLine(QPointF(self.PAD, row_y),
                             QPointF(self._bay_px_w - self.PAD, row_y))
            painter.setPen(QPen(QColor("#64748b")))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(QRectF(0, row_y, self.PAD - 2, YCUT_LAYER_H),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"L{li}")
        _y_blk_rects: dict = {}
        for blk_id, li, x_start, x_end in self._cached_y_segments:
            if li >= self._n_layers:
                continue
            row_y = yp0 + (self._n_layers - 1 - li) * YCUT_LAYER_H
            px_left = self.PAD + x_start * self._scale
            px_right = self.PAD + x_end * self._scale
            seg_w = max(px_right - px_left, 1.0)
            gray = _layer_gray(li)
            rect = QRectF(px_left, row_y, seg_w, YCUT_LAYER_H)
            painter.fillRect(rect, QColor(gray, gray, gray, 210))
            _draw_hatch(painter, rect)
            _y_blk_rects.setdefault(blk_id, []).append(rect)
        # Per-block connected outline -- adjacent layer boundaries removed
        painter.setPen(QPen(QColor("#222222"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for rects in _y_blk_rects.values():
            path = QPainterPath()
            for r in rects:
                path.addRect(r)
            painter.drawPath(path.simplified())
        # Y cross-section collision overlay
        for li, x_start, x_end in self._cached_y_collision_segs:
            if li >= self._n_layers:
                continue
            row_y = yp0 + (self._n_layers - 1 - li) * YCUT_LAYER_H
            px_left = self.PAD + x_start * self._scale
            px_right = self.PAD + x_end * self._scale
            seg_w = max(px_right - px_left, 1.0)
            painter.fillRect(QRectF(px_left, row_y, seg_w, YCUT_LAYER_H),
                             QColor(255, 50, 50, 120))
            painter.setPen(QPen(QColor(200, 0, 0, 200), 0.7))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(px_left, row_y, seg_w, YCUT_LAYER_H))
        # Y cross-section drag preview
        _pen_prev = QPen(QColor(37, 99, 235, 220), 1.5)
        _pen_prev.setStyle(Qt.PenStyle.DashLine)
        _pen_prev.setDashPattern([4, 3])
        for blk_id, li, x_start, x_end in self._preview_y_segs:
            if li >= self._n_layers:
                continue
            row_y = yp0 + (self._n_layers - 1 - li) * YCUT_LAYER_H
            px_left = self.PAD + x_start * self._scale
            px_right = self.PAD + x_end * self._scale
            seg_w = max(px_right - px_left, 1.0)
            painter.fillRect(QRectF(px_left, row_y, seg_w, YCUT_LAYER_H),
                             QColor(37, 99, 235, 55))
            painter.setPen(_pen_prev)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(px_left, row_y, seg_w, YCUT_LAYER_H))
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(self.PAD, yp0,
                                self._bay_px_w - 2 * self.PAD,
                                self._n_layers * YCUT_LAYER_H))

        painter.end()

    def mark_dirty(self):
        """Full recalculation -- used for layer toggle, bulk block changes, etc."""
        bay = UtilsBay(width=int(self.bay_w), height=int(self.bay_h))
        # Blocks being EXITed are in crane-lifting state and excluded from collision checks
        util_blocks = [
            UtilsBlock(block_id=b.id, block_data=b.data,
                       x=b.x, y=b.y, orient_idx=b.orient_idx)
            for b in self.blocks
            if b.id not in self.exiting_block_ids
        ]
        results = check_collisions(bay, util_blocks, layer_indices=self.visible_layers)
        self._cached_collision_results = results
        self._cached_collisions = [r.intersection for r in results]
        self._refresh_segments()

    def mark_after_removal(self, removed_blk_id: int):
        """Called after a block is removed from the bay -- O(1) without Shapely."""
        self._cached_collision_results = [
            r for r in self._cached_collision_results
            if r.block_a.block_id != removed_blk_id
            and r.block_b.block_id != removed_blk_id
        ]
        self._cached_collisions = [r.intersection for r in self._cached_collision_results]
        self._refresh_segments()

    def mark_dirty_for_block(self, changed_blk_id: int):
        """Called after a block's position/rotation changes -- recomputes only pairs involving this block O(N)."""
        # 1) Remove entries involving this block from previous results
        self._cached_collision_results = [
            r for r in self._cached_collision_results
            if r.block_a.block_id != changed_blk_id
            and r.block_b.block_id != changed_blk_id
        ]
        # 2) Re-check each pair between the changed block and all others (N-1 pairs)
        changed_item = next((b for b in self.blocks if b.id == changed_blk_id), None)
        if changed_item is not None:
            bay = UtilsBay(width=int(self.bay_w), height=int(self.bay_h))
            changed_util = UtilsBlock(
                block_id=changed_item.id, block_data=changed_item.data,
                x=changed_item.x, y=changed_item.y, orient_idx=changed_item.orient_idx,
            )
            for other in self.blocks:
                if other.id == changed_blk_id:
                    continue
                other_util = UtilsBlock(
                    block_id=other.id, block_data=other.data,
                    x=other.x, y=other.y, orient_idx=other.orient_idx,
                )
                pair_results = check_collisions(
                    bay, [changed_util, other_util],
                    layer_indices=self.visible_layers,
                )
                self._cached_collision_results.extend(pair_results)
        self._cached_collisions = [r.intersection for r in self._cached_collision_results]
        self._refresh_segments()

    def set_drag_preview(self, blk: "BlockItem", nx: float, ny: float):
        """Preview the cross-section at the drop position while dragging a block into this bay."""
        import copy as _copy
        ghost = _copy.copy(blk)
        bb = ghost.anchored_bb()
        ghost.x = max(float(-bb[0]), min(nx, float(self.bay_w) - bb[2]))
        ghost.y = max(float(-bb[1]), min(ny, float(self.bay_h) - bb[3]))
        self._preview_x_segs = _compute_cross_section(
            [ghost], self.bay_w, self.bay_h, 'x', self._cut_x_pos, self.visible_layers)
        self._preview_y_segs = _compute_cross_section(
            [ghost], self.bay_w, self.bay_h, 'y', self._cut_y_pos, self.visible_layers)
        self.update()

    def clear_drag_preview(self):
        """Clear drag preview."""
        if self._preview_x_segs or self._preview_y_segs:
            self._preview_x_segs = []
            self._preview_y_segs = []
            self.update()

    def drop_block(self, blk: BlockItem, global_pos: QPoint,
                   grab_offset: QPointF = None):
        local = self.mapFromGlobal(global_pos)
        bx, by = self._to_bay(local.x(), local.y())
        bb = blk.anchored_bb()
        bw = bb[2] - bb[0]
        bh = bb[3] - bb[1]
        if grab_offset is not None:
            # Position so that the relative click offset lands exactly at the drop position
            nx = bx - grab_offset.x()
            ny = by - grab_offset.y()
        else:
            nx = bx - bw / 2
            ny = by - bh / 2
        # Snap to integer position then clamp
        blk.x = float(max(int(math.ceil(-bb[0])), min(int(round(nx)), int(self.bay_w) - max(1, int(math.ceil(bb[2]))))))
        blk.y = float(max(int(math.ceil(-bb[1])), min(int(round(ny)), int(self.bay_h) - max(1, int(math.ceil(bb[3]))))))
        if blk not in self.blocks:
            self.blocks.append(blk)
        self.mark_dirty_for_block(blk.id)
        self.block_changed.emit()

    # -- Cut line helpers ----------------------------------------------------
    def _near_cut_x(self, px: float, py: float) -> bool:
        """Return True if pixel (px, py) is near the X cut line (vertical) within ±6px."""
        cpx = self.PAD + self._cut_x_pos * self._scale
        return abs(px - cpx) <= 6

    def _near_cut_y(self, px: float, py: float) -> bool:
        """Return True if pixel (px, py) is near the Y cut line (horizontal) within ±6px."""
        cpy = self._bay_px_h - self.PAD - self._cut_y_pos * self._scale
        return abs(py - cpy) <= 6

    def _refresh_segments(self):
        """Update cross-section data on block change or cut line move."""
        if self.blocks:
            n = max((len(b.anchored_layers()) for b in self.blocks), default=1)
            if self.visible_layers:
                n = max(n, max(self.visible_layers) + 1)
            self._n_layers = max(n, 1)
        else:
            self._n_layers = 1
        self._cached_x_segments = _compute_cross_section(
            self.blocks, self.bay_w, self.bay_h, 'x', self._cut_x_pos, self.visible_layers)
        self._cached_y_segments = _compute_cross_section(
            self.blocks, self.bay_w, self.bay_h, 'y', self._cut_y_pos, self.visible_layers)
        self._cached_x_collision_segs = _compute_collision_cross_section(
            self._cached_collision_results, self.bay_w, self.bay_h, 'x', self._cut_x_pos)
        self._cached_y_collision_segs = _compute_collision_cross_section(
            self._cached_collision_results, self.bay_w, self.bay_h, 'y', self._cut_y_pos)
        self._update_size()
        self.update()


# -----------------------------------------------------------------------------
# DragOverlay
# -----------------------------------------------------------------------------
class DragOverlay(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent.rect())
        self.hide()
        self._blk: Optional[BlockItem] = None
        self._pos: QPoint = QPoint()
        self._scale = PANEL_SCALE
        self._visible_layers: Optional[set] = None
        self._grab_offset: QPointF = QPointF(0.0, 0.0)

    def start(self, blk, pos, scale, visible_layers, grab_offset: QPointF = None):
        self._blk = blk
        self._pos = pos
        self._scale = scale
        self._visible_layers = visible_layers
        self._grab_offset = grab_offset if grab_offset is not None else QPointF(0.0, 0.0)
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.update()

    def move_to(self, pos: QPoint):
        self._pos = pos
        self.update()

    def stop(self):
        self._blk = None
        self.hide()
        self.update()

    def paintEvent(self, ev):
        if self._blk is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Adjust origin by click offset so the mouse stays fixed at the click position
        ox = self._pos.x() - self._grab_offset.x() * self._scale
        oy = self._pos.y() + self._grab_offset.y() * self._scale
        _draw_block_at(painter, self._blk, QPointF(ox, oy), self._scale,
                       self._visible_layers, alpha=140, flip_y=True)
        painter.end()


# -----------------------------------------------------------------------------
# Gantt chart (block timing visualization)
# -----------------------------------------------------------------------------
_G_BAY_HUES    = [210, 35, 130, 280, 10, 170, 310, 60]
_G_ROW_H       = 26
_G_LABEL_W     = 80
_G_AXIS_H      = 24
_G_RIGHT_PAD   = 20
_G_MIN_PPU     = 2   # Minimum pixels per time unit


def _g_bay_color(bay_id: int, alpha: int = 200) -> QColor:
    hue = _G_BAY_HUES[bay_id % len(_G_BAY_HUES)]
    return QColor.fromHsv(hue, 180, 210, alpha)


def _g_bay_color_dark(bay_id: int) -> QColor:
    hue = _G_BAY_HUES[bay_id % len(_G_BAY_HUES)]
    return QColor.fromHsv(hue, 220, 150, 255)


class BayGanttCanvas(QWidget):
    """Block timing Gantt drawing widget.

    Each row = one block.
    bar = [release_time, release_time + processing_time].
    Shows due_date markers. Blocks placed in bays are highlighted with bay colors.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self._rows: list[dict] = []   # {block_id, bay_id, release, proc, due}
        self._t_min:  float = 0.0
        self._t_max:  float = 1.0
        self._ppu:    float = 10.0    # px per time unit
        self._show_due:     bool = True
        self._show_release: bool = True
        self._hovered: int = -1

    def set_data(self, rows: list[dict], t_min: float, t_max: float, ppu: float):
        self._rows  = rows
        self._t_min = t_min
        self._t_max = t_max
        self._ppu   = max(_G_MIN_PPU, ppu)
        self._resize()
        self.update()

    def set_show_due(self, v: bool):
        self._show_due = v
        self.update()

    def set_show_release(self, v: bool):
        self._show_release = v
        self.update()

    def _resize(self):
        w = _G_LABEL_W + int((self._t_max - self._t_min) * self._ppu) + _G_RIGHT_PAD
        h = _G_AXIS_H + len(self._rows) * _G_ROW_H + 4
        self.setMinimumSize(w, h)
        self.resize(w, h)

    def _t2x(self, t: float) -> float:
        return _G_LABEL_W + (t - self._t_min) * self._ppu

    def _row_y(self, i: int) -> float:
        return _G_AXIS_H + i * _G_ROW_H

    def _row_at(self, py: int) -> int:
        ry = py - _G_AXIS_H
        if ry < 0:
            return -1
        idx = ry // _G_ROW_H
        return idx if idx < len(self._rows) else -1

    def paintEvent(self, _ev):
        if not self._rows:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self._draw_axis(p)
        for i, row in enumerate(self._rows):
            self._draw_row(p, i, row)
        p.end()

    def _draw_axis(self, p: QPainter):
        w = self.width()
        t_span = self._t_max - self._t_min
        if t_span <= 0:
            return
        p.fillRect(0, 0, _G_LABEL_W, _G_AXIS_H, QColor("#f1f5f9"))
        p.fillRect(_G_LABEL_W, 0, w - _G_LABEL_W, _G_AXIS_H, QColor("#e2e8f0"))

        nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]
        target = max(4, min(20, int((w - _G_LABEL_W) / 60)))
        step = nice[-1]
        for ni in nice:
            if ni >= t_span / target:
                step = ni
                break

        font = QFont("Helvetica", 9)
        p.setFont(font)
        fm = QFontMetricsF(font)
        t = math.ceil(self._t_min / step) * step
        while t <= self._t_max + 1e-6:
            x = self._t2x(t)
            p.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, _G_AXIS_H), QPointF(x, self.height()))
            p.setPen(QPen(QColor("#94a3b8"), 1))
            p.drawLine(QPointF(x, _G_AXIS_H - 5), QPointF(x, _G_AXIS_H))
            lbl = str(int(t))
            lw = fm.horizontalAdvance(lbl)
            p.setPen(QPen(QColor("#334155"), 1))
            p.drawText(QPointF(x - lw / 2, _G_AXIS_H - 8), lbl)
            t += step

        p.setPen(QPen(QColor("#94a3b8"), 1))
        p.drawLine(0, _G_AXIS_H, w, _G_AXIS_H)

    def _draw_row(self, p: QPainter, ri: int, row: dict):
        y        = self._row_y(ri)
        w        = self.width()
        bay_id   = row["bay_id"]      # -1 = not placed
        release  = row["release"]
        proc     = row["proc"]
        due      = row["due"]
        block_id = row["block_id"]
        bar_end  = release + proc
        in_bay   = bay_id >= 0

        bg = QColor("#f8fafc") if ri % 2 == 0 else QColor("#f1f5f9")
        p.fillRect(0, int(y), w, _G_ROW_H, bg)
        if ri == self._hovered:
            p.fillRect(0, int(y), w, _G_ROW_H, QColor(0, 100, 255, 18))

        bar_h = _G_ROW_H - 8
        bar_y = int(y + 4)

        # Shade the pre-release interval
        if self._show_release and release > self._t_min:
            rx_end = self._t2x(min(release, self._t_max))
            p.fillRect(int(_G_LABEL_W), int(y + 2),
                       max(0, int(rx_end - _G_LABEL_W)), _G_ROW_H - 4,
                       QColor(0, 0, 0, 22))

        # bar
        x0 = self._t2x(release)
        x1 = self._t2x(bar_end)
        bar_w = max(2, int(x1 - x0))
        if in_bay:
            fill   = _g_bay_color(bay_id, 200)
            border = _g_bay_color_dark(bay_id)
        else:
            fill   = QColor(148, 163, 184, 180)   # slate-400
            border = QColor(71, 85, 105, 220)      # slate-600
        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 1))
        p.drawRect(int(x0), bar_y, bar_w, bar_h)

        # release marker (▷)
        if self._show_release and self._t_min <= release <= self._t_max:
            rel_x = self._t2x(release)
            mid_y = y + _G_ROW_H / 2
            sz = 5
            p.setBrush(QBrush(QColor(6, 182, 212, 220)))
            p.setPen(QPen(QColor(14, 116, 144), 1))
            p.drawPolygon(QPolygonF([
                QPointF(rel_x, mid_y - sz),
                QPointF(rel_x, mid_y + sz),
                QPointF(rel_x + sz * 1.2, mid_y),
            ]))

        # due-date marker (▼)
        if self._show_due and self._t_min <= due <= self._t_max:
            due_x  = self._t2x(due)
            c_due  = QColor("#16a34a")
            cd_dark = QColor("#166534")
            p.setPen(QPen(c_due, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(due_x, y), QPointF(due_x, y + _G_ROW_H))
            sz = 5
            p.setBrush(QBrush(c_due))
            p.setPen(QPen(cd_dark, 1))
            p.drawPolygon(QPolygonF([
                QPointF(due_x - sz, y),
                QPointF(due_x + sz, y),
                QPointF(due_x,      y + sz + 1),
            ]))

        # label
        p.fillRect(0, int(y), _G_LABEL_W, _G_ROW_H, bg)
        font_b = QFont("Helvetica", 9, QFont.Weight.Bold)
        font_s = QFont("Helvetica", 8)
        p.setFont(font_b)
        p.setPen(QPen(QColor("#1e293b"), 1))
        lbl = f"B{block_id}"
        fm_b = QFontMetricsF(font_b)
        p.drawText(QPointF(_G_LABEL_W - fm_b.horizontalAdvance(lbl) - 6,
                           y + _G_ROW_H / 2 - 1), lbl)
        if in_bay:
            p.setFont(font_s)
            p.setPen(QPen(QColor("#64748b"), 1))
            sub = f"bay{bay_id}"
            fm_s = QFontMetricsF(font_s)
            p.drawText(QPointF(_G_LABEL_W - fm_s.horizontalAdvance(sub) - 6,
                               y + _G_ROW_H / 2 + fm_s.ascent() - 1), sub)
            p.fillRect(0, int(y + 2), 5, _G_ROW_H - 4, _g_bay_color(bay_id, 220))
        else:
            p.fillRect(0, int(y + 2), 5, _G_ROW_H - 4, QColor(148, 163, 184, 200))

        p.setPen(QPen(QColor("#e2e8f0"), 1))
        p.drawLine(0, int(y + _G_ROW_H - 1), w, int(y + _G_ROW_H - 1))

    def mouseMoveEvent(self, ev):
        ri = self._row_at(ev.pos().y())
        if ri != self._hovered:
            self._hovered = ri
            self.update()
        if 0 <= ri < len(self._rows):
            row = self._rows[ri]
            bay_str = f"Bay {row['bay_id']}" if row["bay_id"] >= 0 else "unplaced"
            tip = (
                f"<b>Block {row['block_id']}</b>  {bay_str}<br>"
                f"Release: <b>{row['release']}</b> &nbsp; "
                f"Proc: <b>{row['proc']}</b> &nbsp; "
                f"Due: <b>{row['due']}</b>"
            )
            QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, _ev):
        self._hovered = -1
        self.update()


class BayGanttPanel(QWidget):
    """Gantt panel attached to the bottom of BayLayoutTab (scroll + toolbar)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instance: Optional[dict] = None
        self._rows: list[dict] = []
        self._ppu: float = 10.0

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Toolbar
        bar = QHBoxLayout()
        bar.setSpacing(6)

        bar.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Block ID", "Release", "Due date"])
        self._sort_combo.setFixedWidth(120)
        self._sort_combo.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self._sort_combo)

        bar.addSpacing(8)
        self._chk_due = QCheckBox("Due markers")
        self._chk_due.setChecked(True)
        self._chk_due.toggled.connect(self._canvas_show_due)
        bar.addWidget(self._chk_due)

        self._chk_rel = QCheckBox("Release shade")
        self._chk_rel.setChecked(True)
        self._chk_rel.toggled.connect(self._canvas_show_release)
        bar.addWidget(self._chk_rel)

        bar.addSpacing(8)
        bar.addWidget(QLabel("Zoom:"))
        btn_out = QPushButton("-")
        btn_out.setFixedSize(24, 24)
        btn_in  = QPushButton("+")
        btn_in.setFixedSize(24, 24)
        btn_fit = QPushButton("Fit")
        btn_fit.setFixedWidth(34)
        btn_out.clicked.connect(self._zoom_out)
        btn_in.clicked.connect(self._zoom_in)
        btn_fit.clicked.connect(self._zoom_fit)
        bar.addWidget(btn_out)
        bar.addWidget(btn_in)
        bar.addWidget(btn_fit)
        bar.addStretch()

        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet("color:#64748b; font-size:10px;")
        bar.addWidget(self._lbl_info)
        layout.addLayout(bar)

        # Canvas
        self._canvas = BayGanttCanvas()
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self._canvas)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFixedHeight(200)
        self._scroll = scroll
        layout.addWidget(scroll)

    # -- Public API ----------------------------------------------------------
    def refresh_from_bays(self, instance: dict,
                          bay_canvases: "list[BayCanvas]"):
        """Update the Gantt row list to reflect current bay placements."""
        self._instance = instance
        # Per-bay block id -> bay_id mapping
        bay_map: dict[int, int] = {}
        for bc in bay_canvases:
            for blk in bc.blocks:
                bay_map[blk.id] = bc.bay_idx
        rows = []
        for bi, blk_data in enumerate(instance["blocks"]):
            release = float(blk_data.get("release_time", 0) or 0)
            proc    = float(blk_data.get("processing_time", 1) or 1)
            due     = float(blk_data.get("due_date", release + proc) or (release + proc))
            rows.append({
                "block_id": bi,
                "bay_id":   bay_map.get(bi, -1),
                "release":  release,
                "proc":     proc,
                "due":      due,
            })
        self._rows = rows
        self._zoom_fit()
        self._update_info()

    # -- Internal helpers ----------------------------------------------------
    def _sorted_rows(self) -> list[dict]:
        rows = list(self._rows)
        key = self._sort_combo.currentText()
        if key == "Block ID":
            rows.sort(key=lambda r: r["block_id"])
        elif key == "Release":
            rows.sort(key=lambda r: r["release"])
        elif key == "Due date":
            rows.sort(key=lambda r: r["due"])
        return rows

    def _refresh(self):
        if not self._rows:
            return
        rows = self._sorted_rows()
        all_t = [r["release"] for r in rows] + [r["release"] + r["proc"] for r in rows]
        if self._chk_due.isChecked():
            all_t += [r["due"] for r in rows]
        t_min = max(0.0, min(all_t) - 0.5)
        t_max = max(all_t) + 1.0
        self._canvas.set_data(rows, t_min, t_max, self._ppu)

    def _zoom_fit(self):
        if not self._rows:
            return
        rows = self._sorted_rows()
        all_t = [r["release"] for r in rows] + [r["release"] + r["proc"] for r in rows]
        if self._chk_due.isChecked():
            all_t += [r["due"] for r in rows]
        t_min = max(0.0, min(all_t))
        t_max = max(all_t) + 1.0
        span  = max(1.0, t_max - t_min)
        avail = max(200, self._scroll.viewport().width() - _G_LABEL_W - _G_RIGHT_PAD - 20)
        self._ppu = avail / span
        self._refresh()

    def _zoom_in(self):
        self._ppu = min(self._ppu * 1.4, 200.0)
        self._refresh()

    def _zoom_out(self):
        self._ppu = max(self._ppu / 1.4, _G_MIN_PPU)
        self._refresh()

    def _canvas_show_due(self, v: bool):
        self._canvas.set_show_due(v)

    def _canvas_show_release(self, v: bool):
        self._canvas.set_show_release(v)

    def _update_info(self):
        if not self._rows:
            self._lbl_info.setText("")
            return
        n = len(self._rows)
        n_bay = sum(1 for r in self._rows if r["bay_id"] >= 0)
        self._lbl_info.setText(
            f"{n} blocks  |  placed: {n_bay}  unplaced: {n - n_bay}"
        )


# -----------------------------------------------------------------------------
# BayLayoutTab
# -----------------------------------------------------------------------------
class BayLayoutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._instance: Optional[dict] = None
        self._visible_layers: Optional[set] = None
        self._layer_checkboxes: list[QCheckBox] = []
        self._bay_scale = BAY_SCALE_DEF

        self._drag_blk: Optional[BlockItem] = None
        self._drag_src = None
        self._drag_grab_offset: QPointF = QPointF(0.0, 0.0)
        self._drag_over_bay: Optional["BayCanvas"] = None  # Bay currently under the drag cursor
        self._overlay: Optional[DragOverlay] = None

        self._panel_canvas: Optional[BlockPanelCanvas] = None
        self._bay_canvases: list[BayCanvas] = []
        self._panel_blocks: list[BlockItem] = []
        self._gantt_panel: Optional[BayGanttPanel] = None

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Top control row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Layers:"))
        self._layer_row_widget = QWidget()
        self._layer_row_inner = QHBoxLayout(self._layer_row_widget)
        self._layer_row_inner.setContentsMargins(0, 0, 0, 0)
        self._layer_row_inner.setSpacing(4)
        ctrl.addWidget(self._layer_row_widget)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Scale:"))
        btn_minus = QPushButton("-"); btn_minus.setFixedWidth(26)
        btn_plus  = QPushButton("+"); btn_plus.setFixedWidth(26)
        btn_minus.clicked.connect(lambda: self._adjust_scale(-1))
        btn_plus.clicked.connect(lambda:  self._adjust_scale(+1))
        ctrl.addWidget(btn_minus)
        ctrl.addWidget(btn_plus)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._reset_layout)
        ctrl.addSpacing(8)
        ctrl.addWidget(btn_reset)
        ctrl.addStretch()

        self._info_label = QLabel("")
        ctrl.addWidget(self._info_label)
        outer.addLayout(ctrl)

        # Top area: block panel | bay panel  (horizontal splitter)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)

        # Left: block panel scroll
        panel_frame = QWidget()
        panel_frame.setFixedWidth(PANEL_W)
        pf_layout = QVBoxLayout(panel_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setSpacing(0)
        lbl = QLabel("  Block Panel")
        lbl.setFixedHeight(22)
        lbl.setStyleSheet("background:#334155;color:white;font-weight:bold;")
        pf_layout.addWidget(lbl)
        self._panel_scroll = QScrollArea()
        self._panel_scroll.setWidgetResizable(False)
        self._panel_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pf_layout.addWidget(self._panel_scroll, stretch=1)

        # Right: bay vertical scroll
        self._bays_scroll = QScrollArea()
        self._bays_scroll.setWidgetResizable(True)
        self._bays_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._bays_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._bays_container = QWidget()
        self._bays_layout = QVBoxLayout(self._bays_container)
        self._bays_layout.setContentsMargins(4, 4, 4, 4)
        self._bays_layout.setSpacing(8)
        self._bays_layout.addStretch()
        self._bays_scroll.setWidget(self._bays_container)

        self._splitter.addWidget(panel_frame)
        self._splitter.addWidget(self._bays_scroll)
        self._splitter.setSizes([PANEL_W, 9999])

        # Gantt panel
        self._gantt_panel = BayGanttPanel()

        # Vertical splitter: top (bay layout) | bottom (Gantt)
        self._v_splitter = QSplitter(Qt.Orientation.Vertical)
        self._v_splitter.setChildrenCollapsible(True)
        self._v_splitter.addWidget(self._splitter)
        self._v_splitter.addWidget(self._gantt_panel)
        self._v_splitter.setSizes([600, 230])
        outer.addWidget(self._v_splitter, stretch=1)

        # Drag overlay
        self._overlay = DragOverlay(self)

    # -- Instance loading ----------------------------------------------------
    def set_instance(self, instance: dict):
        self._instance = instance
        self._panel_blocks = [
            BlockItem(bi, blk_data)
            for bi, blk_data in enumerate(instance["blocks"])
        ]

        # Auto-calculate initial scale based on bay size -- fit the largest bay
        bays = instance.get("bays", [{}])
        bay_w = max((b.get("width",  200) for b in bays), default=200)
        bay_h = max((b.get("height",  60) for b in bays), default=60)
        self._bay_scale = _auto_scale(bay_w, bay_h)

        max_layers = max(
            (len(_resolve_layers(o["layers"]))
             for blk in instance["blocks"]
             for o in blk["shape"]),
            default=1,
        )
        self._rebuild_layer_checkboxes(max_layers)
        self._rebuild_bays(instance)
        self._rebuild_panel()
        self._refresh_gantt()

        n_blocks = len(instance["blocks"])
        n_bays   = len(instance["bays"])
        self._info_label.setText(
            f"{n_blocks} block(s)  |  {n_bays} bay(s)  |  scale={self._bay_scale:.1f}"
        )

    def _rebuild_panel(self):
        old = self._panel_scroll.takeWidget()
        if old:
            old.deleteLater()
        canvas = BlockPanelCanvas()
        canvas.blocks = list(sorted(self._panel_blocks, key=lambda b: b.id))
        canvas.visible_layers = self._visible_layers
        canvas.SCALE = self._bay_scale          # Apply same scale as bay
        # Pass average bay area (for area ratio display)
        if self._instance:
            bays = self._instance.get("bays", [])
            if bays:
                canvas._bay_area = sum(b["width"] * b["height"] for b in bays) / len(bays)
        canvas.drag_started.connect(self._on_drag_started)
        self._panel_canvas = canvas
        self._panel_scroll.setWidget(canvas)
        canvas.rebuild()

    def _rebuild_bays(self, instance: dict):
        for bc in self._bay_canvases:
            bc.setParent(None)
            bc.deleteLater()
        self._bay_canvases.clear()
        while self._bays_layout.count():
            self._bays_layout.takeAt(0)

        bays = instance["bays"]
        for i, bay in enumerate(bays):
            canvas = BayCanvas(bay["width"], bay["height"], bay_idx=i,
                               bay_id=bay.get("id", i + 1))
            canvas.set_scale(self._bay_scale)
            canvas.visible_layers = self._visible_layers
            canvas.drag_started.connect(self._on_drag_started)
            canvas.block_changed.connect(self._refresh_gantt)
            self._bay_canvases.append(canvas)
            self._bays_layout.addWidget(canvas)
        self._bays_layout.addStretch()
        self._bays_container.adjustSize()

    # -- Layer checkboxes ----------------------------------------------------
    def _rebuild_layer_checkboxes(self, max_layers: int):
        while self._layer_row_inner.count():
            item = self._layer_row_inner.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._layer_checkboxes.clear()
        for i in range(max_layers):
            gray = _layer_gray(i)
            cb = QCheckBox(f"L{i}")
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox {{ background: rgb({gray},{gray},{gray}); "
                f"padding: 1px 4px; border-radius: 3px; }}"
            )
            cb.stateChanged.connect(self._on_layer_toggled)
            self._layer_row_inner.addWidget(cb)
            self._layer_checkboxes.append(cb)
        self._visible_layers = None

    def _on_layer_toggled(self):
        checked = {i for i, cb in enumerate(self._layer_checkboxes) if cb.isChecked()}
        self._visible_layers = None if len(checked) == len(self._layer_checkboxes) else checked
        if self._panel_canvas:
            self._panel_canvas.visible_layers = self._visible_layers
            self._panel_canvas.update()
        for bc in self._bay_canvases:
            bc.visible_layers = self._visible_layers
            bc.mark_dirty()

    def _adjust_scale(self, delta: int):
        self._bay_scale = max(_SCALE_MIN, min(_SCALE_MAX,
                              self._bay_scale + delta * _SCALE_STEP))
        for bc in self._bay_canvases:
            bc.set_scale(self._bay_scale)
        if self._panel_canvas:
            self._panel_canvas.set_scale(self._bay_scale)
        self._bays_container.adjustSize()
        # Reflect current scale in info label
        if self._instance:
            n_blocks = len(self._instance["blocks"])
            n_bays   = len(self._instance["bays"])
            self._info_label.setText(
                f"{n_blocks} block(s)  |  {n_bays} bay(s)  |  scale={self._bay_scale:.1f}"
            )

    def _refresh_gantt(self):
        if self._instance and self._gantt_panel:
            self._gantt_panel.refresh_from_bays(
                self._instance, self._bay_canvases)

    def _reset_layout(self):
        if self._instance:
            self.set_instance(self._instance)

    # -- Centralized drag management ------------------------------------------
    def _on_drag_started(self, blk: BlockItem, global_pos: QPoint,
                          grab_offset: QPointF = None):
        src = self.sender()
        if isinstance(src, BlockPanelCanvas):
            if blk in self._panel_blocks:
                self._panel_blocks.remove(blk)
            src.blocks = list(sorted(self._panel_blocks, key=lambda b: b.id))
            src.rebuild()
        elif isinstance(src, BayCanvas):
            if blk in src.blocks:
                src.blocks.remove(blk)
            src.mark_after_removal(blk.id)

        self._drag_blk = blk
        self._drag_src = src
        self._drag_grab_offset = grab_offset if grab_offset is not None else QPointF(0.0, 0.0)

        ghost_scale = self._bay_scale
        local_pos = self.mapFromGlobal(global_pos)
        self._overlay.start(blk, local_pos, ghost_scale, self._visible_layers,
                            self._drag_grab_offset)
        self.grabMouse()

    def mouseMoveEvent(self, ev):
        if self._drag_blk is not None:
            self._overlay.move_to(ev.pos())
            # -- Cross-section preview: update only the bay under the cursor ---
            global_pos = self.mapToGlobal(ev.pos())
            over_bay = None
            for bc in self._bay_canvases:
                local = bc.mapFromGlobal(global_pos)
                if bc.rect().contains(local) and local.x() < bc._bay_px_w and local.y() < bc._bay_px_h:
                    over_bay = bc
                    break
            if over_bay is not self._drag_over_bay:
                if self._drag_over_bay is not None:
                    self._drag_over_bay.clear_drag_preview()
                self._drag_over_bay = over_bay
            if over_bay is not None:
                local = over_bay.mapFromGlobal(global_pos)
                bx, by = over_bay._to_bay(local.x(), local.y())
                bb = self._drag_blk.anchored_bb()
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                nx = bx - self._drag_grab_offset.x()
                ny = by - self._drag_grab_offset.y()
                over_bay.set_drag_preview(self._drag_blk, nx, ny)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_blk is not None and ev.button() == Qt.MouseButton.LeftButton:
            self.releaseMouse()
            self._overlay.stop()
            # Clear preview
            for bc in self._bay_canvases:
                bc.clear_drag_preview()
            self._drag_over_bay = None

            global_pos = self.mapToGlobal(ev.pos())
            dropped = False

            # Drop onto bay?
            for bc in self._bay_canvases:
                if bc.rect().contains(bc.mapFromGlobal(global_pos)):
                    bc.drop_block(self._drag_blk, global_pos, self._drag_grab_offset)
                    dropped = True
                    break

            # Drop onto block panel?
            if not dropped:
                panel_local = self._panel_scroll.mapFromGlobal(global_pos)
                if self._panel_scroll.rect().contains(panel_local):
                    if self._drag_blk not in self._panel_blocks:
                        self._panel_blocks.append(self._drag_blk)
                    self._panel_canvas.blocks = list(
                        sorted(self._panel_blocks, key=lambda b: b.id))
                    self._panel_canvas.rebuild()
                    dropped = True

            # No drop target -> revert
            if not dropped:
                if isinstance(self._drag_src, BlockPanelCanvas):
                    if self._drag_blk not in self._panel_blocks:
                        self._panel_blocks.append(self._drag_blk)
                    self._panel_canvas.blocks = list(
                        sorted(self._panel_blocks, key=lambda b: b.id))
                    self._panel_canvas.rebuild()
                elif isinstance(self._drag_src, BayCanvas):
                    self._drag_src.blocks.append(self._drag_blk)
                    self._drag_src.mark_dirty()

            self._drag_blk = None
            self._drag_src = None
            self._refresh_gantt()
        super().mouseReleaseEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._overlay:
            self._overlay.setGeometry(self.rect())
