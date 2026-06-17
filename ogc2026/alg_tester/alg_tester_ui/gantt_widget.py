"""
Gantt chart tab for alg_tester.

Shows block entry/exit intervals on a horizontal timeline.

Layout:
  +----------------------------------------------------------+
  | [Sort: Block ID ▼]  [Group by Bay ☐]  [Show due dates ☑] |
  +--------+-------------------------------------------------+
  | Block  |  time axis  ->                                   |
  |  0     |  ████████░░░░░░   due▲ T=3                     |
  |  1     |       ██████████                                |
  | ...    |                                                 |
  +--------+-------------------------------------------------+

Colour coding:
  Bay 0 -> hue  210  (blue family)
  Bay 1 -> hue   40  (orange family)
  Bay 2 -> hue  130  (green family)
  Bay 3 -> hue  280  (purple family)
  ...

  Bar fill: per-bay colour, solid.
  Tardiness overlay: red hatching on the [due_date, exit_time] portion when tardy.
  Due-date tick: small triangle / vertical line on the bar.
  Release-time: left edge of the row shaded lightly.
"""

import math
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QCheckBox, QComboBox,
    QSizePolicy, QToolTip, QSplitter, QAbstractScrollArea,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF,
    QFont, QFontMetricsF, QPainterPath, QCursor,
)


# -----------------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------------
_BAY_HUES = [210, 35, 130, 280, 10, 170, 310, 60]

def _bay_color(bay_id: int, alpha: int = 200) -> QColor:
    hue = _BAY_HUES[bay_id % len(_BAY_HUES)]
    return QColor.fromHsv(hue, 180, 210, alpha)

def _bay_color_dark(bay_id: int, alpha: int = 255) -> QColor:
    hue = _BAY_HUES[bay_id % len(_BAY_HUES)]
    return QColor.fromHsv(hue, 220, 150, alpha)


# -----------------------------------------------------------------------------
# GanttCanvas -- the actual drawing widget (inside a scroll area)
# -----------------------------------------------------------------------------
_ROW_H     = 26       # row height px
_LABEL_W   = 90       # left label column width px (wide enough for "B0 / Bay 0")
_AXIS_H    = 28       # top time-axis height px
_RIGHT_PAD = 20       # right padding px
_MIN_PX_PER_UNIT = 4  # minimum pixels per time unit (prevents overcrowding)


class GanttCanvas(QWidget):
    """Pure drawing widget for the Gantt chart; no scroll logic."""

    block_hovered = pyqtSignal(int)   # block_id, -1 = none

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self._rows:    list[dict] = []   # [{block_id, bay_id, entry, exit, due, release, workload}]
        self._t_min:   float = 0.0
        self._t_max:   float = 1.0
        self._px_per_unit: float = 10.0
        self._show_due:    bool  = True
        self._show_release: bool = True

        self._hovered_row: int = -1   # row index

    # -- data -----------------------------------------------------------------
    def set_data(self, rows: list[dict], t_min: float, t_max: float,
                 px_per_unit: float):
        self._rows = rows
        self._t_min = t_min
        self._t_max = t_max
        self._px_per_unit = max(_MIN_PX_PER_UNIT, px_per_unit)
        self._update_size()
        self.update()

    def set_show_due(self, v: bool):
        self._show_due = v
        self.update()

    def set_show_release(self, v: bool):
        self._show_release = v
        self.update()

    def _update_size(self):
        w = _LABEL_W + int((self._t_max - self._t_min) * self._px_per_unit) + _RIGHT_PAD
        h = _AXIS_H + len(self._rows) * _ROW_H + 4
        self.setMinimumSize(w, h)
        self.resize(w, h)

    # -- coordinate helpers ----------------------------------------------------
    def _t_to_px(self, t: float) -> float:
        return _LABEL_W + (t - self._t_min) * self._px_per_unit

    def _row_y(self, row_idx: int) -> float:
        return _AXIS_H + row_idx * _ROW_H

    def _row_at(self, y: int) -> int:
        ry = y - _AXIS_H
        if ry < 0:
            return -1
        idx = ry // _ROW_H
        return idx if idx < len(self._rows) else -1

    # -- paint -----------------------------------------------------------------
    def paintEvent(self, _event):
        if not self._rows:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        self._draw_axis(p)
        for ri, row in enumerate(self._rows):
            self._draw_row(p, ri, row)
        p.end()

    def _draw_axis(self, p: QPainter):
        w = self.width()
        t_span = self._t_max - self._t_min
        if t_span <= 0:
            return

        # Background
        p.fillRect(0, 0, _LABEL_W, _AXIS_H, QColor("#f1f5f9"))
        p.fillRect(_LABEL_W, 0, w - _LABEL_W, _AXIS_H, QColor("#e2e8f0"))

        # Tick spacing: choose a "nice" interval
        nice_intervals = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]
        target_ticks = max(4, min(20, int((w - _LABEL_W) / 60)))
        raw_step = t_span / target_ticks
        tick_step = nice_intervals[0]
        for ni in nice_intervals:
            if ni >= raw_step:
                tick_step = ni
                break

        font = QFont("Helvetica", 9)
        p.setFont(font)
        fm = QFontMetricsF(font)
        pen_tick = QPen(QColor("#94a3b8"), 1)
        pen_label = QPen(QColor("#334155"), 1)

        t = math.ceil(self._t_min / tick_step) * tick_step
        while t <= self._t_max + 1e-6:
            x = self._t_to_px(t)
            # Major tick line through all rows
            p.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, _AXIS_H), QPointF(x, self.height()))
            # Axis tick + label
            p.setPen(pen_tick)
            p.drawLine(QPointF(x, _AXIS_H - 5), QPointF(x, _AXIS_H))
            lbl = str(int(t))
            lw = fm.horizontalAdvance(lbl)
            p.setPen(pen_label)
            p.drawText(QPointF(x - lw / 2, _AXIS_H - 8), lbl)
            t += tick_step

        # Axis bottom border
        p.setPen(QPen(QColor("#94a3b8"), 1))
        p.drawLine(0, _AXIS_H, w, _AXIS_H)

    def _draw_row(self, p: QPainter, ri: int, row: dict):
        y = self._row_y(ri)
        w = self.width()
        bay_id   = row["bay_id"]
        entry    = row["entry"]
        exit_t   = row["exit"]
        due      = row["due"]
        release  = row["release"]
        block_id = row["block_id"]
        tardy    = max(0.0, exit_t - due)

        # -- Row background (alternating) --------------------------------------
        bg = QColor("#f8fafc") if ri % 2 == 0 else QColor("#f1f5f9")
        p.fillRect(0, int(y), w, _ROW_H, bg)

        # -- Hover highlight ---------------------------------------------------
        if ri == self._hovered_row:
            p.fillRect(0, int(y), w, _ROW_H, QColor(0, 100, 255, 18))

        # -- Release-time shade (before release) -------------------------------
        if self._show_release and release > self._t_min:
            rx_end = self._t_to_px(min(release, self._t_max))
            p.fillRect(
                int(_LABEL_W), int(y + 2),
                max(0, int(rx_end - _LABEL_W)), _ROW_H - 4,
                QColor(0, 0, 0, 22)
            )

        # -- Bar ---------------------------------------------------------------
        x0 = self._t_to_px(entry)
        x1 = self._t_to_px(exit_t)
        bar_h = _ROW_H - 8
        bar_y = int(y + 4)
        bar_w = max(2, int(x1 - x0))

        color_fill = _bay_color(bay_id, 200)
        color_border = _bay_color_dark(bay_id)

        p.setBrush(QBrush(color_fill))
        p.setPen(QPen(color_border, 1))
        p.drawRect(int(x0), bar_y, bar_w, bar_h)

        # -- Tardiness hatch overlay -------------------------------------------
        if tardy > 1e-6 and self._show_due:
            due_x = self._t_to_px(due)
            hatch_x = max(int(x0), int(due_x))
            hatch_w = int(x1) - hatch_x
            if hatch_w > 0:
                p.save()
                clip = QPainterPath()
                clip.addRect(QRectF(hatch_x, bar_y, hatch_w, bar_h))
                p.setClipPath(clip)
                p.fillRect(hatch_x, bar_y, hatch_w, bar_h, QColor(220, 38, 38, 80))
                pen_hatch = QPen(QColor(180, 0, 0, 120), 1.0)
                p.setPen(pen_hatch)
                step = 5
                i = step
                while i < hatch_w + bar_h:
                    p.drawLine(
                        QPointF(hatch_x + i - bar_h, bar_y + bar_h),
                        QPointF(hatch_x + i, bar_y)
                    )
                    i += step
                p.restore()

        # -- Release marker (▷ cyan triangle on bar left edge) ----------------
        if self._show_release and self._t_min <= release <= self._t_max:
            rel_x = self._t_to_px(release)
            mid_y = y + _ROW_H / 2
            sz = 5
            tri_rel = QPolygonF([
                QPointF(rel_x,      mid_y - sz),
                QPointF(rel_x,      mid_y + sz),
                QPointF(rel_x + sz * 1.2, mid_y),
            ])
            p.setBrush(QBrush(QColor(6, 182, 212, 220)))    # cyan-500
            p.setPen(QPen(QColor(14, 116, 144), 1))         # cyan-700
            p.drawPolygon(tri_rel)

        # -- Due-date tick (▼ downward triangle above bar + vertical line) -----
        if self._show_due and self._t_min <= due <= self._t_max:
            due_x = self._t_to_px(due)
            color_due = QColor("#dc2626") if tardy > 1e-6 else QColor("#16a34a")
            color_due_dark = QColor("#991b1b") if tardy > 1e-6 else QColor("#166534")
            # Vertical dashed line through the full row
            p.setPen(QPen(color_due, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(due_x, y), QPointF(due_x, y + _ROW_H))
            # Downward triangle above bar (▼)
            sz = 5
            tri_due = QPolygonF([
                QPointF(due_x - sz, y),
                QPointF(due_x + sz, y),
                QPointF(due_x,      y + sz + 1),
            ])
            p.setBrush(QBrush(color_due))
            p.setPen(QPen(color_due_dark, 1))
            p.drawPolygon(tri_due)

        # -- Row label (left column): block id + bay id ------------------------
        p.fillRect(0, int(y), _LABEL_W, _ROW_H, bg)
        font_b = QFont("Helvetica", 9, QFont.Weight.Bold)
        font_s = QFont("Helvetica", 8)
        p.setFont(font_b)
        p.setPen(QPen(QColor("#1e293b"), 1))
        lbl_block = f"B{block_id}"
        fm_b = QFontMetricsF(font_b)
        lbl_x = _LABEL_W - fm_b.horizontalAdvance(lbl_block) - 6
        p.drawText(QPointF(lbl_x, y + _ROW_H / 2 - 1), lbl_block)

        # Bay id sub-label in muted colour
        p.setFont(font_s)
        p.setPen(QPen(QColor("#64748b"), 1))
        lbl_bay = f"bay{bay_id + 1}"
        fm_s = QFontMetricsF(font_s)
        bay_x = _LABEL_W - fm_s.horizontalAdvance(lbl_bay) - 6
        p.drawText(QPointF(bay_x, y + _ROW_H / 2 + fm_s.ascent() - 1), lbl_bay)

        # -- Bay colour strip in label area ------------------------------------
        strip_color = _bay_color(bay_id, 220)
        p.fillRect(0, int(y + 2), 6, _ROW_H - 4, strip_color)

        # -- Tardiness label inside / beside bar ------------------------------
        if tardy > 1e-6:
            t_str = f"T={tardy:.0f}"
            font_t = QFont("Helvetica", 8)
            p.setFont(font_t)
            p.setPen(QPen(QColor("#7f1d1d"), 1))
            ft = QFontMetricsF(font_t)
            tx = int(x1) + 3
            ty = int(y + _ROW_H / 2 + ft.ascent() / 2 - 1)
            if tx + ft.horizontalAdvance(t_str) < w:
                p.drawText(QPointF(tx, ty), t_str)

        # -- Row separator -----------------------------------------------------
        p.setPen(QPen(QColor("#e2e8f0"), 1))
        p.drawLine(0, int(y + _ROW_H - 1), w, int(y + _ROW_H - 1))

    # -- Mouse events ----------------------------------------------------------
    def mouseMoveEvent(self, event):
        ri = self._row_at(event.pos().y())
        if ri != self._hovered_row:
            self._hovered_row = ri
            self.update()
        if 0 <= ri < len(self._rows):
            row = self._rows[ri]
            tardy = max(0.0, row["exit"] - row["due"])
            tip = (
                f"Block {row['block_id']}  |  Bay {row['bay_id'] + 1}\n"
                f"Entry: {row['entry']:.1f}   Exit: {row['exit']:.1f}\n"
                f"Release: {row['release']}   Due: {row['due']}   "
                f"Proc: {row['exit'] - row['entry']:.1f}\n"
                f"Workload: {row['workload']}   "
                f"Tardiness: {tardy:.1f}"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
            self.block_hovered.emit(row["block_id"])
        else:
            QToolTip.hideText()
            self.block_hovered.emit(-1)

    def leaveEvent(self, _event):
        self._hovered_row = -1
        self.update()
        self.block_hovered.emit(-1)


# -----------------------------------------------------------------------------
# GanttTab -- scrollable container + controls
# -----------------------------------------------------------------------------
class GanttTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._instance: Optional[dict] = None
        self._solution: Optional[dict] = None
        self._rows: list[dict] = []

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # -- Toolbar -----------------------------------------------------------
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Block ID", "Bay -> Entry", "Entry time", "Due date", "Tardiness v"])
        self._sort_combo.setFixedWidth(130)
        self._sort_combo.currentIndexChanged.connect(self._refresh)
        toolbar.addWidget(self._sort_combo)

        toolbar.addSpacing(8)
        self._chk_due = QCheckBox("Due-date markers")
        self._chk_due.setChecked(True)
        self._chk_due.toggled.connect(lambda v: (self._canvas.set_show_due(v)))
        toolbar.addWidget(self._chk_due)

        self._chk_release = QCheckBox("Release shade")
        self._chk_release.setChecked(True)
        self._chk_release.toggled.connect(lambda v: (self._canvas.set_show_release(v)))
        toolbar.addWidget(self._chk_release)

        toolbar.addSpacing(8)
        toolbar.addWidget(QLabel("Zoom:"))
        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_out.setFixedSize(26, 26)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        toolbar.addWidget(self._btn_zoom_out)
        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedSize(26, 26)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        toolbar.addWidget(self._btn_zoom_in)
        self._btn_zoom_fit = QPushButton("Fit")
        self._btn_zoom_fit.setFixedWidth(38)
        self._btn_zoom_fit.clicked.connect(self._zoom_fit)
        toolbar.addWidget(self._btn_zoom_fit)

        toolbar.addStretch()

        # -- Legend ------------------------------------------------------------
        self._legend_layout = QHBoxLayout()
        self._legend_layout.setSpacing(6)
        self._legend_layout.addStretch()
        toolbar.addLayout(self._legend_layout)

        outer.addLayout(toolbar)

        # -- Canvas in scroll area ---------------------------------------------
        self._canvas = GanttCanvas()
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setWidget(self._canvas)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        outer.addWidget(self._scroll, stretch=1)

        # -- Status bar --------------------------------------------------------
        self._lbl_info = QLabel("No solution loaded")
        self._lbl_info.setStyleSheet("color:#64748b; font-size:11px;")
        outer.addWidget(self._lbl_info)

        self._px_per_unit: float = 10.0

    # -- Public API ------------------------------------------------------------
    def set_solution(self, instance: dict, solution: dict):
        self._instance = instance
        self._solution = solution
        self._rows = self._build_rows()
        self._zoom_fit()
        self._update_legend()
        self._update_info()

    # -- Internal helpers ------------------------------------------------------
    def _build_rows(self) -> list[dict]:
        if self._instance is None or self._solution is None:
            return []

        blocks_data = self._instance["blocks"]
        ops = self._solution.get("operations", {})

        # Reconstruct entry/exit times per block from operations
        entry_map: dict[int, dict] = {}
        exit_map:  dict[int, float] = {}
        for t_str, ops_at_t in ops.items():
            t = int(t_str)
            for op in ops_at_t:
                bid = op["block_id"]
                if op["type"] == "ENTRY":
                    entry_map[bid] = {
                        "bay_id":   op["bay_id"],
                        "entry":    float(t),
                        "x":        op.get("x", 0.0),
                        "y":        op.get("y", 0.0),
                    }
                elif op["type"] == "EXIT":
                    exit_map[bid] = float(t)

        rows = []
        for bid, ed in entry_map.items():
            blk = blocks_data[bid]
            rows.append({
                "block_id":   bid,
                "bay_id":   ed["bay_id"],
                "entry":    ed["entry"],
                "exit":     exit_map.get(bid, ed["entry"]),
                "due":      float(blk.get("due_date", 0)),
                "release":  float(blk.get("release_time", 0)),
                "workload": blk.get("workload", 0),
            })
        return rows

    def _sorted_rows(self) -> list[dict]:
        sort_key = self._sort_combo.currentText()
        rows = list(self._rows)
        if sort_key == "Block ID":
            rows.sort(key=lambda r: r["block_id"])
        elif sort_key == "Bay -> Entry":
            rows.sort(key=lambda r: (r["bay_id"], r["entry"]))
        elif sort_key == "Entry time":
            rows.sort(key=lambda r: r["entry"])
        elif sort_key == "Due date":
            rows.sort(key=lambda r: r["due"])
        elif sort_key == "Tardiness v":
            rows.sort(key=lambda r: -max(0.0, r["exit"] - r["due"]))
        return rows

    def _refresh(self):
        if not self._rows:
            return
        rows = self._sorted_rows()
        t_min = min(r["entry"] for r in rows)
        t_max = max(r["exit"] for r in rows)
        # Add small margin
        span = max(1.0, t_max - t_min)
        t_min = max(0.0, t_min - span * 0.02)
        t_max = t_max + span * 0.03
        self._canvas.set_data(rows, t_min, t_max, self._px_per_unit)

    def _zoom_fit(self):
        if not self._rows:
            return
        # Compute px_per_unit so the chart fits the scroll area width
        available_w = max(400, self._scroll.viewport().width() - _LABEL_W - _RIGHT_PAD - 20)
        rows = self._sorted_rows()
        t_min = min(r["entry"] for r in rows)
        t_max = max(r["exit"] for r in rows)
        span = max(1.0, t_max - t_min)
        self._px_per_unit = available_w / span
        self._refresh()

    def _zoom_in(self):
        self._px_per_unit = min(self._px_per_unit * 1.4, 200.0)
        self._refresh()

    def _zoom_out(self):
        self._px_per_unit = max(self._px_per_unit / 1.4, _MIN_PX_PER_UNIT)
        self._refresh()

    def _update_legend(self):
        # Clear old legend items
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._instance is None:
            return

        n_bays = len(self._instance.get("bays", []))
        for j in range(n_bays):
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            col = _bay_color(j, 255)
            swatch.setStyleSheet(
                f"background:{col.name()}; border:1px solid {_bay_color_dark(j).name()};"
            )
            lbl = QLabel(f"Bay {j + 1}")
            lbl.setStyleSheet("font-size:11px; color:#334155;")
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(3)
            row_l.addWidget(swatch)
            row_l.addWidget(lbl)
            self._legend_layout.addWidget(row_w)

        # Ready / Due / Tardiness legend entries
        for color, label in [
            ("#06b6d4", "▷ Ready (release)"),
            ("#16a34a", "▼ Due (on-time)"),
            ("#dc2626", "▼ Due (tardy) + hatch"),
        ]:
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background:{color}; border:1px solid #64748b;")
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size:11px; color:#334155;")
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(3)
            row_l.addWidget(swatch)
            row_l.addWidget(lbl)
            self._legend_layout.addWidget(row_w)

    def _update_info(self):
        if not self._rows:
            self._lbl_info.setText("No solution loaded")
            return
        n = len(self._rows)
        n_tardy = sum(1 for r in self._rows if r["exit"] > r["due"] + 1e-6)
        total_t = sum(max(0.0, r["exit"] - r["due"]) for r in self._rows)
        bays_used = sorted(set(r["bay_id"] for r in self._rows))
        self._lbl_info.setText(
            f"{n} blocks  |  {n_tardy} tardy  |  SigmaT={total_t:.1f}  "
            f"|  bays used: {[f'B{j}' for j in bays_used]}"
        )
