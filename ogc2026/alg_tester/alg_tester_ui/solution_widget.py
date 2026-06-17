"""
Solution visualization tab for alg_tester.

Layout (vertical splitter, 3 panels):
  +- status + time/event selectors -----------------------+
  +- Bay canvases  (scrollable)       [panel 1] ----------+
  +- Gantt chart   (scrollable)       [panel 2] ----------+
  +- Algorithm output console         [panel 3] ----------+

Real-time stdout streaming:
  The algorithm runs as a subprocess launched via subprocess.Popen with
  stdout=PIPE + stderr=STDOUT.  A reader QThread drains the pipe line by
  line and emits each line as a Qt signal so the console updates live.
  The result (solution dict) is written as JSON to a temp file and read
  back by the main thread on completion.
"""

import sys
import importlib.util
import subprocess
import pathlib
import traceback
import time
import json
import tempfile
import os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QScrollArea, QPushButton, QSplitter,
    QPlainTextEdit, QFrame, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor

from alg_tester_ui.bay_layout_widget import (
    BayCanvas, BlockItem, _auto_scale,
    _layer_gray, _SCALE_MIN, _SCALE_MAX, _SCALE_STEP,
)
from alg_tester_ui.gantt_widget import GanttCanvas, _bay_color, _bay_color_dark
from utils import _resolve_layers

# -----------------------------------------------------------------------------
# Subprocess runner script (written to a temp file and executed by Popen)
# -----------------------------------------------------------------------------
_RUNNER_SCRIPT = """\
import sys, json, pathlib, traceback

prob_info_file = sys.argv[1]
timelimit      = float(sys.argv[2])
alg_folder     = sys.argv[3]
out_file       = sys.argv[4]

folder = str(pathlib.Path(alg_folder).resolve())
if folder not in sys.path:
    sys.path.insert(0, folder)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "myalgorithm", pathlib.Path(folder) / "myalgorithm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with open(prob_info_file, "r", encoding="utf-8") as f:
        prob_info = json.load(f)

    sol = mod.algorithm(prob_info, timelimit)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ok": True, "solution": sol}, f)
except Exception:
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ok": False, "traceback": traceback.format_exc()}, f)
"""


# -----------------------------------------------------------------------------
# StdoutReader -- drains the subprocess stdout pipe in a background thread
# -----------------------------------------------------------------------------
class _StdoutReader(QThread):
    line_ready = pyqtSignal(str)
    finished_reading = pyqtSignal()

    def __init__(self, proc: subprocess.Popen):
        super().__init__()
        self._proc = proc

    def run(self):
        try:
            for raw in self._proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    line = repr(raw)
                self.line_ready.emit(line)
        except Exception:
            pass
        self.finished_reading.emit()


# -----------------------------------------------------------------------------
# AlgorithmWorker -- launches the subprocess and waits for it
# -----------------------------------------------------------------------------
class AlgorithmWorker(QThread):
    done       = pyqtSignal(object)              # solution dict on success
    error      = pyqtSignal(str)                 # error message string on failure
    proc_ready = pyqtSignal(object)              # emits the Popen object once started

    def __init__(self, prob_info_file: str, out_file: str,
                 timelimit: float, alg_folder: str, runner_file: str):
        super().__init__()
        self._prob_info_file = prob_info_file
        self._out_file       = out_file
        self._timelimit      = timelimit
        self._alg_folder     = alg_folder
        self._runner_file    = runner_file
        self._proc: Optional[subprocess.Popen] = None

    def run(self):
        try:
            _extra = {}
            if sys.platform == "win32":
                _extra["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._proc = subprocess.Popen(
                [sys.executable, "-u", self._runner_file,
                 self._prob_info_file, str(self._timelimit),
                 self._alg_folder, self._out_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **_extra,
            )
        except Exception:
            self.error.emit(traceback.format_exc())
            return

        self.proc_ready.emit(self._proc)
        self._proc.wait()

        # stdout already drained by _StdoutReader; just read result file
        try:
            with open(self._out_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("ok"):
                self.done.emit(result["solution"])
            else:
                self.error.emit(result.get("traceback", "Unknown error"))
        except Exception as e:
            self.error.emit(f"Could not read result file: {e}")

    def kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Gantt panel (inline, not a separate tab)
# -----------------------------------------------------------------------------
class _GanttPanel(QWidget):
    """Compact Gantt panel embedded below the bay canvases."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instance: Optional[dict] = None
        self._rows: list[dict] = []
        self._px_per_unit: float = 10.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        # -- Toolbar -----------------------------------------------------------
        tb = QHBoxLayout()
        tb.setSpacing(6)
        lbl = QLabel("Gantt")
        lbl.setStyleSheet("font-weight:bold; color:#334155;")
        tb.addWidget(lbl)

        tb.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Block ID", "Bay -> Entry", "Entry time", "Due date", "Tardiness v"])
        self._sort_combo.setFixedWidth(120)
        self._sort_combo.currentIndexChanged.connect(self._refresh)
        tb.addWidget(self._sort_combo)

        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_out.setFixedSize(22, 22)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        tb.addWidget(self._btn_zoom_out)
        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedSize(22, 22)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        tb.addWidget(self._btn_zoom_in)
        self._btn_zoom_fit = QPushButton("Fit")
        self._btn_zoom_fit.setFixedWidth(32)
        self._btn_zoom_fit.clicked.connect(self._zoom_fit)
        tb.addWidget(self._btn_zoom_fit)

        tb.addStretch()

        # Legend (filled dynamically)
        self._legend_layout = QHBoxLayout()
        self._legend_layout.setSpacing(4)
        tb.addLayout(self._legend_layout)

        outer.addLayout(tb)

        # -- Canvas in scroll area ---------------------------------------------
        self._canvas = GanttCanvas()
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setWidget(self._canvas)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(self._scroll, stretch=1)

    # -- Public ----------------------------------------------------------------
    def set_solution(self, instance: dict, solution: dict):
        self._instance = instance
        self._rows = self._build_rows(instance, solution)
        self._update_legend()
        self._zoom_fit()

    # -- Internals -------------------------------------------------------------
    def _build_rows(self, instance: dict, solution: dict) -> list[dict]:
        blocks_data = instance["blocks"]
        ops = solution.get("operations", {})
        entry_map: dict[int, dict] = {}
        exit_map:  dict[int, float] = {}
        for t_str, ops_at_t in ops.items():
            t = int(t_str)
            for op in ops_at_t:
                bid = op["block_id"]
                if op["type"] == "ENTRY":
                    entry_map[bid] = {"bay_id": op["bay_id"], "entry": float(t)}
                elif op["type"] == "EXIT":
                    exit_map[bid] = float(t)
        rows = []
        for bid, ed in entry_map.items():
            blk = blocks_data[bid]
            rows.append({
                "block_id": bid,
                "bay_id":   ed["bay_id"],
                "entry":    ed["entry"],
                "exit":     exit_map.get(bid, ed["entry"]),
                "due":      float(blk.get("due_date", 0)),
                "release":  float(blk.get("release_time", 0)),
                "workload": blk.get("workload", 0),
            })
        return rows

    def _sorted_rows(self) -> list[dict]:
        key = self._sort_combo.currentText()
        rows = list(self._rows)
        if key == "Block ID":
            rows.sort(key=lambda r: r["block_id"])
        elif key == "Bay -> Entry":
            rows.sort(key=lambda r: (r["bay_id"], r["entry"]))
        elif key == "Entry time":
            rows.sort(key=lambda r: r["entry"])
        elif key == "Due date":
            rows.sort(key=lambda r: r["due"])
        elif key == "Tardiness v":
            rows.sort(key=lambda r: -max(0.0, r["exit"] - r["due"]))
        return rows

    def _refresh(self):
        if not self._rows:
            return
        rows = self._sorted_rows()
        t_min = min(r["entry"] for r in rows)
        t_max = max(r["exit"] for r in rows)
        span  = max(1.0, t_max - t_min)
        t_min = max(0.0, t_min - span * 0.02)
        t_max = t_max + span * 0.03
        self._canvas.set_data(rows, t_min, t_max, self._px_per_unit)

    def _zoom_fit(self):
        if not self._rows:
            return
        available = max(300, self._scroll.viewport().width() - 75)
        rows = self._sorted_rows()
        span = max(1.0, max(r["exit"] for r in rows) - min(r["entry"] for r in rows))
        self._px_per_unit = available / span
        self._refresh()

    def _zoom_in(self):
        self._px_per_unit = min(self._px_per_unit * 1.4, 200.0)
        self._refresh()

    def _zoom_out(self):
        self._px_per_unit = max(self._px_per_unit / 1.4, 2.0)
        self._refresh()

    def _update_legend(self):
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self._instance is None:
            return
        n_bays = len(self._instance.get("bays", []))
        for j in range(n_bays):
            col = _bay_color(j, 255)
            border = _bay_color_dark(j)
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background:{col.name()}; border:1px solid {border.name()};"
            )
            lbl = QLabel(f"Bay {j}")
            lbl.setStyleSheet("font-size:10px; color:#334155;")
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)
            hl.addWidget(swatch)
            hl.addWidget(lbl)
            self._legend_layout.addWidget(w)


# -----------------------------------------------------------------------------
# SolutionTab
# -----------------------------------------------------------------------------
class SolutionTab(QWidget):
    run_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._instance:   Optional[dict] = None
        self._solution:   Optional[dict] = None
        self._worker:     Optional[AlgorithmWorker] = None
        self._reader:     Optional[_StdoutReader]   = None
        self._alg_folder: str = ""

        # temp files for IPC
        self._tmp_prob:   Optional[tempfile.NamedTemporaryFile] = None
        self._tmp_runner: Optional[tempfile.NamedTemporaryFile] = None
        self._tmp_out:    Optional[str] = None

        self._op_times: list[int] = []
        self._ops_at:   dict[int, list[dict]] = {}

        self._bay_canvases: list[BayCanvas] = []
        self._bay_scale: float = 5.0
        self._visible_layers: Optional[set] = None
        self._layer_checkboxes: list[QCheckBox] = []

        self._build_ui()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # -- Status row --------------------------------------------------------
        self._lbl_status = QLabel("No instance loaded")
        self._lbl_status.setStyleSheet("color:#64748b;")
        outer.addWidget(self._lbl_status)

        # -- Time / event selectors --------------------------------------------
        sel = QHBoxLayout()
        sel.setSpacing(4)

        sel.addWidget(QLabel("Time point:"))
        self._btn_time_prev = QPushButton("◀")
        self._btn_time_prev.setFixedWidth(28)
        self._btn_time_prev.clicked.connect(self._on_time_prev)
        sel.addWidget(self._btn_time_prev)
        self._time_combo = QComboBox()
        self._time_combo.setMinimumWidth(150)
        self._time_combo.currentIndexChanged.connect(self._on_time_changed)
        sel.addWidget(self._time_combo)
        self._btn_time_next = QPushButton("▶")
        self._btn_time_next.setFixedWidth(28)
        self._btn_time_next.clicked.connect(self._on_time_next)
        sel.addWidget(self._btn_time_next)

        sel.addSpacing(12)

        sel.addWidget(QLabel("Event:"))
        self._btn_event_prev = QPushButton("◀")
        self._btn_event_prev.setFixedWidth(28)
        self._btn_event_prev.clicked.connect(self._on_event_prev)
        sel.addWidget(self._btn_event_prev)
        self._event_combo = QComboBox()
        self._event_combo.setMinimumWidth(300)
        self._event_combo.currentIndexChanged.connect(self._on_event_changed)
        sel.addWidget(self._event_combo)
        self._btn_event_next = QPushButton("▶")
        self._btn_event_next.setFixedWidth(28)
        self._btn_event_next.clicked.connect(self._on_event_next)
        sel.addWidget(self._btn_event_next)

        sel.addStretch()
        outer.addLayout(sel)

        # -- Layer checkboxes + Scale controls --------------------------------
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
        ctrl.addSpacing(12)
        _toggle_ss = (
            "QPushButton { padding: 1px 8px; border-radius: 4px;"
            " border: 1px solid #888; background: #4a90d9; color: #fff; font-weight: bold; }"
            "QPushButton:checked { background: #4a90d9; color: #fff; }"
            "QPushButton:!checked { background: #555; color: #999; font-weight: normal; }"
        )
        self._btn_toggle_gantt = QPushButton("Gantt")
        self._btn_toggle_gantt.setCheckable(True)
        self._btn_toggle_gantt.setChecked(True)
        self._btn_toggle_gantt.setFixedHeight(22)
        self._btn_toggle_gantt.setStyleSheet(_toggle_ss)
        self._btn_toggle_gantt.clicked.connect(self._toggle_gantt)
        ctrl.addWidget(self._btn_toggle_gantt)

        self._btn_toggle_output = QPushButton("Output")
        self._btn_toggle_output.setCheckable(True)
        self._btn_toggle_output.setChecked(True)
        self._btn_toggle_output.setFixedHeight(22)
        self._btn_toggle_output.setStyleSheet(_toggle_ss)
        self._btn_toggle_output.clicked.connect(self._toggle_output)
        ctrl.addWidget(self._btn_toggle_output)

        ctrl.addStretch()
        self._info_label = QLabel("")
        ctrl.addWidget(self._info_label)
        outer.addLayout(ctrl)

        # -- Panel 1: bay canvases (scrollable) -------------------------------
        self._bays_scroll = QScrollArea()
        self._bays_scroll.setWidgetResizable(True)
        self._bays_container = QWidget()
        self._bays_layout = QVBoxLayout(self._bays_container)
        self._bays_layout.setContentsMargins(4, 4, 4, 4)
        self._bays_layout.setSpacing(8)
        self._bays_layout.addStretch()
        self._bays_scroll.setWidget(self._bays_container)

        # -- Panel 2: Gantt ----------------------------------------------------
        self._gantt_panel = _GanttPanel()

        # -- Panel 3: Algorithm output console --------------------------------
        self._console_frame = console_frame = QFrame()
        console_frame.setFrameShape(QFrame.Shape.StyledPanel)
        console_vbox = QVBoxLayout(console_frame)
        console_vbox.setContentsMargins(4, 4, 4, 4)
        console_vbox.setSpacing(2)

        console_hdr = QHBoxLayout()
        console_hdr.addWidget(QLabel("Algorithm Output"))
        console_hdr.addStretch()
        self._btn_clear_log = QPushButton("Clear")
        self._btn_clear_log.setFixedHeight(22)
        self._btn_clear_log.clicked.connect(self._clear_log)
        console_hdr.addWidget(self._btn_clear_log)
        console_vbox.addLayout(console_hdr)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(10000)
        font = QFont()
        font.setFamilies(["Menlo", "Consolas", "D2Coding", "Courier New"])
        font.setPointSize(11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(font)
        self._log.setStyleSheet(
            "QPlainTextEdit { background:#1e1e2e; color:#cdd6f4; border:none; }"
        )
        console_vbox.addWidget(self._log)

        # -- 3-way vertical splitter -------------------------------------------
        self._main_splitter = splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._bays_scroll)
        splitter.addWidget(self._gantt_panel)
        splitter.addWidget(console_frame)
        splitter.setSizes([380, 220, 160])
        splitter.setChildrenCollapsible(True)
        outer.addWidget(splitter, stretch=1)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------
    def set_instance(self, instance: dict):
        self._instance = instance
        self._solution = None
        self._lbl_status.setText("Instance loaded -- select an algorithm and press Run")
        self._lbl_status.setStyleSheet("color:#334155;")
        self._clear_solution_ui()

    def set_algorithm_folder(self, folder: str):
        self._alg_folder = folder

    def run_algorithm(self, timelimit: float):
        if self._instance is None or not self._alg_folder:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._lbl_status.setText("Running algorithm...")
        self._lbl_status.setStyleSheet("color:#2563eb;")
        self._log.clear()

        # Write prob_info to a temp file so the subprocess can read it
        self._tmp_prob = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(self._instance, self._tmp_prob)
        self._tmp_prob.flush()
        prob_path = self._tmp_prob.name

        # Write the runner script to a temp file
        self._tmp_runner = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        self._tmp_runner.write(_RUNNER_SCRIPT)
        self._tmp_runner.flush()
        runner_path = self._tmp_runner.name

        # Output result file
        fd, self._tmp_out = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        self._append_output(
            f"[{time.strftime('%H:%M:%S')}] ► Algorithm started  "
            f"(timelimit={timelimit:.1f}s)\n"
            f"{'-' * 60}\n"
        )

        worker = AlgorithmWorker(prob_path, self._tmp_out, timelimit,
                                  self._alg_folder, runner_path)
        worker.done.connect(self._on_run_done)
        worker.error.connect(self._on_run_error)
        worker.finished.connect(self._on_worker_finished)

        # Stdout reader -- drains pipe and forwards lines live
        reader = _StdoutReader(worker._proc if False else None)  # will be wired below

        # We need the Popen object, which is created inside worker.run().
        # So we wire the reader after the process starts: use a tiny wrapper
        # that creates Popen here instead.
        worker.proc_ready.connect(self._on_proc_ready)
        self._worker = worker
        self._worker.start()

    def _on_proc_ready(self, proc):
        """Called from the worker thread via Qt signal once Popen is alive."""
        self._reader = _StdoutReader(proc)
        self._reader.line_ready.connect(self._append_output)
        self._reader.start()

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # -------------------------------------------------------------------------
    # Worker callbacks
    # -------------------------------------------------------------------------
    def _on_worker_finished(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        # Clean up temp files
        for attr in ("_tmp_prob", "_tmp_runner"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    os.unlink(f.name)
                except Exception:
                    pass
                setattr(self, attr, None)
        self.run_finished.emit()

    def _on_run_done(self, solution: dict):
        self._solution = solution

        from utils import check_feasibility
        result = check_feasibility(self._instance, solution)

        n_assigned = sum(1 for ops in solution.get("operations", {}).values()
                         for op in ops if op["type"] == "ENTRY")
        n_total = len(self._instance.get("blocks", []))
        elapsed_str = f"[{time.strftime('%H:%M:%S')}]"

        if result["feasible"]:
            self._lbl_status.setText(
                f"Feasible  |  {n_assigned}/{n_total} blocks  |  "
                f"obj={result['objective']:.2f}  "
                f"(T={result['obj1']:.1f}  L={result['obj2']:.1f}  P={result['obj3']:.1f})"
            )
            self._lbl_status.setStyleSheet("color:#16a34a; font-weight:bold;")
            self._append_output(
                f"{'-' * 60}\n"
                f"{elapsed_str} ■ Done\n"
                f"[Feasibility] FEASIBLE  ({n_assigned}/{n_total} blocks)\n"
                f"  objective = {result['objective']:.4f}\n"
                f"  T (tardiness)       = {result['obj1']:.4f}\n"
                f"  L (load imbalance)  = {result['obj2']:.4f}\n"
                f"  P (bay preference)  = {result['obj3']:.4f}\n"
            )
        else:
            self._lbl_status.setText(
                f"Infeasible (stage {result['stage']})  |  {n_assigned}/{n_total} blocks"
            )
            self._lbl_status.setStyleSheet("color:#dc2626; font-weight:bold;")
            lines = [
                f"{'-' * 60}\n",
                f"{elapsed_str} ■ Done\n",
                f"[Feasibility] INFEASIBLE  (failed at stage {result['stage']},"
                f" {n_assigned}/{n_total} blocks)\n",
            ]
            for v in result.get("violations", []):
                lines.append(f"  * {v}\n")
            self._append_output("".join(lines))

        self._load_operations(solution)
        self._rebuild_bay_canvases()
        self._populate_time_combo()
        self._gantt_panel.set_solution(self._instance, solution)

    def _on_run_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg[:160]}")
        self._lbl_status.setStyleSheet("color:#dc2626;")
        self._lbl_status.setToolTip(msg)
        self._append_output(f"[ERROR]\n{msg}\n")

    # -------------------------------------------------------------------------
    # Console helpers
    # -------------------------------------------------------------------------
    def _append_output(self, text: str):
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertPlainText(text)
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_log(self):
        self._log.clear()

    # -------------------------------------------------------------------------
    # Operations helpers
    # -------------------------------------------------------------------------
    def _load_operations(self, solution: dict):
        raw: dict = solution.get("operations", {})
        self._ops_at = {int(k): v for k, v in raw.items()}
        self._op_times = sorted(self._ops_at.keys())

    # -------------------------------------------------------------------------
    # Combo population
    # -------------------------------------------------------------------------
    def _populate_time_combo(self):
        self._time_combo.blockSignals(True)
        self._time_combo.clear()
        for t in self._op_times:
            ops = self._ops_at[t]
            n = len(ops)
            self._time_combo.addItem(f"t = {t}  ({n} op{'s' if n != 1 else ''})")
        self._time_combo.blockSignals(False)
        if self._op_times:
            self._time_combo.setCurrentIndex(0)
            self._on_time_changed(0)

    def _on_time_prev(self):
        idx = self._time_combo.currentIndex()
        if idx > 0:
            self._time_combo.setCurrentIndex(idx - 1)

    def _on_time_next(self):
        idx = self._time_combo.currentIndex()
        if idx < self._time_combo.count() - 1:
            self._time_combo.setCurrentIndex(idx + 1)

    def _on_event_prev(self):
        idx = self._event_combo.currentIndex()
        if idx > 0:
            self._event_combo.setCurrentIndex(idx - 1)

    def _on_event_next(self):
        idx = self._event_combo.currentIndex()
        if idx < self._event_combo.count() - 1:
            self._event_combo.setCurrentIndex(idx + 1)

    def _on_time_changed(self, idx: int):
        self._event_combo.blockSignals(True)
        self._event_combo.clear()
        if idx < 0 or idx >= len(self._op_times):
            self._event_combo.blockSignals(False)
            return
        t = self._op_times[idx]
        ops = self._ops_at[t]
        for op_idx, op in enumerate(ops):
            kind = op["type"]
            bid  = op["block_id"]
            bay  = op["bay_id"]
            prefix = f"[{op_idx + 1}/{len(ops)}]"
            label = f"{prefix} {kind}  Block {bid}  Bay {bay}"
            if self._instance:
                blk_data = self._instance["blocks"][bid]
                due  = blk_data.get("due_date", "?")
                proc = blk_data.get("processing_time", "?")
                label += f"  (due={due}, proc={proc})"
            self._event_combo.addItem(label)
        self._event_combo.blockSignals(False)
        if ops:
            self._event_combo.setCurrentIndex(0)
            self._on_event_changed(0)

    def _on_event_changed(self, idx: int):
        t_idx = self._time_combo.currentIndex()
        if t_idx < 0 or t_idx >= len(self._op_times):
            return
        t = self._op_times[t_idx]
        ops = self._ops_at[t]
        if idx < 0 or idx >= len(ops):
            return
        self._render_bay_state(t, ops, idx)

    # -------------------------------------------------------------------------
    # Bay canvas helpers
    # -------------------------------------------------------------------------
    def _rebuild_bay_canvases(self):
        for bc in self._bay_canvases:
            bc.setParent(None)
            bc.deleteLater()
        self._bay_canvases.clear()
        while self._bays_layout.count():
            self._bays_layout.takeAt(0)

        if self._instance is None:
            return

        bays = self._instance["bays"]
        bay_w = max((b.get("width",  200) for b in bays), default=200)
        bay_h = max((b.get("height",  60) for b in bays), default=60)
        self._bay_scale = _auto_scale(bay_w, bay_h)

         # Rebuild layer checkboxes
        max_layers = max(
            (len(_resolve_layers(o["layers"]))
             for blk in self._instance["blocks"]
             for o in blk["shape"]),
            default=1,
        )
        self._rebuild_layer_checkboxes(max_layers)

        n_blocks = len(self._instance["blocks"])
        n_bays   = len(bays)
        self._info_label.setText(
            f"{n_blocks} block(s)  |  {n_bays} bay(s)  |  scale={self._bay_scale:.1f}"
        )

        for i, bay in enumerate(bays):
            canvas = BayCanvas(bay["width"], bay["height"], bay_idx=i,
                               bay_id=bay.get("id", i + 1), read_only=True)
            canvas.set_scale(self._bay_scale)
            canvas.visible_layers = self._visible_layers
            self._bay_canvases.append(canvas)
            self._bays_layout.addWidget(canvas)
        self._bays_layout.addStretch()
        self._bays_container.adjustSize()

    def _render_bay_state(self, t: int, ops: list[dict], op_idx: int):
        if self._solution is None or self._instance is None:
            return

        cur_op   = ops[op_idx]
        cur_kind = cur_op["type"]
        cur_bid  = cur_op["block_id"]

        done_exit:  set[int] = set()
        done_entry: set[int] = set()

        for prev_t in self._op_times:
            if prev_t > t:
                break
            prev_ops = self._ops_at[prev_t]
            for oi, op in enumerate(prev_ops):
                if prev_t == t and oi > op_idx:
                    break
                if prev_t == t and oi == op_idx:
                    if op["type"] == "ENTRY":
                        done_entry.add(op["block_id"])
                    break
                if op["type"] == "EXIT":
                    done_exit.add(op["block_id"])
                else:
                    done_entry.add(op["block_id"])

        stably_present = done_entry - done_exit

        assign_by_id: dict[int, dict] = {}
        for _t_str, _ops in self._solution.get("operations", {}).items():
            _t = float(int(_t_str))
            for _op in _ops:
                if _op["type"] == "ENTRY":
                    assign_by_id[_op["block_id"]] = {
                        "block_id":   _op["block_id"],
                        "bay_id":     _op["bay_id"],
                        "x":          _op.get("x", 0.0),
                        "y":          _op.get("y", 0.0),
                        "orient_idx": _op.get("orient_idx", 0),
                        "entry_time": _t,
                    }

        for bc in self._bay_canvases:
            bc.blocks.clear()
            bc.exiting_block_ids = set()
            bc.entering_block_id = -1
            bc.failed_op_block_id = -1
            bc.obstruction_block_ids = set()

        def _add_block(bid: int):
            a = assign_by_id.get(bid)
            if a is None:
                return None
            bay_id = a["bay_id"]
            if bay_id >= len(self._bay_canvases):
                return None
            blk_data = self._instance["blocks"][bid]
            item = BlockItem(bid, blk_data)
            item.orient_idx = a["orient_idx"]
            item.x = float(a["x"])
            item.y = float(a["y"])
            self._bay_canvases[bay_id].blocks.append(item)
            return bay_id

        for bid in stably_present:
            _add_block(bid)

        if cur_kind == "EXIT":
            bay_id = _add_block(cur_bid)
            if bay_id is not None and bay_id < len(self._bay_canvases):
                self._bay_canvases[bay_id].exiting_block_ids.add(cur_bid)
                try:
                    from utils import Bay as UtilsBay, Block as UtilsBlock, check_exit
                    bay_data = self._instance["bays"][bay_id]
                    util_bay = UtilsBay(width=int(bay_data["width"]), height=int(bay_data["height"]))
                    present_ids = stably_present | {cur_bid}
                    util_blocks = [
                        UtilsBlock(block_id=bid2,
                                   block_data=self._instance["blocks"][bid2],
                                   x=int(round(assign_by_id[bid2]["x"])),
                                   y=int(round(assign_by_id[bid2]["y"])),
                                   orient_idx=assign_by_id[bid2]["orient_idx"])
                        for bid2 in present_ids
                        if assign_by_id.get(bid2, {}).get("bay_id") == bay_id
                    ]
                    util_target = UtilsBlock(block_id=cur_bid,
                                             block_data=self._instance["blocks"][cur_bid],
                                             x=int(round(assign_by_id[cur_bid]["x"])),
                                             y=int(round(assign_by_id[cur_bid]["y"])),
                                             orient_idx=assign_by_id[cur_bid]["orient_idx"])
                    obs = check_exit(util_bay, util_blocks, util_target)
                    if obs:
                        self._bay_canvases[bay_id].failed_op_block_id = cur_bid
                        self._bay_canvases[bay_id].obstruction_block_ids = {
                            o.existing_block.block_id for o in obs
                            if o.existing_block.block_id != cur_bid
                        }
                except Exception:
                    pass
        else:
            a = assign_by_id.get(cur_bid)
            if a is not None:
                bay_id = a["bay_id"]
                if bay_id < len(self._bay_canvases):
                    self._bay_canvases[bay_id].entering_block_id = cur_bid
                    try:
                        from utils import Bay as UtilsBay, Block as UtilsBlock, check_entry
                        bay_data = self._instance["bays"][bay_id]
                        util_bay = UtilsBay(width=int(bay_data["width"]), height=int(bay_data["height"]))
                        util_present = [
                            UtilsBlock(block_id=bid2,
                                       block_data=self._instance["blocks"][bid2],
                                       x=int(round(assign_by_id[bid2]["x"])),
                                       y=int(round(assign_by_id[bid2]["y"])),
                                       orient_idx=assign_by_id[bid2]["orient_idx"])
                            for bid2 in stably_present
                            if bid2 != cur_bid
                            and assign_by_id.get(bid2, {}).get("bay_id") == bay_id
                        ]
                        util_new = UtilsBlock(block_id=cur_bid,
                                              block_data=self._instance["blocks"][cur_bid],
                                              x=int(round(a["x"])),
                                              y=int(round(a["y"])),
                                              orient_idx=a["orient_idx"])
                        obs = check_entry(util_bay, util_present, util_new)
                        if obs:
                            self._bay_canvases[bay_id].failed_op_block_id = cur_bid
                            self._bay_canvases[bay_id].obstruction_block_ids = {
                                o.existing_block.block_id for o in obs
                                if o.existing_block.block_id != cur_bid
                            }
                    except Exception:
                        pass

        for bc in self._bay_canvases:
            bc.mark_dirty()

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Panel toggle
    # -------------------------------------------------------------------------
    def _toggle_gantt(self):
        visible = self._btn_toggle_gantt.isChecked()
        self._gantt_panel.setVisible(visible)

    def _toggle_output(self):
        visible = self._btn_toggle_output.isChecked()
        self._console_frame.setVisible(visible)

    # -------------------------------------------------------------------------
    # Layer checkboxes / scale
    # -------------------------------------------------------------------------
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
        for bc in self._bay_canvases:
            bc.visible_layers = self._visible_layers
            bc.mark_dirty()

    def _adjust_scale(self, delta: int):
        self._bay_scale = max(_SCALE_MIN, min(_SCALE_MAX,
                              self._bay_scale + delta * _SCALE_STEP))
        for bc in self._bay_canvases:
            bc.set_scale(self._bay_scale)
        self._bays_container.adjustSize()
        if self._instance:
            n_blocks = len(self._instance["blocks"])
            n_bays   = len(self._instance["bays"])
            self._info_label.setText(
                f"{n_blocks} block(s)  |  {n_bays} bay(s)  |  scale={self._bay_scale:.1f}"
            )

    # -------------------------------------------------------------------------
    # Clear UI state
    # -------------------------------------------------------------------------
    def _clear_solution_ui(self):
        self._op_times = []
        self._ops_at = {}
        self._time_combo.clear()
        self._event_combo.clear()
        for bc in self._bay_canvases:
            bc.blocks.clear()
            bc.mark_dirty()
