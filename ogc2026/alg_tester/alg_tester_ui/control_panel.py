"""Top control bar -- instance file selection, algorithm folder selection, and Run."""
import json
import pathlib

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QLineEdit,
    QSizePolicy, QFileDialog, QFrame,
)
from PyQt6.QtCore import Qt

# Accent colours for the three steps
_C_INST = "#3b82f6"   # blue   – ① Instance
_C_ALG  = "#f59e0b"   # amber  – ② Algorithm
_C_RUN  = "#22c55e"   # green  – ③ Run


class ControlPanel(QWidget):
    def __init__(self, settings_path: pathlib.Path, parent=None):
        super().__init__(parent)
        self._settings_path = settings_path
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "ControlPanel { background:#1e293b; border-bottom:1px solid #0f172a; }"
            "QLabel  { color:#cbd5e1; font-size:11px; }"
            "QLineEdit { background:#0f172a; color:#e2e8f0; border:1px solid #475569;"
            "            border-radius:3px; padding:1px 4px; font-size:11px; }"
            "QPushButton { background:#334155; color:#e2e8f0; border:1px solid #475569;"
            "              border-radius:3px; padding:1px 6px; }"
            "QPushButton:hover { background:#475569; }"
            "QDoubleSpinBox { background:#0f172a; color:#e2e8f0; border:1px solid #475569;"
            "                 border-radius:3px; padding:1px 2px; font-size:11px; }"
        )

        # Root: two rows (controls + hint)
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Row 1: controls ──────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(0)
        ctrl.setContentsMargins(0, 0, 0, 0)

        # -- helpers ----------------------------------------------------------
        def _badge(char: str, color: str) -> QLabel:
            b = QLabel(char)
            b.setFixedSize(20, 20)
            b.setAlignment(Qt.AlignmentFlag.AlignCenter)
            b.setStyleSheet(
                f"QLabel {{ background:{color}; color:#fff; border-radius:10px;"
                f"          font-size:11px; font-weight:bold; }}"
            )
            return b

        def _section_lbl(text: str, color: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"QLabel {{ color:{color}; font-size:10px; font-weight:bold; }}"
            )
            return lbl

        def _vsep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setFixedWidth(1)
            f.setStyleSheet("QFrame { background:#334155; margin:0; }")
            return f

        def _section(step: str, title: str, accent: str,
                     path_w, btn_w, info_w) -> QWidget:
            """Two-row section with coloured left border, badge, and info line."""
            outer = QWidget()
            outer.setStyleSheet(
                f"QWidget#sec {{ background:transparent;"
                f"  border-left:3px solid {accent}; }}"
            )
            outer.setObjectName("sec")
            col = QVBoxLayout(outer)
            col.setContentsMargins(8, 4, 8, 3)
            col.setSpacing(2)

            top = QHBoxLayout()
            top.setSpacing(5)
            top.addWidget(_badge(step, accent))
            top.addWidget(_section_lbl(title, accent))
            top.addWidget(path_w, 1)
            top.addWidget(btn_w)
            col.addLayout(top)
            col.addWidget(info_w)
            return outer

        # -- ① Instance -------------------------------------------------------
        self.inst_path = QLineEdit()
        self.inst_path.setPlaceholderText("instance JSON")
        self.inst_path.setReadOnly(True)
        self.inst_path.setMinimumWidth(160)
        self.btn_inst = QPushButton("…")
        self.btn_inst.setFixedSize(26, 22)
        self.btn_inst.setToolTip("Select instance JSON file")
        self.btn_inst.clicked.connect(self._browse_instance)
        self._inst_info = QLabel("—  no file loaded")
        self._inst_info.setStyleSheet("color:#475569; font-size:10px;")

        ctrl.addWidget(_section("①", "Instance", _C_INST,
                                self.inst_path, self.btn_inst, self._inst_info), 2)
        ctrl.addWidget(_vsep())

        # -- ② Algorithm ------------------------------------------------------
        self.alg_path = QLineEdit()
        self.alg_path.setPlaceholderText("algorithm folder")
        self.alg_path.setReadOnly(True)
        self.alg_path.setMinimumWidth(140)
        self.btn_alg = QPushButton("…")
        self.btn_alg.setFixedSize(26, 22)
        self.btn_alg.setToolTip("Select folder containing myalgorithm.py")
        self.btn_alg.clicked.connect(self._browse_algorithm)
        self._alg_info = QLabel("—  no algorithm selected")
        self._alg_info.setStyleSheet("color:#475569; font-size:10px;")

        ctrl.addWidget(_section("②", "Algorithm", _C_ALG,
                                self.alg_path, self.btn_alg, self._alg_info), 2)
        ctrl.addWidget(_vsep())

        # -- ③ Run ------------------------------------------------------------
        run_outer = QWidget()
        run_outer.setStyleSheet(
            f"QWidget#run {{ background:transparent; border-left:3px solid {_C_RUN}; }}"
        )
        run_outer.setObjectName("run")
        run_col = QVBoxLayout(run_outer)
        run_col.setContentsMargins(8, 4, 12, 3)
        run_col.setSpacing(2)

        run_top = QHBoxLayout()
        run_top.setSpacing(5)
        run_top.addWidget(_badge("③", _C_RUN))
        run_top.addWidget(_section_lbl("Run", _C_RUN))

        tl_lbl = QLabel("Time limit:")
        tl_lbl.setStyleSheet("color:#94a3b8; font-size:10px;")
        run_top.addWidget(tl_lbl)
        self.spin_tl = QDoubleSpinBox()
        self.spin_tl.setRange(1.0, 3600.0)
        self.spin_tl.setValue(60.0)
        self.spin_tl.setDecimals(1)
        self.spin_tl.setSuffix(" s")
        self.spin_tl.setFixedWidth(72)
        run_top.addWidget(self.spin_tl)
        run_col.addLayout(run_top)

        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setFixedHeight(24)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "QPushButton { background:#1976D2; color:white; font-weight:bold;"
            "              border-radius:4px; border:none; }"
            "QPushButton:disabled { background:#1e3a5f; color:#475569; }"
            "QPushButton:hover:!disabled { background:#1565C0; }"
        )
        run_col.addWidget(self.btn_run)
        ctrl.addWidget(run_outer)
        root.addLayout(ctrl)

        # ── Row 2: hint bar ──────────────────────────────────────────────────
        hint_bar = QWidget()
        hint_bar.setStyleSheet("QWidget { background:#0f172a; }")
        hint_layout = QHBoxLayout(hint_bar)
        hint_layout.setContentsMargins(10, 2, 10, 2)
        hint_layout.setSpacing(4)

        def _hint_step(step: str, color: str, text: str):
            b = QLabel(step)
            b.setStyleSheet(
                f"QLabel {{ color:{color}; font-size:10px; font-weight:bold; }}"
            )
            t = QLabel(text)
            t.setStyleSheet("QLabel { color:#64748b; font-size:10px; }")
            return b, t

        for step, color, text, arrow in [
            ("①", _C_INST, "Select an instance JSON file", True),
            ("②", _C_ALG,  "Select the algorithm folder  (must contain myalgorithm.py)", True),
            ("③", _C_RUN,  "Set time limit and click ▶ Run  —  results appear in the Solution tab", False),
        ]:
            b, t = _hint_step(step, color, text)
            hint_layout.addWidget(b)
            hint_layout.addWidget(t)
            if arrow:
                arr = QLabel("→")
                arr.setStyleSheet("QLabel { color:#334155; font-size:10px; }")
                hint_layout.addWidget(arr)
        hint_layout.addStretch()
        root.addWidget(hint_bar)

        # -- Internal state ---------------------------------------------------
        self._inst_file: str = ""
        self._alg_folder: str = ""
        self._last_inst_dir: str = str(pathlib.Path.home())
        self._last_alg_dir: str = str(pathlib.Path.home())

        self._restore_settings()


    # -- Settings persistence -------------------------------------------------
    def _restore_settings(self):
        if not self._settings_path.is_file():
            return
        try:
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        inst = data.get("last_instance_file", "")
        alg  = data.get("last_alg_folder", "")
        tl   = data.get("timelimit")
        if tl is not None:
            try:
                self.spin_tl.setValue(float(tl))
            except Exception:
                pass
        if inst:
            inst_path = pathlib.Path(inst)
            if not inst_path.is_absolute():
                inst_path = self._settings_path.parent / inst_path
            inst = str(inst_path)
        if inst and pathlib.Path(inst).is_file():
            self._last_inst_dir = str(pathlib.Path(inst).parent)
            self._apply_instance_path(inst)
        if alg and pathlib.Path(alg).is_dir():
            self._last_alg_dir = alg
            self._apply_alg_folder(alg)

    def save_settings(self):
        data = {
            "last_instance_file": self._inst_file,
            "last_alg_folder":    self._alg_folder,
            "timelimit":          self.spin_tl.value(),
        }
        self._settings_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- Internal apply helpers -----------------------------------------------
    def _apply_instance_path(self, path: str):
        self._inst_file = path
        self.inst_path.setText(pathlib.Path(path).name)
        self.inst_path.setToolTip(path)
        self._inst_info.setText(str(pathlib.Path(path).parent))
        self._inst_info.setStyleSheet("color:#64748b; font-size:10px;")
        self._refresh_run_btn()

    def _apply_alg_folder(self, folder: str):
        myalg = pathlib.Path(folder) / "myalgorithm.py"
        if not myalg.exists():
            self._alg_info.setText("myalgorithm.py not found!")
            self._alg_info.setStyleSheet("color:#dc2626; font-size:10px;")
            return
        self._alg_folder = folder
        self.alg_path.setText(pathlib.Path(folder).name)
        self.alg_path.setToolTip(folder)
        self._alg_info.setText(folder)
        self._alg_info.setStyleSheet("color:#16a34a; font-size:10px;")
        self._refresh_run_btn()

    # -- Browser callbacks ----------------------------------------------------
    def _browse_instance(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Instance JSON", self._last_inst_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._last_inst_dir = str(pathlib.Path(path).parent)
            self._apply_instance_path(path)

    def _browse_algorithm(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Algorithm Folder", self._last_alg_dir
        )
        if folder:
            self._last_alg_dir = folder
            self._apply_alg_folder(folder)

    def _refresh_run_btn(self):
        self.btn_run.setEnabled(bool(self._inst_file) and bool(self._alg_folder))

    # -- Public accessors -----------------------------------------------------
    @property
    def instance_file(self) -> str:
        return self._inst_file

    @property
    def algorithm_folder(self) -> str:
        return self._alg_folder

    def timelimit(self) -> float:
        return self.spin_tl.value()

    def set_instance_info(self, n_bays: int, n_blocks: int):
        self._inst_info.setText(f"{n_bays} bay(s)  |  {n_blocks} block(s)")
        self._inst_info.setStyleSheet("color:#16a34a; font-size:10px;")

