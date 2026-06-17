"""Main window for alg_tester -- instance + algorithm selector with solution visualization."""
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QMainWindow, QStatusBar, QMessageBox, QTabWidget,
    QWidget, QVBoxLayout,
)
from PyQt6.QtCore import Qt, QTimer

from alg_tester_ui.control_panel import ControlPanel
from alg_tester_ui.bay_layout_widget import BayLayoutTab
from alg_tester_ui.solution_widget import SolutionTab


class MainWindow(QMainWindow):
    def __init__(self, settings_path: pathlib.Path):
        super().__init__()
        self.setWindowTitle("OGC2026 Algorithm Tester")
        self.resize(1500, 900)

        self._instance = None

        # -- Widgets ----------------------------------------------------------
        self.control_panel  = ControlPanel(settings_path)
        self.tab_layout     = BayLayoutTab()
        self.tab_solution   = SolutionTab()

        self._tabs = QTabWidget()
        self._tabs.addTab(self.tab_layout,   "Problem")
        self._tabs.addTab(self.tab_solution, "Solution")

        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.control_panel)
        vbox.addWidget(self._tabs, stretch=1)
        self.setCentralWidget(central)

        # -- Status bar -------------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # -- Signal connections -----------------------------------------------
        self.control_panel.btn_inst.clicked.connect(self._on_instance_selected)
        self.control_panel.btn_alg.clicked.connect(self._on_algorithm_selected)
        self.control_panel.btn_run.clicked.connect(self._on_run)
        self.tab_solution.run_finished.connect(self._on_run_finished)

        # Auto-load restored paths after the event loop starts
        QTimer.singleShot(0, self._restore_last_session)

    # -- Instance file selected -----------------------------------------------
    def _on_instance_selected(self):
        self._load_instance_file(self.control_panel.instance_file)

    def _load_instance_file(self, path: str):
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                instance = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._instance = instance
        n_bays   = len(instance.get("bays", []))
        n_blocks = len(instance.get("blocks", []))
        self.control_panel.set_instance_info(n_bays, n_blocks)

        self.tab_layout.set_instance(instance)
        self.tab_solution.set_instance(instance)

        name = instance.get("name", pathlib.Path(path).stem)
        self._status.showMessage(
            f"Loaded: {name}  |  {n_bays} bay(s)  |  {n_blocks} block(s)"
        )

    # -- Algorithm folder selected --------------------------------------------
    def _on_algorithm_selected(self):
        """Propagate the algorithm folder to SolutionTab."""
        folder = self.control_panel.algorithm_folder
        if folder:
            self.tab_solution.set_algorithm_folder(folder)
            self._status.showMessage(f"Algorithm folder: {folder}")

    # -- Run ------------------------------------------------------------------
    def _on_run(self):
        if self._instance is None:
            QMessageBox.warning(self, "No Instance", "Please load an instance first.")
            return
        if not self.control_panel.algorithm_folder:
            QMessageBox.warning(self, "No Algorithm", "Please select an algorithm folder.")
            return
        if self.tab_solution.is_running():
            return

        self._tabs.setCurrentWidget(self.tab_solution)
        self._status.showMessage("Running algorithm...")
        self.control_panel.btn_run.setEnabled(False)
        self.tab_solution.run_algorithm(self.control_panel.timelimit())

    def _restore_last_session(self):
        """Auto-load instance and algorithm folder saved from last session."""
        inst_path = self.control_panel.instance_file
        alg_folder = self.control_panel.algorithm_folder
        if inst_path:
            self._load_instance_file(inst_path)
        if alg_folder:
            self.tab_solution.set_algorithm_folder(alg_folder)

    def closeEvent(self, event):
        self.control_panel.save_settings()
        super().closeEvent(event)

    def _on_run_finished(self):
        self.control_panel.btn_run.setEnabled(True)
        self._status.showMessage(self.tab_solution._lbl_status.text())
