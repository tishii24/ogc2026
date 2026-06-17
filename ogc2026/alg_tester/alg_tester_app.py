"""alg_tester entry point.

Usage:
    uv run python alg_tester_app.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from alg_tester_ui.main_window import MainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"


def main():
    # Retina / high-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("OGC2026 Algorithm Tester")
    app.setOrganizationName("OGC2026")

    win = MainWindow(_SETTINGS_FILE)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
