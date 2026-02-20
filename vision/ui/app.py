"""Entrypoint helpers for starting the PySide6 UI application."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from vision.ui.main_window import VisionMainWindow


def run_ui(config_path: str = "config/vision_config.json") -> int:
    """Start Qt app and open the CNC vision operator window."""

    logging.getLogger("vision.ui").info("Starting UI with config=%s", config_path)
    app = QApplication(sys.argv)
    window = VisionMainWindow(config_path=config_path)
    window.show()
    return app.exec()
