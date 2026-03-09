"""Entrypoint helpers for starting the PySide6 UI application."""

from __future__ import annotations

import logging
import sys


def run_ui(config_path: str = "config/vision_config.json") -> int:
    """Start Qt app and open the CNC vision operator window."""

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        raise SystemExit(
            "PySide6 nie je nainštalované. Spustite `pip install -r requirements-runtime.txt`."
        ) from exc

    from vision.ui.main_window import VisionMainWindow

    logging.getLogger("vision.ui").info("Spúšťam UI s konfiguráciou=%s", config_path)
    app = QApplication.instance() or QApplication(sys.argv)
    window = VisionMainWindow(config_path=config_path)
    window.show()
    return app.exec()
