#!/usr/bin/env python3
"""Spustenie PySide6 rozhrania pre CNC Vision."""

from __future__ import annotations

import argparse

from vision.ui.app import run_ui


def main() -> int:
    """Parse args and run the Qt operator console."""

    parser = argparse.ArgumentParser(description="Spustí rozhranie CNC Vision")
    parser.add_argument("--config", default="config/vision_config.json", help="Cesta ku konfigurácii detektora")
    args = parser.parse_args()
    return run_ui(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
