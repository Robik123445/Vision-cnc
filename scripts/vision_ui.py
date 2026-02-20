#!/usr/bin/env python3
"""Run the CNC vision PySide6 operator UI."""

from __future__ import annotations

import argparse

from vision.ui.app import run_ui


def main() -> int:
    """Parse args and run the Qt operator console."""

    parser = argparse.ArgumentParser(description="Run CNC Vision UI")
    parser.add_argument("--config", default="config/vision_config.json", help="Path to detector config")
    args = parser.parse_args()
    return run_ui(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
