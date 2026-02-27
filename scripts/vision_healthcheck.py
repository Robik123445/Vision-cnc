#!/usr/bin/env python3
"""Preflight healthcheck for CNC vision engine readiness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
import numpy as np

from vision.detect.detector_api import load_config


LOGGER = logging.getLogger("vision.healthcheck")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )


def run_healthcheck(config_path: str) -> dict:
    """Run deterministic checks and return structured status report."""

    config = load_config(config_path)
    checks: dict[str, dict] = {}

    model_path = Path(config.get("yolo", {}).get("model_path", "models/workpiece-seg.pt"))
    checks["model_exists"] = {"ok": model_path.exists(), "path": str(model_path)}

    cal_cfg = config.get("calibration", {})
    intr_path = Path(cal_cfg.get("intrinsics_path", "calibration/intrinsics.json"))
    plane_path = Path(cal_cfg.get("plane_path", "calibration/plane.json"))
    checks["intrinsics_exists"] = {"ok": intr_path.exists(), "path": str(intr_path)}
    checks["plane_exists"] = {"ok": plane_path.exists(), "path": str(plane_path)}

    try:
        from vision.calib.plane import load_plane, pixel_to_mm

        if plane_path.exists():
            plane = load_plane(str(plane_path))
            x0, y0 = pixel_to_mm(0.0, 0.0, plane)
            checks["pixel_to_mm"] = {"ok": np.isfinite([x0, y0]).all().item(), "value": [x0, y0]}
        else:
            checks["pixel_to_mm"] = {"ok": False, "value": None}
    except Exception as exc:
        checks["pixel_to_mm"] = {"ok": False, "value": None, "error": str(exc)}

    try:
        from vision.detect.detector_api import create_detector

        detector = create_detector(config)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result = detector.detect(frame)
        checks["detector_contract"] = {
            "ok": isinstance(result.ok, bool) and isinstance(result.fail_reason, str) and isinstance(result.masks, dict),
            "fail_reason": result.fail_reason,
        }
    except Exception as exc:
        checks["detector_contract"] = {"ok": False, "fail_reason": "import_or_runtime_error", "error": str(exc)}

    overall = all(check["ok"] for check in checks.values())
    return {"ok": overall, "checks": checks}


def main() -> int:
    """CLI entrypoint for healthcheck script."""

    parser = argparse.ArgumentParser(description="Vision preflight healthcheck")
    parser.add_argument("--config", default="config/vision_config.json")
    parser.add_argument("--out", default="runs/healthcheck.json")
    args = parser.parse_args()

    report = run_healthcheck(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Healthcheck status: %s", "PASS" if report["ok"] else "FAIL")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
