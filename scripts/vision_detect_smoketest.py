#!/usr/bin/env python3
"""CLI smoketest for YOLO/fallback detector and artifact export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from vision.calib.plane import load_plane, mm_to_pixel
from vision.dataset.hard_cases import HardCaseLogger
from vision.detect.detector_api import create_detector, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image/camera smoke testing."""

    parser = argparse.ArgumentParser(description="Vision detection smoketest")
    parser.add_argument("--image", type=str, default=None, help="Path to image input")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--config", type=str, default="config/vision_config.json")
    parser.add_argument("--with_mm_grid", action="store_true", help="Overlay mm grid using plane calibration")
    return parser.parse_args()


def draw_overlay(frame: np.ndarray, result) -> np.ndarray:
    """Overlay binary masks for quick visual validation."""

    overlay = frame.copy()
    color_map = {
        "workpiece": (0, 255, 0),
        "clamp": (0, 0, 255),
        "hand": (0, 255, 255),
        "tool": (255, 0, 0),
        "safety_mask": (255, 0, 255),
    }
    for key, color in color_map.items():
        mask = result.masks.get(key)
        if mask is None:
            continue
        overlay[mask.astype(bool)] = color
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def draw_mm_grid(frame: np.ndarray, plane_path: str, step_mm: int = 50) -> np.ndarray:
    """Render mm grid projected into image for plane-calibration debug."""

    plane = load_plane(plane_path)
    out = frame.copy()
    w_mm, h_mm = plane.workspace_mm

    for x_mm in range(0, int(w_mm) + 1, step_mm):
        pts = [mm_to_pixel(x_mm, 0.0, plane), mm_to_pixel(x_mm, h_mm, plane)]
        p1, p2 = (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))
        cv2.line(out, p1, p2, (255, 255, 0), 1)
    for y_mm in range(0, int(h_mm) + 1, step_mm):
        pts = [mm_to_pixel(0.0, y_mm, plane), mm_to_pixel(w_mm, y_mm, plane)]
        p1, p2 = (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))
        cv2.line(out, p1, p2, (255, 255, 0), 1)
    return out


def main() -> int:
    """Run one-shot detection, print summary, and save artifacts."""

    args = parse_args()
    config = load_config(args.config)
    detector = create_detector(config)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Failed to read image: {args.image}")
    else:
        cam_idx = 0 if args.camera is None else args.camera
        cap = cv2.VideoCapture(cam_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Failed to read frame from camera {cam_idx}")

    result = detector.detect(frame)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    preview = draw_overlay(frame, result)
    if args.with_mm_grid:
        plane_path = config.get("calibration", {}).get("plane_path", "calibration/plane.json")
        if Path(plane_path).exists():
            preview = draw_mm_grid(preview, plane_path)

    cv2.imwrite(str(out_dir / "input.png"), frame)
    cv2.imwrite(str(out_dir / "preview.png"), preview)

    summary = {
        "ok": result.ok,
        "fail_reason": result.fail_reason,
        "confidences": result.confidences,
        "timing_ms": result.timing_ms,
        "debug": result.debug,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hard_cfg = config.get("dataset", {})
    hard_logger = HardCaseLogger(
        to_label_dir=hard_cfg.get("to_label_dir", "dataset/to_label"),
        cooldown_seconds=int(hard_cfg.get("cooldown_seconds", 5)),
    )
    reason = result.fail_reason or ""
    if result.confidences.get("workpiece", 0.0) < config.get("yolo", {}).get("min_conf_by_class", {}).get("workpiece", 0.0):
        reason = "low_confidence_workpiece"
    saved = hard_logger.save(frame, reason, summary, result.masks) if reason else None

    print(json.dumps({"run_dir": str(out_dir), "hard_case_saved": str(saved) if saved else None, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
