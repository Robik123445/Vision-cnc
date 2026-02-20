#!/usr/bin/env python3
"""CLI smoketest for YOLO/fallback detector and artifact export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

from vision.dataset.hard_cases import HardCaseLogger
from vision.detect.detector_api import create_detector, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image/camera smoke testing."""

    parser = argparse.ArgumentParser(description="Vision detection smoketest")
    parser.add_argument("--image", type=str, default=None, help="Path to image input")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--config", type=str, default="config/vision_config.json")
    return parser.parse_args()


def draw_overlay(frame, result):
    """Overlay binary masks for quick visual validation."""

    overlay = frame.copy()
    for mask, color in [
        (result.workpiece_mask, (0, 255, 0)),
        (result.clamp_mask, (0, 0, 255)),
        (result.hand_mask, (0, 255, 255)),
        (result.tool_mask, (255, 0, 0)),
    ]:
        if mask is None:
            continue
        overlay[mask.astype(bool)] = color
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


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
    cv2.imwrite(str(out_dir / "input.png"), frame)
    cv2.imwrite(str(out_dir / "preview.png"), preview)

    summary = {
        "source": result.source,
        "fail_reason": result.fail_reason,
        "confidences": result.confidences,
        "inference_ms": result.inference_ms,
        "image_size": result.image_size,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hard_cfg = config.get("dataset", {})
    hard_logger = HardCaseLogger(
        to_label_dir=hard_cfg.get("to_label_dir", "dataset/to_label"),
        cooldown_seconds=int(hard_cfg.get("cooldown_seconds", 5)),
    )
    reason = result.fail_reason
    if result.confidences.get("workpiece", 0.0) < config.get("yolo", {}).get("min_conf_by_class", {}).get("workpiece", 0.0):
        reason = "low_confidence_workpiece"
    saved = hard_logger.save(frame, reason, summary) if reason else None

    print(json.dumps({"run_dir": str(out_dir), "hard_case_saved": str(saved) if saved else None, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
