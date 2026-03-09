#!/usr/bin/env python3
"""CLI smoketest for YOLO/fallback detector and artifact export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from vision.dataset.hard_cases import HardCaseLogger
from vision.detect.detector_api import create_detector, load_config
from vision.ui.image_overlay import compose_overlay


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image/camera smoke testing."""

    parser = argparse.ArgumentParser(description="Vision detection smoketest")
    parser.add_argument("--image", type=str, default=None, help="Path to image input")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--config", type=str, default="config/vision_config.json")
    parser.add_argument("--with_mm_grid", action="store_true", help="Overlay mm grid using plane calibration")
    return parser.parse_args()


def draw_mm_grid(frame, plane_path: str, step_mm: int = 50):
    """Render a projected mm grid into the preview image."""

    import cv2  # type: ignore

    from vision.calib.plane import load_plane, mm_to_pixel

    plane = load_plane(plane_path)
    out = frame.copy()
    width_mm, height_mm = plane.workspace_mm

    for x_mm in range(0, int(width_mm) + 1, step_mm):
        start = mm_to_pixel(x_mm, 0.0, plane)
        end = mm_to_pixel(x_mm, height_mm, plane)
        cv2.line(out, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 1)
    for y_mm in range(0, int(height_mm) + 1, step_mm):
        start = mm_to_pixel(0.0, y_mm, plane)
        end = mm_to_pixel(width_mm, y_mm, plane)
        cv2.line(out, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 1)
    return out


def main() -> int:
    """Run one-shot detection, print summary, and save artifacts."""

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "OpenCV is not installed. Run `pip install -r requirements-runtime.txt`."
        ) from exc

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
        if not ok or frame is None:
            raise SystemExit(f"Failed to read frame from camera {cam_idx}")

    result = detector.detect(frame)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    preview = compose_overlay(frame, result)
    if args.with_mm_grid:
        plane_path = config.get("calibration", {}).get("plane_path", "calibration/plane.json")
        if Path(plane_path).exists():
            preview = draw_mm_grid(preview, plane_path)

    cv2.imwrite(str(out_dir / "input.png"), frame)
    cv2.imwrite(str(out_dir / "preview.png"), preview)

    summary = {
        "ok": result.ok,
        "source": result.source,
        "fail_reason": result.fail_reason,
        "confidences": result.confidences,
        "inference_ms": result.inference_ms,
        "image_size": result.image_size,
        "debug": result.debug,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hard_cfg = config.get("dataset", {})
    hard_logger = HardCaseLogger(
        to_label_dir=hard_cfg.get("to_label_dir", "dataset/to_label"),
        cooldown_seconds=int(hard_cfg.get("cooldown_seconds", 5)),
    )

    reason = result.fail_reason
    min_workpiece_conf = config.get("yolo", {}).get("min_conf_by_class", {}).get("workpiece", 0.0)
    if result.workpiece_mask is not None and result.confidences.get("workpiece", 0.0) < min_workpiece_conf:
        reason = "low_confidence_workpiece"

    saved = hard_logger.save(frame, reason, summary) if reason else None
    print(json.dumps({"run_dir": str(out_dir), "hard_case_saved": str(saved) if saved else None, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
