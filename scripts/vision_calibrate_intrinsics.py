#!/usr/bin/env python3
"""CLI script for intrinsic camera calibration using chessboard images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vision.calib.intrinsics import calibrate_intrinsics_from_chessboard, save_intrinsics


def parse_pattern(value: str) -> tuple[int, int]:
    """Parse chessboard pattern syntax WIDTHxHEIGHT."""

    w, h = value.lower().split("x")
    return int(w), int(h)


def main() -> int:
    """Load input images, run calibration and save intrinsics JSON."""

    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard photos")
    parser.add_argument("--images_dir", required=True, help="Directory with calibration images")
    parser.add_argument("--pattern", default="9x6", help="Chessboard pattern, e.g., 9x6")
    parser.add_argument("--square_mm", type=float, default=25.0, help="Square size in millimeters")
    parser.add_argument("--out", default="calibration/intrinsics.json", help="Output JSON path")
    parser.add_argument("--camera_id", default="", help="Optional camera identifier")
    args = parser.parse_args()

    files = sorted([p for p in Path(args.images_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not files:
        raise SystemExit("No calibration images found")

    images = []
    for path in files:
        frame = cv2.imread(str(path))
        if frame is not None:
            images.append(frame)

    intrinsics = calibrate_intrinsics_from_chessboard(images, pattern=parse_pattern(args.pattern), square_mm=args.square_mm, camera_id=args.camera_id)
    out_path = save_intrinsics(intrinsics, args.out)
    print(f"Saved intrinsics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
