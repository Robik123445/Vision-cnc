#!/usr/bin/env python3
"""Manual 4-point plane calibration by clicking image points."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vision.calib.intrinsics import load_intrinsics, undistort_frame
from vision.calib.plane import compute_homography_px_to_mm, save_plane


CLICKED_POINTS: list[tuple[float, float]] = []


def on_click(event, x, y, _flags, _param):
    """Capture up to 4 click points in image coordinates."""

    if event == cv2.EVENT_LBUTTONDOWN and len(CLICKED_POINTS) < 4:
        CLICKED_POINTS.append((float(x), float(y)))


def parse_workspace(text: str) -> tuple[float, float]:
    """Parse workspace string WxH into floats."""

    w, h = text.lower().split("x")
    return float(w), float(h)


def main() -> int:
    """Open image, collect points and save plane calibration."""

    parser = argparse.ArgumentParser(description="Manual plane calibration with 4 clicked points")
    parser.add_argument("--image", required=True, help="Calibration frame path")
    parser.add_argument("--workspace_mm", default="800x500", help="Workspace size in mm, format WxH")
    parser.add_argument("--intrinsics", default="", help="Optional intrinsics path for undistortion")
    parser.add_argument("--out", default="calibration/plane.json", help="Output plane JSON path")
    parser.add_argument("--notes", default="manual_4_point", help="Optional notes")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Failed to load image: {args.image}")

    intrinsics = None
    if args.intrinsics and Path(args.intrinsics).exists():
        intrinsics = load_intrinsics(args.intrinsics)
        frame = undistort_frame(frame, intrinsics)

    preview = frame.copy()
    cv2.namedWindow("plane_calibration")
    cv2.setMouseCallback("plane_calibration", on_click)

    while True:
        canvas = preview.copy()
        for idx, (x, y) in enumerate(CLICKED_POINTS):
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(canvas, str(idx + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("plane_calibration", canvas)
        key = cv2.waitKey(30)
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")
        if key in (13, 10) and len(CLICKED_POINTS) == 4:
            break

    cv2.destroyAllWindows()
    ws = parse_workspace(args.workspace_mm)
    mm_points = [(0.0, 0.0), (ws[0], 0.0), (ws[0], ws[1]), (0.0, ws[1])]

    plane = compute_homography_px_to_mm(CLICKED_POINTS, mm_points, intrinsics=None, workspace_mm=ws, origin_mm="bottom_left", notes=args.notes)
    out = save_plane(plane, args.out)
    print(f"Saved plane calibration: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
