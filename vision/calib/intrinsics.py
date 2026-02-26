"""Intrinsic calibration and undistortion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from vision.calib.types import Intrinsics


def calibrate_intrinsics_from_chessboard(
    images: Sequence[np.ndarray], pattern: Tuple[int, int] = (9, 6), square_mm: float = 25.0
) -> Intrinsics:
    """Estimate camera intrinsics from chessboard images."""

    if not images:
        raise ValueError("No images provided for intrinsic calibration")

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    objp *= float(square_mm)

    obj_points = []
    img_points = []
    image_size = None

    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 3:
        raise ValueError("Not enough valid chessboard detections, need at least 3 images")

    ok, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    if not ok:
        raise RuntimeError("cv2.calibrateCamera failed")
    return Intrinsics(K=K, dist=dist, image_size=image_size)


def undistort_frame(frame: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Undistort frame with calibrated camera intrinsics."""

    return cv2.undistort(frame, intrinsics.K, intrinsics.dist)


def save_intrinsics(intrinsics: Intrinsics, path: str = "calibration/intrinsics.json") -> Path:
    """Store intrinsics JSON file on disk."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(intrinsics.to_dict(), indent=2), encoding="utf-8")
    return out


def load_intrinsics(path: str = "calibration/intrinsics.json") -> Intrinsics:
    """Load intrinsics JSON artifact from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Intrinsics.from_dict(payload)
