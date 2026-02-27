"""Plane calibration and pixel/mm conversion routines."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from vision.calib.types import CalibrationMeta, Intrinsics, PlaneCalibration


def _points_checksum(points_px: np.ndarray) -> str:
    """Generate deterministic checksum from clicked pixel points."""

    return hashlib.sha256(points_px.astype(np.float64).tobytes()).hexdigest()


def compute_homography_px_to_mm(
    points_px: Iterable[Tuple[float, float]],
    points_mm: Iterable[Tuple[float, float]],
    intrinsics: Intrinsics | None = None,
    workspace_mm: Tuple[float, float] = (0.0, 0.0),
    origin_mm: str = "bottom_left",
    notes: str = "",
    camera_id: str = "",
) -> PlaneCalibration:
    """Compute homography matrix that maps camera pixels to CNC millimeters."""

    src = np.asarray(list(points_px), dtype=np.float64)
    dst = np.asarray(list(points_mm), dtype=np.float64)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required")
    if src.shape != dst.shape:
        raise ValueError("points_px and points_mm must have the same shape")

    if intrinsics is not None:
        src = cv2.undistortPoints(src.reshape(-1, 1, 2), intrinsics.K, intrinsics.dist, P=intrinsics.K).reshape(-1, 2)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("Failed to compute homography")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    meta = CalibrationMeta(
        schema_version="1.1",
        created_at=datetime.now(timezone.utc).isoformat(),
        camera_id=camera_id,
        frame_checksum=_points_checksum(src),
    )

    return PlaneCalibration(
        H_px_to_mm=H,
        workspace_mm=workspace_mm,
        origin_mm=origin_mm,
        notes=notes,
        rmse_mm=rmse,
        meta=meta,
    )


def pixel_to_mm(x_px: float, y_px: float, calib: PlaneCalibration) -> Tuple[float, float]:
    """Transform one pixel position into CNC mm coordinates."""

    pt = np.array([[[x_px, y_px]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(pt, calib.H_px_to_mm)[0, 0]
    return float(mapped[0]), float(mapped[1])


def mm_to_pixel(x_mm: float, y_mm: float, calib: PlaneCalibration) -> Tuple[float, float]:
    """Transform one CNC mm coordinate back to camera pixels (debug utility)."""

    inv = np.linalg.inv(calib.H_px_to_mm)
    pt = np.array([[[x_mm, y_mm]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(pt, inv)[0, 0]
    return float(mapped[0]), float(mapped[1])


def mask_px_to_mask_mm_grid(
    mask_px: np.ndarray,
    calib: PlaneCalibration,
    target_shape: Tuple[int, int],
    workspace_mm: Tuple[float, float],
) -> np.ndarray:
    """Warp a pixel mask into mm-aligned occupancy grid for CNC workspace."""

    grid_h, grid_w = target_shape
    w_mm, h_mm = workspace_mm
    sx = grid_w / float(w_mm) if w_mm else 1.0
    sy = grid_h / float(h_mm) if h_mm else 1.0

    H_scaled = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64) @ calib.H_px_to_mm
    warped = cv2.warpPerspective((mask_px > 0).astype(np.uint8), H_scaled, (grid_w, grid_h), flags=cv2.INTER_NEAREST)
    return (warped > 0).astype(np.uint8)


def save_plane(calib: PlaneCalibration, path: str = "calibration/plane.json") -> Path:
    """Store plane calibration JSON artifact on disk."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    return out


def load_plane(path: str = "calibration/plane.json") -> PlaneCalibration:
    """Load plane calibration from JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlaneCalibration.from_dict(payload)
