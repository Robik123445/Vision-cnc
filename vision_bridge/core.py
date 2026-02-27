"""Core adapter that bridges existing vision engine to API snapshot contract."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vision.detect.detector_api import create_detector, load_config
from vision_bridge.schemas import (
    DebugData,
    MaterialBBox,
    MaterialData,
    MetricsData,
    SnapshotResponse,
    TransformData,
)
from vision_bridge.version import SCHEMA_VERSION

LOGGER = logging.getLogger("vision.bridge.core")


class VisionCoreAdapter:
    """Adapter object that runs frame capture, detection and contract mapping."""

    def __init__(self, config_path: str = "config/vision_config.json") -> None:
        """Initialize adapter with detector and calibration configuration."""

        self.config_path = config_path
        self.config = load_config(config_path)
        self._detector = None

    def capture_frame(self) -> tuple[Optional[np.ndarray], str]:
        """Capture one camera frame; return reason when unavailable."""

        try:
            import cv2

            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None, "camera_read_failed"
            return frame, ""
        except Exception as exc:
            return None, f"camera_error:{exc}"

    def run_detection(self, frame: np.ndarray, min_confidence: float) -> Tuple[Optional[dict], str]:
        """Run existing detector and convert raw result to generic dict representation."""

        try:
            if self._detector is None:
                self._detector = create_detector(self.config)
            result = self._detector.detect(frame)
        except Exception as exc:
            return None, f"detector_error:{exc}"

        workpiece = result.masks.get("workpiece")
        confidence = float(result.confidences.get("workpiece", 0.0))
        if workpiece is None or confidence < min_confidence:
            return {
                "ok": False,
                "confidence": confidence,
                "mask": None,
                "fail_reason": result.fail_reason or "no_detection",
                "timing_ms": result.timing_ms,
                "debug": result.debug,
            }, ""

        return {
            "ok": True,
            "confidence": confidence,
            "mask": workpiece,
            "fail_reason": "",
            "timing_ms": result.timing_ms,
            "debug": result.debug,
        }, ""

    def compute_homography_and_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Load homography and derive scale vector; fallback to identity when unavailable."""

        try:
            from vision.calib.plane import load_plane

            plane_path = self.config.get("calibration", {}).get("plane_path", "calibration/plane.json")
            if not Path(plane_path).exists():
                raise FileNotFoundError(plane_path)
            plane = load_plane(plane_path)
            H = plane.H_px_to_mm.astype(float)
            sx = float(np.hypot(H[0, 0], H[0, 1]))
            sy = float(np.hypot(H[1, 0], H[1, 1]))
            return H, np.array([sx, sy], dtype=float)
        except Exception:
            return np.eye(3, dtype=float), np.array([1.0, 1.0], dtype=float)

    def build_material_geometry(self, detection: Dict[str, Any], H: np.ndarray) -> MaterialData:
        """Build polygon, bbox, orientation and origin from detected mask."""

        mask = detection.get("mask")
        if mask is None:
            poly = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            bbox = MaterialBBox(min=[0.0, 0.0], max=[0.0, 0.0])
            return MaterialData(material_polygon_mm=poly, material_bbox_mm=bbox, material_angle_deg=0.0, origin_mm=[0.0, 0.0])

        ys, xs = np.where(mask > 0)
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        corners_px = np.array(
            [[[min_x, min_y]], [[max_x, min_y]], [[max_x, max_y]], [[min_x, max_y]]],
            dtype=np.float32,
        )

        try:
            import cv2

            corners_mm = cv2.perspectiveTransform(corners_px, H).reshape(-1, 2)
        except Exception:
            corners_mm = corners_px.reshape(-1, 2)

        poly = [[float(p[0]), float(p[1])] for p in corners_mm[:3]]
        bbox = MaterialBBox(
            min=[float(np.min(corners_mm[:, 0])), float(np.min(corners_mm[:, 1]))],
            max=[float(np.max(corners_mm[:, 0])), float(np.max(corners_mm[:, 1]))],
        )
        dx = corners_mm[1][0] - corners_mm[0][0]
        dy = corners_mm[1][1] - corners_mm[0][1]
        angle = float(np.degrees(np.arctan2(dy, dx)))
        origin = [float(np.mean(corners_mm[:, 0])), float(np.mean(corners_mm[:, 1]))]

        return MaterialData(material_polygon_mm=poly, material_bbox_mm=bbox, material_angle_deg=angle, origin_mm=origin)

    def build_snapshot_from_frame(self, frame: np.ndarray, min_confidence: float = 0.25) -> SnapshotResponse:
        """Run full pipeline from frame to stable versioned snapshot response."""

        detection, err = self.run_detection(frame, min_confidence)
        if err:
            return self._error_snapshot(err)

        H, scale = self.compute_homography_and_mm()
        material = self.build_material_geometry(detection or {}, H)
        status = "OK" if (detection and detection.get("ok")) else "NO_DETECTION"

        return SnapshotResponse(
            schema_version=SCHEMA_VERSION,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_id=str(uuid.uuid4()),
            material=material,
            transforms=TransformData(
                H_cam_to_machine=H.astype(float).tolist(),
                px_to_mm_scale=[float(scale[0]), float(scale[1])],
            ),
            metrics=MetricsData(
                confidence=float((detection or {}).get("confidence", 0.0)),
                error_mm_est=float((detection or {}).get("timing_ms", {}).get("inference", 0.0) / 1000.0),
            ),
            debug=DebugData(
                reason=str((detection or {}).get("fail_reason", "")),
                debug_image_path=None,
                raw=(detection or {}).get("debug", {}),
            ),
        )

    def _error_snapshot(self, reason: str) -> SnapshotResponse:
        """Create canonical ERROR snapshot response."""

        return SnapshotResponse(
            schema_version=SCHEMA_VERSION,
            status="ERROR",
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_id=str(uuid.uuid4()),
            material=MaterialData(
                material_polygon_mm=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                material_bbox_mm=MaterialBBox(min=[0.0, 0.0], max=[0.0, 0.0]),
                material_angle_deg=0.0,
                origin_mm=[0.0, 0.0],
            ),
            transforms=TransformData(H_cam_to_machine=np.eye(3).tolist(), px_to_mm_scale=[1.0, 1.0]),
            metrics=MetricsData(confidence=0.0, error_mm_est=0.0),
            debug=DebugData(reason=reason, debug_image_path=None, raw={}),
        )
