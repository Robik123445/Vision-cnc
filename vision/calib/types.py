"""Type definitions for camera and plane calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Intrinsics:
    """Camera intrinsics and distortion parameters used for undistortion."""

    K: np.ndarray
    dist: np.ndarray
    image_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize intrinsics to JSON-safe dictionary."""

        return {
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Intrinsics":
        """Deserialize intrinsics from dictionary payload."""

        return cls(
            K=np.asarray(payload["K"], dtype=np.float64),
            dist=np.asarray(payload["dist"], dtype=np.float64).reshape(1, -1),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        )


@dataclass
class PlaneCalibration:
    """Homography mapping pixels to machine-plane millimeters."""

    H_px_to_mm: np.ndarray
    workspace_mm: Tuple[float, float]
    origin_mm: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plane calibration to JSON-safe dictionary."""

        return {
            "H_px_to_mm": self.H_px_to_mm.tolist(),
            "workspace_mm": {
                "width": float(self.workspace_mm[0]),
                "height": float(self.workspace_mm[1]),
            },
            "origin_mm": self.origin_mm,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlaneCalibration":
        """Deserialize plane calibration from dictionary payload."""

        ws = payload.get("workspace_mm", {})
        return cls(
            H_px_to_mm=np.asarray(payload["H_px_to_mm"], dtype=np.float64),
            workspace_mm=(float(ws.get("width", 0.0)), float(ws.get("height", 0.0))),
            origin_mm=str(payload.get("origin_mm", "bottom_left")),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class FullCalibration:
    """Combined intrinsic + plane calibration object for runtime use."""

    intrinsics: Intrinsics | None
    plane: PlaneCalibration

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete calibration set."""

        return {
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "plane": self.plane.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FullCalibration":
        """Deserialize complete calibration set."""

        intr = payload.get("intrinsics")
        return cls(
            intrinsics=Intrinsics.from_dict(intr) if intr else None,
            plane=PlaneCalibration.from_dict(payload["plane"]),
        )
