"""Active-learning hard case exporter for detector failures."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


HARD_REASONS = {
    "no_detections",
    "model_not_loaded",
    "fallback_no_object",
    "hand_detected",
    "low_confidence_workpiece",
}


class HardCaseLogger:
    """Persist difficult frames for later annotation with cooldown protection."""

    def __init__(self, to_label_dir: str = "dataset/to_label", cooldown_seconds: int = 5) -> None:
        self.to_label_dir = Path(to_label_dir)
        self.cooldown_seconds = cooldown_seconds
        self._last_save_by_reason: Dict[str, float] = {}

    def should_save(self, reason: Optional[str]) -> bool:
        """Return True when reason is tracked and not suppressed by cooldown."""

        if reason not in HARD_REASONS:
            return False
        now = time.time()
        last = self._last_save_by_reason.get(reason, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_save_by_reason[reason] = now
        return True

    def _build_overlay(self, frame: np.ndarray, masks: Dict[str, np.ndarray | None]) -> np.ndarray:
        """Render frame + masks overlay image for quick offline review."""

        import cv2

        overlay = frame.copy()
        color_map = {
            "workpiece": (0, 255, 0),
            "clamp": (0, 0, 255),
            "hand": (0, 255, 255),
            "tool": (255, 0, 0),
            "safety_mask": (255, 0, 255),
        }
        for key, color in color_map.items():
            mask = masks.get(key)
            if mask is None:
                continue
            overlay[mask.astype(bool)] = color
        return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

    def save(self, frame: np.ndarray, reason: str, meta: Dict, masks: Dict[str, np.ndarray | None] | None = None) -> Optional[Path]:
        """Save frame, overlay and metadata into date-scoped hard-case folder."""

        if not self.should_save(reason):
            return None

        day = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        case_dir = self.to_label_dir / day / f"{ts}_{reason}"
        case_dir.mkdir(parents=True, exist_ok=True)

        image_path = case_dir / "frame.png"
        overlay_path = case_dir / "overlay.png"
        meta_path = case_dir / "meta.json"

        import cv2

        cv2.imwrite(str(image_path), frame)
        if masks:
            cv2.imwrite(str(overlay_path), self._build_overlay(frame, masks))

        payload = {"reason": reason, "timestamp": ts, **meta}
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return case_dir
