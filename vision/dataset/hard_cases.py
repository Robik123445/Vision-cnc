"""Active-learning hard case exporter for detector failures."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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

    def save(self, frame: Any, reason: str, meta: Dict[str, Any]) -> Optional[Path]:
        """Save metadata always; save frame image when OpenCV is available."""

        if not self.should_save(reason):
            return None

        day = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        case_dir = self.to_label_dir / day / f"{ts}_{reason}"
        case_dir.mkdir(parents=True, exist_ok=True)

        image_path = case_dir / "frame.png"
        meta_path = case_dir / "meta.json"

        image_saved = False
        try:
            import cv2  # type: ignore

            image_saved = bool(cv2.imwrite(str(image_path), frame))
        except Exception:
            image_saved = False

        payload = {"reason": reason, "timestamp": ts, "image_saved": image_saved, **meta}
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return case_dir
