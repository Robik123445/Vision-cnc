"""Fallback segmenter for CPU-safe basic workpiece extraction."""

from __future__ import annotations

import time
from typing import Any, Dict

from vision.detect.detector_api import DetectionResult, Detector


def _frame_size(frame: Any) -> tuple[int, int]:
    """Read frame shape without assuming numpy is installed at import time."""

    if hasattr(frame, "shape") and len(frame.shape) >= 2:
        return int(frame.shape[0]), int(frame.shape[1])
    return 0, 0


class FallbackSegmenter(Detector):
    """Threshold-based segmentation used when YOLO is unavailable or unreliable."""

    def __init__(self, config: Dict) -> None:
        self._config = config

    def detect(self, frame: Any) -> DetectionResult:
        """Segment the largest bright object as a workpiece candidate."""

        start = time.perf_counter()
        h, w = _frame_size(frame)

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
                source="fallback",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason="fallback_no_object",
                debug={"reason": "missing_runtime_dependencies"},
            )

        if h == 0 or w == 0:
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
                source="fallback",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason="fallback_no_object",
                debug={"reason": "invalid_frame"},
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        workpiece_mask = None

        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            region = (labels == largest_label).astype(np.uint8)
            if int(np.count_nonzero(region)) > 0:
                workpiece_mask = region

        return DetectionResult(
            workpiece_mask=workpiece_mask,
            clamp_mask=None,
            hand_mask=None,
            tool_mask=None,
            confidences={"workpiece": 1.0 if workpiece_mask is not None else 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
            source="fallback",
            inference_ms=(time.perf_counter() - start) * 1000.0,
            image_size=(h, w),
            fail_reason=None if workpiece_mask is not None else "fallback_no_object",
            debug={"algorithm": "otsu_threshold"},
        )
