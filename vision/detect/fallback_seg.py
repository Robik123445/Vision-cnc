"""Fallback segmenter for CPU-safe basic workpiece extraction."""

from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np

from vision.detect.detector_api import DetectionResult, Detector


class FallbackSegmenter(Detector):
    """Threshold-based segmentation used when YOLO is unavailable or unreliable."""

    def __init__(self, config: Dict) -> None:
        self._config = config

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Segment largest object as workpiece using grayscale threshold + morphology."""

        start = time.perf_counter()
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        workpiece_mask = None
        fail_reason = None

        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            region = (labels == largest_label).astype(np.uint8)
            if np.count_nonzero(region) > 0:
                workpiece_mask = region

        if workpiece_mask is None:
            fail_reason = "fallback_no_object"

        return DetectionResult(
            workpiece_mask=workpiece_mask,
            clamp_mask=None,
            hand_mask=None,
            tool_mask=None,
            confidences={"workpiece": 1.0 if workpiece_mask is not None else 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
            source="fallback",
            inference_ms=(time.perf_counter() - start) * 1000.0,
            image_size=(h, w),
            fail_reason=fail_reason,
        )
