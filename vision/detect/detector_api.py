"""Unified detector API for CNC vision segmentation."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from vision.dataset.hard_cases import HardCaseLogger


LOGGER = logging.getLogger("vision.detect")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )


@dataclass
class DetectionResult:
    """Contract output for all segmentation detectors."""

    workpiece_mask: Optional[np.ndarray]
    clamp_mask: Optional[np.ndarray]
    hand_mask: Optional[np.ndarray]
    tool_mask: Optional[np.ndarray]
    confidences: Dict[str, float]
    source: str
    inference_ms: float
    image_size: Tuple[int, int]
    fail_reason: Optional[str]


class Detector(ABC):
    """Abstract detector interface shared by YOLO and fallback implementations."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run segmentation and always return a valid DetectionResult."""


def load_config(config_path: str = "config/vision_config.json") -> dict:
    """Load detector configuration JSON from disk."""

    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class HybridDetector(Detector):
    """Try YOLO first and fallback based on failures or confidence gating."""

    def __init__(self, yolo: Detector, fallback: Optional[Detector], config: dict) -> None:
        self._yolo = yolo
        self._fallback = fallback
        self._config = config
        ds_cfg = config.get("dataset", {})
        self._hard_cases = HardCaseLogger(
            to_label_dir=ds_cfg.get("to_label_dir", "dataset/to_label"),
            cooldown_seconds=int(ds_cfg.get("cooldown_seconds", 5)),
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO and optionally switch to fallback under guarded conditions."""

        yolo_result = self._yolo.detect(frame)
        min_conf = self._config.get("yolo", {}).get("min_conf_by_class", {}).get("workpiece", 0.0)
        workpiece_conf = yolo_result.confidences.get("workpiece", 0.0)

        hand_conf = yolo_result.confidences.get("hand", 0.0)
        if yolo_result.hand_mask is not None and hand_conf > 0:
            yolo_result.fail_reason = "hand_detected"
            self._hard_cases.save(frame, "hand_detected", {"source": yolo_result.source, "confidences": yolo_result.confidences})
            return yolo_result

        should_fallback = yolo_result.fail_reason in {"model_not_loaded", "no_detections"}
        should_fallback = should_fallback or (yolo_result.workpiece_mask is None)
        should_fallback = should_fallback or (workpiece_conf < min_conf)

        if should_fallback and self._fallback is not None:
            fallback_reason = yolo_result.fail_reason or "low_confidence_workpiece"
            LOGGER.info("Switching to fallback segmentation. reason=%s", fallback_reason)
            fallback_result = self._fallback.detect(frame)
            if yolo_result.fail_reason and fallback_result.fail_reason is None:
                fallback_result.fail_reason = yolo_result.fail_reason
            if fallback_reason == "low_confidence_workpiece":
                self._hard_cases.save(frame, "low_confidence_workpiece", {"source": yolo_result.source, "confidences": yolo_result.confidences})
            if fallback_result.fail_reason:
                self._hard_cases.save(frame, fallback_result.fail_reason, {"source": fallback_result.source, "confidences": fallback_result.confidences})
            return fallback_result

        if yolo_result.fail_reason:
            self._hard_cases.save(frame, yolo_result.fail_reason, {"source": yolo_result.source, "confidences": yolo_result.confidences})
        elif workpiece_conf < min_conf:
            self._hard_cases.save(frame, "low_confidence_workpiece", {"source": yolo_result.source, "confidences": yolo_result.confidences})

        return yolo_result


def create_detector(config: Optional[dict] = None) -> Detector:
    """Factory for detector creation with YOLO-first and fallback strategy."""

    config = config or load_config()
    fallback_enabled = config.get("detector", {}).get("fallback_enabled", True)

    from vision.detect.fallback_seg import FallbackSegmenter
    from vision.detect.yolo_seg import YoloSegmenter

    yolo = YoloSegmenter(config)
    fallback = FallbackSegmenter(config) if fallback_enabled else None
    return HybridDetector(yolo=yolo, fallback=fallback, config=config)
