"""Unified detector API for CNC vision segmentation."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from vision.dataset.hard_cases import HardCaseLogger
from vision.detect.result import DetectionResult

if TYPE_CHECKING:
    import numpy as np


LOGGER = logging.getLogger("vision.detect")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler("log.txt")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


class Detector(ABC):
    """Abstract detector interface shared by YOLO and fallback implementations."""

    @abstractmethod
    def detect(self, frame: Any) -> DetectionResult:
        """Run segmentation and always return a valid DetectionResult."""


def load_config(config_path: str = "config/vision_config.json") -> dict:
    """Load detector configuration JSON from disk."""

    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: dict, config_path: str = "config/vision_config.json") -> str:
    """Persist detector configuration JSON to disk."""

    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config_path


class HybridDetector(Detector):
    """Try YOLO first and optionally switch to the fallback segmenter."""

    def __init__(self, yolo: Detector, fallback: Optional[Detector], config: dict) -> None:
        self._yolo = yolo
        self._fallback = fallback
        self._config = config
        ds_cfg = config.get("dataset", {})
        self._hard_cases = HardCaseLogger(
            to_label_dir=ds_cfg.get("to_label_dir", "dataset/to_label"),
            cooldown_seconds=int(ds_cfg.get("cooldown_seconds", 5)),
        )

    def _save_hard_case(self, frame: Any, reason: Optional[str], result: DetectionResult) -> None:
        """Persist tracked hard-case metadata when the reason is actionable."""

        if not reason:
            return

        payload = {
            "source": result.source,
            "fail_reason": result.fail_reason,
            "confidences": result.confidences,
            "image_size": result.image_size,
            "inference_ms": result.inference_ms,
            **result.debug,
        }
        self._hard_cases.save(frame, reason, payload)

    def detect(self, frame: Any) -> DetectionResult:
        """Run YOLO first and route to fallback when policy requires it."""

        yolo_result = self._yolo.detect(frame)
        min_conf = self._config.get("yolo", {}).get("min_conf_by_class", {}).get("workpiece", 0.0)
        workpiece_conf = yolo_result.confidences.get("workpiece", 0.0)
        hand_conf = yolo_result.confidences.get("hand", 0.0)

        if yolo_result.hand_mask is not None and hand_conf > 0.0:
            yolo_result.fail_reason = "hand_detected"
            self._save_hard_case(frame, "hand_detected", yolo_result)
            return yolo_result

        should_fallback = yolo_result.fail_reason in {"model_not_loaded", "no_detections"}
        should_fallback = should_fallback or (yolo_result.workpiece_mask is None)
        should_fallback = should_fallback or (workpiece_conf < min_conf)

        if should_fallback and self._fallback is not None:
            fallback_reason = yolo_result.fail_reason
            if fallback_reason is None and yolo_result.workpiece_mask is None:
                fallback_reason = "no_detections"
            if fallback_reason is None and workpiece_conf < min_conf:
                fallback_reason = "low_confidence_workpiece"

            LOGGER.info("Switching to fallback segmentation. reason=%s", fallback_reason)
            fallback_result = self._fallback.detect(frame)

            if fallback_reason == "low_confidence_workpiece":
                self._save_hard_case(frame, fallback_reason, yolo_result)
            if fallback_result.fail_reason:
                self._save_hard_case(frame, fallback_result.fail_reason, fallback_result)
            return fallback_result

        if yolo_result.fail_reason:
            self._save_hard_case(frame, yolo_result.fail_reason, yolo_result)
        elif workpiece_conf < min_conf:
            self._save_hard_case(frame, "low_confidence_workpiece", yolo_result)

        return yolo_result


def create_detector(config: Optional[dict] = None) -> Detector:
    """Create detector from config with CI-safe imports and runtime lazy loading."""

    config = config or load_config()
    detector_type = config.get("detector", {}).get("type", "yolo")
    fallback_enabled = config.get("detector", {}).get("fallback_enabled", True)

    from vision.detect.fallback_seg import FallbackSegmenter
    from vision.detect.yolo_seg import YoloSegmenter

    if detector_type == "fallback":
        return FallbackSegmenter(config)

    yolo = YoloSegmenter(config)
    fallback = FallbackSegmenter(config) if fallback_enabled else None
    return HybridDetector(yolo=yolo, fallback=fallback, config=config)
