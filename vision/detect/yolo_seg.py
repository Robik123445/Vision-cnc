"""YOLOv8-seg wrapper returning DetectionResult contract for CNC vision."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from vision.detect.detector_api import DetectionResult, Detector
from vision.detect.postprocess import process_yolo_masks


class YoloSegmenter(Detector):
    """Safe wrapper around ultralytics YOLO segmentation model."""

    def __init__(self, config: Dict) -> None:
        self._config = config
        self._model = None
        self._class_map = self._load_class_map()
        self._min_conf = config.get("yolo", {}).get("min_conf_by_class", {})
        self._model_error: Optional[str] = None
        self._load_model()

    def _load_class_map(self) -> Dict[int, str]:
        """Load class mapping from classes.yaml without hard-coding names."""

        class_path = Path("vision/detect/classes.yaml")
        if not class_path.exists():
            return {0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"}

        raw_lines = class_path.read_text(encoding="utf-8").splitlines()
        result: Dict[int, str] = {}
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("classes:"):
                continue
            if ":" not in line:
                continue
            key, value = [x.strip() for x in line.split(":", 1)]
            if key.isdigit():
                result[int(key)] = value
        return result or {0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"}

    def _load_model(self) -> None:
        """Load YOLO model and keep error state instead of raising exceptions."""

        model_path = self._config.get("yolo", {}).get("model_path", "")
        try:
            from ultralytics import YOLO  # type: ignore

            if not model_path or not Path(model_path).exists():
                self._model_error = "model_not_loaded"
                return
            self._model = YOLO(model_path)
        except Exception:
            self._model_error = "model_not_loaded"
            self._model = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO inference and always return a valid contract-compatible result."""

        start = time.perf_counter()
        h, w = frame.shape[:2]

        if self._model is None:
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
                source="none",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason=self._model_error or "model_not_loaded",
            )

        try:
            imgsz = int(self._config.get("yolo", {}).get("imgsz", 640))
            results = self._model.predict(source=frame, imgsz=imgsz, device="cpu", verbose=False)
            first = results[0] if results else None

            if first is None or first.boxes is None or first.masks is None or len(first.boxes) == 0:
                return DetectionResult(
                    workpiece_mask=None,
                    clamp_mask=None,
                    hand_mask=None,
                    tool_mask=None,
                    confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
                    source="yolo",
                    inference_ms=(time.perf_counter() - start) * 1000.0,
                    image_size=(h, w),
                    fail_reason="no_detections",
                )

            masks = [m for m in first.masks.data.cpu().numpy()]
            class_ids = [int(c) for c in first.boxes.cls.cpu().numpy().tolist()]
            confs = [float(c) for c in first.boxes.conf.cpu().numpy().tolist()]
            out_masks, out_conf = process_yolo_masks(
                masks=masks,
                class_ids=class_ids,
                confidences=confs,
                class_map=self._class_map,
                image_size=(h, w),
                min_conf_by_class=self._min_conf,
            )

            fail_reason = None if any(v is not None for v in out_masks.values()) else "no_detections"
            return DetectionResult(
                workpiece_mask=out_masks["workpiece"],
                clamp_mask=out_masks["clamp"],
                hand_mask=out_masks["hand"],
                tool_mask=out_masks["tool"],
                confidences=out_conf,
                source="yolo",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason=fail_reason,
            )
        except Exception:
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
                source="none",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason="model_not_loaded",
            )
