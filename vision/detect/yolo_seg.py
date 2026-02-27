"""YOLOv8-seg wrapper returning DetectionResult contract for CNC vision."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from vision.detect.detector_api import Detector
from vision.detect.postprocess import process_yolo_masks
from vision.detect.result import DetectionResult


class YoloSegmenter(Detector):
    """Safe wrapper around ultralytics YOLO segmentation model."""

    def __init__(self, config: Dict) -> None:
        self._config = config
        self._model = None
        self._class_map = self._load_class_map()
        self._min_conf = config.get("yolo", {}).get("min_conf_by_class", {})
        self._min_area = config.get("yolo", {}).get("min_area_by_class", {})
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

    def _empty(self, reason: str, h: int, w: int, elapsed_ms: float, source: str = "none") -> DetectionResult:
        """Build empty contract response for graceful failures."""

        return DetectionResult(
            ok=False,
            fail_reason=reason,
            masks={"workpiece": None, "clamp": None, "hand": None, "tool": None, "safety_mask": None},
            confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
            timing_ms={"inference": elapsed_ms},
            debug={"source": source, "image_size": [h, w]},
        )

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
            return self._empty(self._model_error or "model_not_loaded", h, w, (time.perf_counter() - start) * 1000.0)

        try:
            imgsz = int(self._config.get("yolo", {}).get("imgsz", 640))
            results = self._model.predict(source=frame, imgsz=imgsz, device="cpu", verbose=False)
            first = results[0] if results else None
            elapsed = (time.perf_counter() - start) * 1000.0

            if first is None or first.boxes is None or first.masks is None or len(first.boxes) == 0:
                return self._empty("no_detections", h, w, elapsed, source="yolo")

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
                min_area_by_class=self._min_area,
            )

            ok = out_masks["workpiece"] is not None
            return DetectionResult(
                ok=ok,
                fail_reason="" if ok else "no_detections",
                masks=out_masks,
                confidences=out_conf,
                timing_ms={"inference": elapsed},
                debug={
                    "source": "yolo",
                    "model_name": str(self._config.get("yolo", {}).get("model_path", "")),
                    "imgsz": imgsz,
                    "image_size": [h, w],
                    "raw_class_ids": class_ids,
                },
            )
        except Exception:
            return self._empty("model_not_loaded", h, w, (time.perf_counter() - start) * 1000.0)
