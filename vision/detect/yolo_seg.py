"""YOLOv8-seg wrapper returning DetectionResult contract for CNC vision."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from vision.detect.detector_api import DetectionResult, Detector
from vision.detect.postprocess import process_yolo_masks


LOGGER = logging.getLogger("vision.detect.yolo")


def _frame_size(frame: Any) -> tuple[int, int]:
    """Read frame shape without requiring numpy at import time."""

    if hasattr(frame, "shape") and len(frame.shape) >= 2:
        return int(frame.shape[0]), int(frame.shape[1])
    return 0, 0


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
        """Load class mapping from classes.yaml without requiring PyYAML."""

        class_path = Path("vision/detect/classes.yaml")
        default_map = {0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"}
        if not class_path.exists():
            return default_map

        parsed: Dict[int, str] = {}
        try:
            for line in class_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or text.startswith("classes"):
                    continue
                if ":" not in text:
                    continue
                key, value = [item.strip() for item in text.split(":", 1)]
                if key.isdigit():
                    parsed[int(key)] = value
        except Exception as exc:
            LOGGER.warning("Failed to parse classes.yaml, using defaults: %s", exc)
            return default_map

        return parsed or default_map

    def _load_model(self) -> None:
        """Load YOLO model and keep error state instead of raising exceptions."""

        model_path = self._config.get("yolo", {}).get("model_path", "")
        try:
            from ultralytics import YOLO  # type: ignore

            if not model_path or not Path(model_path).exists():
                self._model_error = "model_not_loaded"
                return

            self._model = YOLO(model_path)
        except Exception as exc:
            LOGGER.warning("Failed to initialize YOLO model: %s", exc)
            self._model_error = "model_not_loaded"
            self._model = None

    def detect(self, frame: Any) -> DetectionResult:
        """Run YOLO inference and always return a valid contract-compatible result."""

        start = time.perf_counter()
        h, w = _frame_size(frame)
        empty_conf = {"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0}

        if self._model is None:
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences=empty_conf,
                source="none",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason=self._model_error or "model_not_loaded",
                debug={"model_path": self._config.get("yolo", {}).get("model_path", "")},
            )

        try:
            imgsz = int(self._config.get("yolo", {}).get("imgsz", 640))
            device = self._config.get("yolo", {}).get("device", "cpu")
            results = self._model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
            first = results[0] if results else None

            if first is None or first.boxes is None or first.masks is None or len(first.boxes) == 0:
                return DetectionResult(
                    workpiece_mask=None,
                    clamp_mask=None,
                    hand_mask=None,
                    tool_mask=None,
                    confidences=empty_conf,
                    source="yolo",
                    inference_ms=(time.perf_counter() - start) * 1000.0,
                    image_size=(h, w),
                    fail_reason="no_detections",
                    debug={"imgsz": imgsz, "device": device},
                )

            masks = [mask for mask in first.masks.data.cpu().numpy()]
            class_ids = [int(value) for value in first.boxes.cls.cpu().numpy().tolist()]
            confs = [float(value) for value in first.boxes.conf.cpu().numpy().tolist()]
            out_masks, out_conf = process_yolo_masks(
                masks=masks,
                class_ids=class_ids,
                confidences=confs,
                class_map=self._class_map,
                image_size=(h, w),
                min_conf_by_class=self._min_conf,
                min_area_by_class=self._min_area,
            )

            return DetectionResult(
                workpiece_mask=out_masks["workpiece"],
                clamp_mask=out_masks["clamp"],
                hand_mask=out_masks["hand"],
                tool_mask=out_masks["tool"],
                confidences=out_conf,
                source="yolo",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason=None if out_masks["workpiece"] is not None else "no_detections",
                debug={
                    "model_path": self._config.get("yolo", {}).get("model_path", ""),
                    "imgsz": imgsz,
                    "device": device,
                    "raw_class_ids": class_ids,
                },
            )
        except Exception as exc:
            LOGGER.warning("YOLO inference failed: %s", exc)
            return DetectionResult(
                workpiece_mask=None,
                clamp_mask=None,
                hand_mask=None,
                tool_mask=None,
                confidences=empty_conf,
                source="none",
                inference_ms=(time.perf_counter() - start) * 1000.0,
                image_size=(h, w),
                fail_reason="model_not_loaded",
                debug={"error": str(exc)},
            )
