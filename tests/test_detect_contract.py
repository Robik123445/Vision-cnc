import pytest

pytest.importorskip("cv2", reason="OpenCV runtime not available in this environment", exc_type=ImportError)

import numpy as np

from vision.detect.detector_api import create_detector
from vision.detect.result import DetectionResult


def test_detector_contract_always_returns_result():
    config = {
        "detector": {"fallback_enabled": True},
        "yolo": {"model_path": "missing.pt", "imgsz": 640, "min_conf_by_class": {"workpiece": 0.55}},
    }
    detector = create_detector(config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert isinstance(result, DetectionResult)
    assert set(result.masks.keys()) == {"workpiece", "clamp", "hand", "tool", "safety_mask"}
    assert set(result.confidences.keys()) == {"workpiece", "clamp", "hand", "tool"}
    assert isinstance(result.ok, bool)
    assert isinstance(result.fail_reason, str)


def test_mask_shape_matches_input_when_present():
    config = {
        "detector": {"fallback_enabled": True},
        "yolo": {"model_path": "missing.pt", "imgsz": 640, "min_conf_by_class": {"workpiece": 0.55}},
    }
    detector = create_detector(config)

    frame = np.full((80, 120, 3), 255, dtype=np.uint8)
    result = detector.detect(frame)
    mask = result.masks.get("workpiece")
    if mask is not None:
        assert mask.shape == frame.shape[:2]
