import numpy as np

from vision.detect.detector_api import DetectionResult, create_detector


def test_detector_contract_always_returns_result():
    config = {
        "detector": {"fallback_enabled": True},
        "yolo": {"model_path": "missing.pt", "imgsz": 640, "min_conf_by_class": {"workpiece": 0.55}},
    }
    detector = create_detector(config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert isinstance(result, DetectionResult)
    assert result.image_size == (120, 160)
    assert result.source in {"none", "fallback", "yolo"}
    assert set(result.confidences.keys()) == {"workpiece", "clamp", "hand", "tool"}


def test_masks_binary_when_present():
    config = {
        "detector": {"fallback_enabled": True},
        "yolo": {"model_path": "missing.pt", "imgsz": 640, "min_conf_by_class": {"workpiece": 0.55}},
    }
    detector = create_detector(config)

    frame = np.full((80, 120, 3), 255, dtype=np.uint8)
    result = detector.detect(frame)
    if result.workpiece_mask is not None:
        vals = set(np.unique(result.workpiece_mask).tolist())
        assert vals.issubset({0, 1, False, True})
