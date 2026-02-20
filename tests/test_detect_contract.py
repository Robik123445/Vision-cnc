import numpy as np

from vision.detect.detector_api import DetectionResult, Detector, HybridDetector, create_detector


class StubYolo(Detector):
    def __init__(self, result: DetectionResult):
        self._result = result

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self._result


class StubFallback(Detector):
    def __init__(self, result: DetectionResult):
        self._result = result

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self._result


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


def test_hybrid_keeps_fallback_reason_when_yolo_fails():
    yolo_result = DetectionResult(
        workpiece_mask=None,
        clamp_mask=None,
        hand_mask=None,
        tool_mask=None,
        confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
        source="none",
        inference_ms=1.0,
        image_size=(20, 20),
        fail_reason="model_not_loaded",
    )
    fallback_result = DetectionResult(
        workpiece_mask=None,
        clamp_mask=None,
        hand_mask=None,
        tool_mask=None,
        confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
        source="fallback",
        inference_ms=1.0,
        image_size=(20, 20),
        fail_reason="fallback_no_object",
    )

    detector = HybridDetector(
        yolo=StubYolo(yolo_result),
        fallback=StubFallback(fallback_result),
        config={"dataset": {}, "yolo": {"min_conf_by_class": {"workpiece": 0.55}}},
    )
    out = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))
    assert out.fail_reason == "fallback_no_object"


def test_create_detector_supports_fallback_only_mode():
    detector = create_detector({"detector": {"type": "fallback", "fallback_enabled": True}})
    out = detector.detect(np.zeros((30, 30, 3), dtype=np.uint8))
    assert out.source == "fallback"
