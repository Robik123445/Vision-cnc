"""CI-safe tests for detector contract without third-party runtime deps."""

from vision.detect.detector_api import DetectionResult, Detector, HybridDetector


class StubYolo(Detector):
    """Stub YOLO detector used for pure-python contract tests."""

    def __init__(self, result: DetectionResult):
        self._result = result

    def detect(self, frame):
        """Return pre-built result for deterministic tests."""

        return self._result


class StubFallback(Detector):
    """Stub fallback detector used for pure-python contract tests."""

    def __init__(self, result: DetectionResult):
        self._result = result

    def detect(self, frame):
        """Return pre-built result for deterministic tests."""

        return self._result


def test_detection_result_contract_fields():
    """Ensure contract object carries all required fields and types."""

    result = DetectionResult(
        workpiece_mask=None,
        clamp_mask=None,
        hand_mask=None,
        tool_mask=None,
        confidences={"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
        source="none",
        inference_ms=1.23,
        image_size=(10, 20),
        fail_reason="model_not_loaded",
    )

    assert result.image_size == (10, 20)
    assert set(result.confidences.keys()) == {"workpiece", "clamp", "hand", "tool"}


def test_hybrid_preserves_fallback_reason_when_switching():
    """Verify fallback-native fail reason is returned without YOLO overwrite."""

    yolo_result = DetectionResult(
        workpiece_mask=None,
        clamp_mask=None,
        hand_mask=None,
        tool_mask=None,
        confidences={"workpiece": 0.1, "clamp": 0.0, "hand": 0.0, "tool": 0.0},
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

    out = detector.detect(frame={"fake": True})
    assert out.source == "fallback"
    assert out.fail_reason == "fallback_no_object"
