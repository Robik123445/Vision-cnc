from vision.detect.detector_api import DetectionResult
from vision.ui.view_model import detection_to_view_state


def test_detection_to_view_state_status_line_contains_contract_fields():
    result = DetectionResult(
        workpiece_mask=None,
        clamp_mask=None,
        hand_mask=None,
        tool_mask=None,
        confidences={"workpiece": 0.81, "clamp": 0.12, "hand": 0.0, "tool": 0.5},
        source="yolo",
        inference_ms=14.5,
        image_size=(720, 1280),
        fail_reason=None,
    )

    state = detection_to_view_state(result)
    assert state.source == "yolo"
    assert state.fail_reason == "ok"
    assert "workpiece=0.81" in state.status_line
    assert "tool=0.50" in state.status_line
