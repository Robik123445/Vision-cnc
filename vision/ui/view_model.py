"""UI view-model helpers for presenting detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from vision.detect.detector_api import DetectionResult


SOURCE_LABELS = {
    "yolo": "YOLO",
    "fallback": "Fallback",
    "none": "žiadny",
}

FAIL_REASON_LABELS = {
    "ok": "v poriadku",
    "idle": "čaká",
    "model_not_loaded": "model nie je načítaný",
    "no_detections": "nič sa nenašlo",
    "fallback_no_object": "fallback nenašiel objekt",
    "hand_detected": "detegovaná ruka",
    "low_confidence_workpiece": "nízka istota obrobku",
}

CLASS_LABELS = {
    "workpiece": "obrobok",
    "clamp": "upínka",
    "hand": "ruka",
    "tool": "nástroj",
}


@dataclass
class DetectionViewState:
    """Presentation-friendly detection state used by the Qt layer."""

    source: str
    fail_reason: str
    inference_ms: float
    confidences: Dict[str, float]
    status_line: str


def detection_to_view_state(result: DetectionResult) -> DetectionViewState:
    """Convert detection contract into readable UI strings and values."""

    fail_reason_key = result.fail_reason or "ok"
    fail_reason = FAIL_REASON_LABELS.get(fail_reason_key, fail_reason_key.replace("_", " "))
    source_label = SOURCE_LABELS.get(result.source, result.source)
    status_line = (
        f"zdroj={source_label} | stav={fail_reason} | "
        f"{CLASS_LABELS['workpiece']}={result.confidences.get('workpiece', 0.0):.2f} | "
        f"{CLASS_LABELS['clamp']}={result.confidences.get('clamp', 0.0):.2f} | "
        f"{CLASS_LABELS['hand']}={result.confidences.get('hand', 0.0):.2f} | "
        f"{CLASS_LABELS['tool']}={result.confidences.get('tool', 0.0):.2f}"
    )
    return DetectionViewState(
        source=result.source,
        fail_reason=fail_reason,
        inference_ms=result.inference_ms,
        confidences=result.confidences,
        status_line=status_line,
    )
