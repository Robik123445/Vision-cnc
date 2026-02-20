"""UI view-model helpers for presenting detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from vision.detect.detector_api import DetectionResult


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

    fail_reason = result.fail_reason or "ok"
    status_line = (
        f"source={result.source} | fail={fail_reason} | "
        f"workpiece={result.confidences.get('workpiece', 0.0):.2f} | "
        f"clamp={result.confidences.get('clamp', 0.0):.2f} | "
        f"hand={result.confidences.get('hand', 0.0):.2f} | "
        f"tool={result.confidences.get('tool', 0.0):.2f}"
    )
    return DetectionViewState(
        source=result.source,
        fail_reason=fail_reason,
        inference_ms=result.inference_ms,
        confidences=result.confidences,
        status_line=status_line,
    )
