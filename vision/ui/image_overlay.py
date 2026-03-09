"""Image rendering utilities for detector mask overlays."""

from __future__ import annotations

from typing import Any, Tuple

from vision.detect.detector_api import DetectionResult


def _apply_mask(image: Any, mask: Any, color: Tuple[int, int, int]) -> Any:
    """Apply one binary mask with a given BGR color to an image."""

    out = image.copy()
    out[mask.astype(bool)] = color
    return out


def compose_overlay(frame: Any, result: DetectionResult, alpha: float = 0.35) -> Any:
    """Render all available masks over frame for operator-friendly visualization."""

    import cv2  # type: ignore

    overlay = frame.copy()
    if result.workpiece_mask is not None:
        overlay = _apply_mask(overlay, result.workpiece_mask, (0, 180, 0))
    if result.clamp_mask is not None:
        overlay = _apply_mask(overlay, result.clamp_mask, (0, 0, 220))
    if result.hand_mask is not None:
        overlay = _apply_mask(overlay, result.hand_mask, (0, 220, 220))
    if result.tool_mask is not None:
        overlay = _apply_mask(overlay, result.tool_mask, (220, 120, 0))

    return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
