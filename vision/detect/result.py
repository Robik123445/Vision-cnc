"""Detection result contract shared by all segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


CONFIDENCE_KEYS = ("workpiece", "clamp", "hand", "tool")


@dataclass
class DetectionResult:
    """Unified detector result used by CI-safe tests and local runtime tools."""

    workpiece_mask: Optional[Any]
    clamp_mask: Optional[Any]
    hand_mask: Optional[Any]
    tool_mask: Optional[Any]
    confidences: Dict[str, float]
    source: str
    inference_ms: float
    image_size: Tuple[int, int]
    fail_reason: Optional[str]
    debug: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize confidence keys so consumers can rely on a stable contract."""

        self.confidences = {key: float(self.confidences.get(key, 0.0)) for key in CONFIDENCE_KEYS}

    @property
    def ok(self) -> bool:
        """Backward-compatible success flag used by older scripts/tests."""

        return self.workpiece_mask is not None and not self.fail_reason

    @property
    def timing_ms(self) -> Dict[str, float]:
        """Backward-compatible timing map."""

        return {"inference": float(self.inference_ms)}

    @property
    def masks(self) -> Dict[str, Optional[Any]]:
        """Backward-compatible mask dictionary expected by older runtime helpers."""

        return {
            "workpiece": self.workpiece_mask,
            "clamp": self.clamp_mask,
            "hand": self.hand_mask,
            "tool": self.tool_mask,
            "safety_mask": self._build_safety_mask(),
        }

    def _build_safety_mask(self) -> Optional[Any]:
        """Return clamp/hand union when runtime array ops are available."""

        if self.clamp_mask is None and self.hand_mask is None:
            return None
        if self.clamp_mask is None:
            return self.hand_mask
        if self.hand_mask is None:
            return self.clamp_mask

        try:
            import numpy as np  # type: ignore

            return np.logical_or(self.clamp_mask, self.hand_mask).astype(np.uint8)
        except Exception:
            return None
