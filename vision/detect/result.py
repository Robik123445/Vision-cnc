"""Detection result contract shared by all segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class DetectionResult:
    """Unified detector output with safety-first metadata."""

    ok: bool
    fail_reason: str
    masks: Dict[str, np.ndarray | None]
    confidences: Dict[str, float]
    timing_ms: Dict[str, float]
    debug: Dict[str, Any] = field(default_factory=dict)
