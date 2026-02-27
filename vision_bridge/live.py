"""Live mode helpers for drift tracking sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

import numpy as np

from vision_bridge.schemas import LivePollResponse, SnapshotResponse


@dataclass
class LiveSession:
    """Live session state used for repeated drift polling."""

    baseline_snapshot: SnapshotResponse
    drift_threshold_mm: float
    drift_threshold_deg: float
    poll_interval_ms: int


class LiveSessionManager:
    """In-memory manager for live drift sessions."""

    def __init__(self) -> None:
        """Initialize empty session storage."""

        self.live_sessions: Dict[str, LiveSession] = {}

    def start(self, baseline: SnapshotResponse, drift_mm: float, drift_deg: float, poll_ms: int) -> tuple[str, str]:
        """Create a new live session from baseline snapshot."""

        live_id = str(uuid.uuid4())
        self.live_sessions[live_id] = LiveSession(
            baseline_snapshot=baseline,
            drift_threshold_mm=float(drift_mm),
            drift_threshold_deg=float(drift_deg),
            poll_interval_ms=int(poll_ms),
        )
        return live_id, baseline.frame_id

    def stop(self, live_id: str) -> None:
        """Remove live session by id."""

        self.live_sessions.pop(live_id, None)

    def poll(self, live_id: str, current_snapshot: SnapshotResponse) -> LivePollResponse:
        """Compare current snapshot with baseline and return drift status."""

        session = self.live_sessions[live_id]
        if current_snapshot.status == "NO_DETECTION":
            return LivePollResponse(
                status="LOST",
                delta_mm=[0.0, 0.0],
                delta_deg=0.0,
                confidence=current_snapshot.metrics.confidence,
                error_mm_est=current_snapshot.metrics.error_mm_est,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        if current_snapshot.status == "ERROR":
            return LivePollResponse(
                status="ERROR",
                delta_mm=[0.0, 0.0],
                delta_deg=0.0,
                confidence=current_snapshot.metrics.confidence,
                error_mm_est=current_snapshot.metrics.error_mm_est,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        b = np.array(session.baseline_snapshot.material.origin_mm, dtype=float)
        c = np.array(current_snapshot.material.origin_mm, dtype=float)
        d = c - b
        delta_deg = float(current_snapshot.material.material_angle_deg - session.baseline_snapshot.material.material_angle_deg)
        dist = float(np.linalg.norm(d))
        drift = dist > session.drift_threshold_mm or abs(delta_deg) > session.drift_threshold_deg

        return LivePollResponse(
            status="DRIFT" if drift else "OK",
            delta_mm=[float(d[0]), float(d[1])],
            delta_deg=delta_deg,
            confidence=current_snapshot.metrics.confidence,
            error_mm_est=current_snapshot.metrics.error_mm_est,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
