"""Pydantic schemas for deterministic Vision Bridge API contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SnapshotRequest(BaseModel):
    """Request payload for snapshot endpoint."""

    source: Literal["camera", "replay"] = "camera"
    replay_id: Optional[str] = None
    min_confidence: float = 0.25


class MaterialBBox(BaseModel):
    """Bounding box in machine millimeters."""

    min: List[float]
    max: List[float]


class MaterialData(BaseModel):
    """Material geometry in machine coordinate system."""

    material_polygon_mm: List[List[float]]
    material_bbox_mm: MaterialBBox
    material_angle_deg: float
    origin_mm: List[float]


class TransformData(BaseModel):
    """Frame transformation metadata for camera to machine mapping."""

    H_cam_to_machine: List[List[float]]
    px_to_mm_scale: List[float]


class MetricsData(BaseModel):
    """Quality metrics returned with each detection."""

    confidence: float
    error_mm_est: float


class DebugData(BaseModel):
    """Debug details for development and diagnosis."""

    reason: str = ""
    debug_image_path: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class SnapshotResponse(BaseModel):
    """Canonical snapshot response contract."""

    schema_version: str
    status: Literal["OK", "NO_DETECTION", "ERROR"]
    timestamp: str
    frame_id: str
    material: MaterialData
    transforms: TransformData
    metrics: MetricsData
    debug: DebugData


class HealthResponse(BaseModel):
    """Health endpoint response schema."""

    status: Literal["ok"]
    api_version: str


class LiveStartRequest(BaseModel):
    """Live session start options."""

    baseline: Literal["use_last", "new_snapshot"] = "use_last"
    drift_threshold_mm: float = 0.5
    drift_threshold_deg: float = 0.3
    poll_interval_ms: int = 250


class LiveStartResponse(BaseModel):
    """Response after starting live session."""

    status: Literal["OK"]
    live_id: str
    baseline_frame_id: str


class LivePollResponse(BaseModel):
    """Response from polling live drift state."""

    status: Literal["OK", "DRIFT", "LOST", "ERROR"]
    delta_mm: List[float]
    delta_deg: float
    confidence: float
    error_mm_est: float
    timestamp: str


class LiveStopRequest(BaseModel):
    """Live stop request payload."""

    live_id: str


class StatusResponse(BaseModel):
    """Simple status response."""

    status: Literal["OK"]


class ReplayLoadRequest(BaseModel):
    """Replay load request payload."""

    replay_id: str


class ReplaySaveResponse(BaseModel):
    """Replay save response payload."""

    status: Literal["OK"]
    replay_id: str
    path: str
