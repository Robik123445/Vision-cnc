"""FastAPI server exposing Vision Bridge snapshot/live/replay endpoints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from vision_bridge.core import VisionCoreAdapter
from vision_bridge.live import LiveSessionManager
from vision_bridge.replay import load_replay, save_replay
from vision_bridge.schemas import (
    HealthResponse,
    LivePollResponse,
    LiveStartRequest,
    LiveStartResponse,
    LiveStopRequest,
    ReplayLoadRequest,
    ReplaySaveResponse,
    SnapshotRequest,
    SnapshotResponse,
    StatusResponse,
)
from vision_bridge.version import API_VERSION

LOGGER = logging.getLogger("vision.bridge.server")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()],
    )

app = FastAPI(title="Vision Bridge", version=API_VERSION)
adapter = VisionCoreAdapter()
live_manager = LiveSessionManager()
last_snapshot: Optional[SnapshotResponse] = None
last_frame: Optional[np.ndarray] = None
loaded_replay_id: Optional[str] = None


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return bridge service health and API version."""

    return HealthResponse(status="ok", api_version=API_VERSION)


@app.post("/v1/snapshot", response_model=SnapshotResponse)
def snapshot(req: SnapshotRequest) -> SnapshotResponse:
    """Run one-shot snapshot from camera or loaded replay source."""

    global last_snapshot, last_frame, loaded_replay_id

    if req.source == "replay":
        rid = req.replay_id or loaded_replay_id
        if not rid:
            raise HTTPException(status_code=400, detail="Replay source requested but replay_id not provided")
        try:
            snap, frame = load_replay(rid)
            last_snapshot, last_frame, loaded_replay_id = snap, frame, rid
            return snap
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Replay not found: {exc}") from exc

    frame, reason = adapter.capture_frame()
    if frame is None:
        snap = adapter._error_snapshot(reason)
    else:
        snap = adapter.build_snapshot_from_frame(frame, req.min_confidence)

    last_snapshot, last_frame = snap, frame
    return snap


@app.post("/v1/live/start", response_model=LiveStartResponse)
def live_start(req: LiveStartRequest) -> LiveStartResponse:
    """Start live monitoring using previous or fresh baseline snapshot."""

    global last_snapshot

    if req.baseline == "new_snapshot" or last_snapshot is None:
        baseline = snapshot(SnapshotRequest(source="camera", min_confidence=0.25))
    else:
        baseline = last_snapshot

    live_id, baseline_frame_id = live_manager.start(
        baseline=baseline,
        drift_mm=req.drift_threshold_mm,
        drift_deg=req.drift_threshold_deg,
        poll_ms=req.poll_interval_ms,
    )
    return LiveStartResponse(status="OK", live_id=live_id, baseline_frame_id=baseline_frame_id)


@app.get("/v1/live/poll", response_model=LivePollResponse)
def live_poll(live_id: str) -> LivePollResponse:
    """Poll current drift status of an active live session."""

    if live_id not in live_manager.live_sessions:
        raise HTTPException(status_code=404, detail="Unknown live_id")

    current = snapshot(SnapshotRequest(source="camera", min_confidence=0.25))
    return live_manager.poll(live_id, current)


@app.post("/v1/live/stop", response_model=StatusResponse)
def live_stop(req: LiveStopRequest) -> StatusResponse:
    """Stop active live session."""

    live_manager.stop(req.live_id)
    return StatusResponse(status="OK")


@app.post("/v1/replay/save", response_model=ReplaySaveResponse)
def replay_save() -> ReplaySaveResponse:
    """Save last snapshot and frame into replay directory."""

    if last_snapshot is None or last_frame is None:
        raise HTTPException(status_code=400, detail="No snapshot available to save")
    replay_id, path = save_replay(last_snapshot, last_frame)
    return ReplaySaveResponse(status="OK", replay_id=replay_id, path=path)


@app.post("/v1/replay/load", response_model=StatusResponse)
def replay_load(req: ReplayLoadRequest) -> StatusResponse:
    """Load replay data into service state for subsequent replay snapshots."""

    global last_snapshot, last_frame, loaded_replay_id

    snap, frame = load_replay(req.replay_id)
    last_snapshot, last_frame, loaded_replay_id = snap, frame, req.replay_id
    return StatusResponse(status="OK")


def main() -> int:
    """CLI entrypoint for running Vision Bridge server."""

    parser = argparse.ArgumentParser(description="Vision Bridge FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8181)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("vision_bridge.server:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
