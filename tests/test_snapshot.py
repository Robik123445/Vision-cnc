import pytest

pytest.importorskip("fastapi", reason="fastapi unavailable in environment", exc_type=ImportError)
pytest.importorskip("pydantic", reason="pydantic unavailable in environment", exc_type=ImportError)

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from vision_bridge.replay import save_replay
from vision_bridge.schemas import SnapshotResponse
from vision_bridge.server import app


def _example_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "snapshot_example.json"


def _sample_snapshot() -> SnapshotResponse:
    payload = json.loads(_example_path().read_text(encoding="utf-8"))
    payload["frame_id"] = "00000000-0000-0000-0000-000000000222"
    return SnapshotResponse(**payload)


def test_snapshot_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    replay_id, _ = save_replay(_sample_snapshot(), frame, root="replay")

    load_resp = client.post("/v1/replay/load", json={"replay_id": replay_id})
    assert load_resp.status_code == 200

    snap_resp = client.post("/v1/snapshot", json={"source": "replay", "replay_id": replay_id, "min_confidence": 0.25})
    assert snap_resp.status_code == 200

    data = snap_resp.json()
    expected = json.loads(_example_path().read_text(encoding="utf-8"))
    assert set(data.keys()) == set(expected.keys())
    assert data["schema_version"] == "1.0"


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
