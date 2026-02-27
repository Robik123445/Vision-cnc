import pytest

pytest.importorskip("pydantic", reason="pydantic unavailable in environment", exc_type=ImportError)

import numpy as np

from vision_bridge.replay import load_replay, save_replay
from vision_bridge.schemas import SnapshotResponse


def _sample_snapshot() -> SnapshotResponse:
    return SnapshotResponse(
        schema_version="1.0",
        status="NO_DETECTION",
        timestamp="2026-01-01T00:00:00+00:00",
        frame_id="00000000-0000-0000-0000-000000000111",
        material={
            "material_polygon_mm": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            "material_bbox_mm": {"min": [0.0, 0.0], "max": [1.0, 1.0]},
            "material_angle_deg": 0.0,
            "origin_mm": [0.5, 0.5],
        },
        transforms={
            "H_cam_to_machine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "px_to_mm_scale": [1.0, 1.0],
        },
        metrics={"confidence": 0.0, "error_mm_est": 0.0},
        debug={"reason": "", "debug_image_path": None, "raw": {}},
    )


def test_replay_roundtrip(tmp_path):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    snapshot = _sample_snapshot()

    replay_id, _ = save_replay(snapshot, frame, root=str(tmp_path))
    loaded_snapshot, loaded_frame = load_replay(replay_id, root=str(tmp_path))

    assert loaded_snapshot.schema_version == "1.0"
    assert loaded_snapshot.frame_id == snapshot.frame_id
    assert loaded_frame.shape == frame.shape
