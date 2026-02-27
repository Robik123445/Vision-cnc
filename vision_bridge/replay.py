"""Replay persistence helpers for snapshot/frame roundtrip."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from vision_bridge.schemas import SnapshotResponse


def save_replay(snapshot: SnapshotResponse, frame: np.ndarray, root: str = "replay") -> tuple[str, str]:
    """Save snapshot JSON and frame PNG into replay/<replay_id>/ folder."""

    replay_id = str(uuid.uuid4())
    replay_dir = Path(root) / replay_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(frame.astype(np.uint8)).save(replay_dir / "frame.png")
    (replay_dir / "snapshot.json").write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return replay_id, str(replay_dir)


def load_replay(replay_id: str, root: str = "replay") -> Tuple[SnapshotResponse, np.ndarray]:
    """Load previously saved replay snapshot and frame from disk."""

    replay_dir = Path(root) / replay_id
    snapshot_data = json.loads((replay_dir / "snapshot.json").read_text(encoding="utf-8"))
    frame = np.asarray(Image.open(replay_dir / "frame.png").convert("RGB"))
    return SnapshotResponse(**snapshot_data), frame
