"""Helpers for dataset capture sessions and metadata."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def sanitize_session_name(text: str) -> str:
    """Convert user-entered session labels into safe directory names."""

    cleaned = re.sub(r"[^\w-]+", "_", text.strip().lower(), flags=re.UNICODE)
    cleaned = cleaned.strip("_")
    return cleaned or "zber"


def prepare_session_dir(output_root: str | Path, session_name: str, now: datetime | None = None) -> Path:
    """Create and return the directory for one dataset capture session."""

    now = now or datetime.now()
    day = now.strftime("%Y-%m-%d")
    safe_name = sanitize_session_name(session_name)
    out_dir = Path(output_root) / day / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_capture_metadata(
    *,
    timestamp: str,
    note: str,
    machine_state: str,
    camera_label: str,
    frame_index: int,
    saved_index: int,
    session_name: str,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    """Build JSON metadata stored next to each captured dataset frame."""

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "note": note,
        "machine_state": machine_state,
        "camera_label": camera_label,
        "frame_index": frame_index,
        "saved_index": saved_index,
        "session_name": session_name,
    }
    if image_size is not None:
        payload["image_size"] = {"width": int(image_size[0]), "height": int(image_size[1])}
    return payload
