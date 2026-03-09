"""Helpers for enumerating and opening camera devices for the desktop UI."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CameraInfo:
    """Stable UI-facing descriptor for one camera endpoint."""

    name: str
    device_path: str
    index: Optional[int]
    alias: Optional[str] = None

    @property
    def label(self) -> str:
        """Display label shown in the camera picker."""

        base = self.name or friendly_camera_name(self.alias or "")
        if not base:
            base = f"Kamera {self.index if self.index is not None else self.device_path}"
        return f"{base}  |  {self.device_path}"


def friendly_camera_name(alias: str) -> str:
    """Turn a `/dev/v4l/by-id` alias into a readable fallback name."""

    cleaned = alias.replace("usb-", "")
    cleaned = re.sub(r"-video-index\d+$", "", cleaned)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_device_name(raw_name: str) -> str:
    """Reduce verbose V4L2 device names into shorter operator-friendly labels."""

    base = re.sub(r"\s*\(.*\)$", "", raw_name).strip()
    if ":" not in base:
        return base

    left, right = [part.strip() for part in base.split(":", 1)]
    if left and right and left.lower() in right.lower():
        return left
    if left and right and right.lower() in left.lower():
        return right
    return left or right or base


def parse_v4l2_devices(text: str) -> list[tuple[str, list[str]]]:
    """Parse `v4l2-ctl --list-devices` output into named device groups."""

    groups: list[tuple[str, list[str]]] = []
    current_name: Optional[str] = None
    current_devices: list[str] = []

    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith("\t"):
            if current_name and current_devices:
                groups.append((current_name, current_devices))
            current_name = line.rstrip(":").strip()
            current_devices = []
            continue

        device = line.strip()
        if device.startswith("/dev/video"):
            current_devices.append(device)

    if current_name and current_devices:
        groups.append((current_name, current_devices))

    return groups


def _camera_index_from_path(device_path: str) -> Optional[int]:
    """Extract numeric index from `/dev/videoN`."""

    match = re.search(r"video(\d+)$", device_path)
    return int(match.group(1)) if match else None


def _read_by_id_aliases() -> dict[str, str]:
    """Map resolved `/dev/video*` paths to stable symlink aliases when available."""

    aliases: dict[str, str] = {}
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        return aliases

    for entry in sorted(by_id_dir.iterdir()):
        if not entry.is_symlink():
            continue
        try:
            aliases[str(entry.resolve())] = entry.name
        except OSError:
            continue
    return aliases


def enumerate_cameras() -> list[CameraInfo]:
    """Enumerate connected camera endpoints from Linux device metadata."""

    aliases = _read_by_id_aliases()
    cameras_by_path: dict[str, CameraInfo] = {}

    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
        )
        for name, devices in parse_v4l2_devices(proc.stdout):
            for device_path in devices:
                cameras_by_path[device_path] = CameraInfo(
                    name=clean_device_name(name),
                    device_path=device_path,
                    index=_camera_index_from_path(device_path),
                    alias=aliases.get(device_path),
                )
    except FileNotFoundError:
        pass

    for device_path, alias in aliases.items():
        cameras_by_path.setdefault(
            device_path,
            CameraInfo(
                name=friendly_camera_name(alias),
                device_path=device_path,
                index=_camera_index_from_path(device_path),
                alias=alias,
            ),
        )

    for entry in sorted(Path("/dev").glob("video*")):
        device_path = str(entry)
        cameras_by_path.setdefault(
            device_path,
            CameraInfo(
                name=f"Kamera {entry.name}",
                device_path=device_path,
                index=_camera_index_from_path(device_path),
                alias=aliases.get(device_path),
            ),
        )

    deduped: dict[str, CameraInfo] = {}
    for camera in sorted(cameras_by_path.values(), key=lambda item: item.index if item.index is not None else 999):
        if camera.alias:
            key = re.sub(r"-video-index\d+$", "", camera.alias)
        else:
            key = camera.name.lower()
        deduped.setdefault(key, camera)

    return list(deduped.values())


def open_camera_capture(cv2: Any, camera: CameraInfo) -> Optional[Any]:
    """Open camera by stable device path first and fall back to numeric index."""

    attempts: list[tuple[Any, Optional[int]]] = []
    v4l2_backend = getattr(cv2, "CAP_V4L2", None)

    if camera.device_path.startswith("/dev/"):
        attempts.append((camera.device_path, v4l2_backend))
        attempts.append((camera.device_path, None))
    if camera.index is not None:
        attempts.append((camera.index, v4l2_backend))
        attempts.append((camera.index, None))

    for source, backend in attempts:
        cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()

    return None
