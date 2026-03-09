"""Pure-python helpers for model selection and training workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_project_relative(path: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    """Store project-local paths relatively and external ones as absolute strings."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def discover_model_candidates(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Find likely YOLO weight files inside the project."""

    candidates: set[str] = set()
    patterns: Iterable[tuple[Path, str]] = (
        (project_root / "models", "*.pt"),
        (project_root / "runs" / "train", "**/*.pt"),
    )

    for base, pattern in patterns:
        if not base.exists():
            continue
        for path in base.glob(pattern):
            if path.is_file():
                candidates.add(to_project_relative(path, project_root=project_root))

    return sorted(candidates)


def build_training_command(
    python_executable: str,
    *,
    data_path: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    project_dir: str,
    run_name: str,
    model_path: str,
) -> list[str]:
    """Build the training command used by the model training window."""

    return [
        python_executable,
        "scripts/train_yolo_seg.py",
        "--data",
        data_path,
        "--imgsz",
        str(imgsz),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--device",
        device,
        "--project",
        project_dir,
        "--name",
        run_name,
        "--model",
        model_path,
    ]
