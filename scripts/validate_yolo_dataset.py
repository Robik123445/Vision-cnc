#!/usr/bin/env python3
"""Validate YOLO segmentation dataset structure and labels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json


def _load_classes(classes_path: Path) -> list[str]:
    """Load ordered class names from classes.yaml minimal parser."""

    if not classes_path.exists():
        return ["workpiece", "clamp", "hand", "tool"]
    classes: list[str] = []
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("classes:"):
            continue
        if ":" in line and line.split(":", 1)[0].strip().isdigit():
            classes.append(line.split(":", 1)[1].strip())
    return classes


def _validate_label_line(line: str, classes_count: int) -> tuple[bool, str]:
    """Validate single YOLO segmentation label line."""

    parts = line.strip().split()
    if len(parts) < 7:
        return False, "polygon_too_short"
    if (len(parts) - 1) % 2 != 0:
        return False, "polygon_not_pairs"

    try:
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return False, "parse_error"

    if cls < 0 or cls >= classes_count:
        return False, "class_out_of_range"
    if any(c < 0.0 or c > 1.0 for c in coords):
        return False, "coords_out_of_range"
    return True, "ok"


def validate_dataset(dataset_root: Path, classes_path: Path) -> dict:
    """Run dataset checks and return a report used before training."""

    classes = _load_classes(classes_path)
    report = {
        "classes": classes,
        "errors": [],
        "stats": {"train_images": 0, "val_images": 0, "train_labels": 0, "val_labels": 0},
    }

    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            report["errors"].append(f"missing_split:{split}")
            continue

        images = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = list(labels_dir.glob("*.txt"))
        report["stats"][f"{split}_images"] = len(images)
        report["stats"][f"{split}_labels"] = len(labels)

        image_stems = {p.stem for p in images}
        for label in labels:
            if label.stem not in image_stems:
                report["errors"].append(f"orphan_label:{split}:{label.name}")
            for idx, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                ok, reason = _validate_label_line(line, len(classes))
                if not ok:
                    report["errors"].append(f"label_error:{split}:{label.name}:{idx}:{reason}")

    report["ok"] = len(report["errors"]) == 0
    return report


def main() -> int:
    """CLI entrypoint for dataset validation script."""

    parser = argparse.ArgumentParser(description="Validate YOLO segmentation dataset")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--classes", default="vision/detect/classes.yaml")
    parser.add_argument("--out", default="runs/dataset_validation.json")
    args = parser.parse_args()

    report = validate_dataset(Path(args.dataset), Path(args.classes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
