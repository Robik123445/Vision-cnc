#!/usr/bin/env python3
"""Training wrapper for YOLOv8 segmentation on CPU."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    """Launch training, persist best weights and generate report artifacts."""

    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="workpiece_seg")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.model)
    train_result = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    save_dir = Path(train_result.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    target = Path("models/workpiece-seg.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, target)

    metrics = {
        "mAP50": float(getattr(train_result, "results_dict", {}).get("metrics/mAP50(B)", 0.0)),
        "precision": float(getattr(train_result, "results_dict", {}).get("metrics/precision(B)", 0.0)),
        "recall": float(getattr(train_result, "results_dict", {}).get("metrics/recall(B)", 0.0)),
        "weights": str(target),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_path = save_dir / "confusion_matrix.png"
    report = (
        f"# YOLO Segmentation Training Report\n\n"
        f"- mAP50: {metrics['mAP50']:.4f}\n"
        f"- Precision: {metrics['precision']:.4f}\n"
        f"- Recall: {metrics['recall']:.4f}\n"
        f"- Best weights: `{target}`\n"
        f"- Confusion matrix: `{cm_path}`\n"
    )
    (save_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Training done. Weights copied to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
