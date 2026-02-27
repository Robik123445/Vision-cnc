# YOLOv8 Segmentation Training

## Dataset layout (Ultralytics YOLO seg)
```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

- Label format: YOLO segmentation polygons.
- Recommended tools: CVAT or Label Studio (export to YOLO segmentation).
- Class order must match `vision/detect/classes.yaml`.

## Raw data capture
```bash
python scripts/capture_dataset.py --out dataset/raw --every_n_frames 8 --note "plywood_800x500" --machine_state "idle"
```

## Training on CPU
```bash
python scripts/train_yolo_seg.py --data dataset/data.yaml --imgsz 640 --epochs 80 --batch 8 --device cpu --project runs/train --name workpiece_seg
```

After training:
- best weights are copied to `models/workpiece-seg.pt`
- metrics stored in `runs/train/<name>/metrics.json`
- short report in `runs/train/<name>/report.md`

## Dataset QA (recommended)
```bash
python scripts/validate_yolo_dataset.py --dataset dataset --classes vision/detect/classes.yaml
```

Training with auto-validation:
```bash
python scripts/train_yolo_seg.py --data dataset/data.yaml --device cpu --validate_first
```
