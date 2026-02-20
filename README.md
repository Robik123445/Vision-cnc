# Vision-cnc YOLO Segmentation Subsystem

Production-oriented segmentation module for CNC Vision with robust YOLO + fallback behavior.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/vision_detect_smoketest.py --image path/to/image.jpg
```

## Main components
- `vision/detect/detector_api.py`: contract + detector factory.
- `vision/detect/yolo_seg.py`: YOLOv8-seg wrapper.
- `vision/detect/fallback_seg.py`: CPU fallback segmentation.
- `vision/dataset/hard_cases.py`: active-learning hard-case export.

## Logging
Runtime logs are written to `log.txt`.
