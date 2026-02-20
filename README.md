# Vision-cnc YOLO Segmentation Subsystem

Production-oriented segmentation module for CNC Vision with robust YOLO + fallback behavior and a PySide6 operator UI.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/vision_detect_smoketest.py --image path/to/image.jpg
python scripts/vision_ui.py
```

## Main components
- `vision/detect/detector_api.py`: contract + detector factory.
- `vision/detect/yolo_seg.py`: YOLOv8-seg wrapper.
- `vision/detect/fallback_seg.py`: CPU fallback segmentation.
- `vision/dataset/hard_cases.py`: active-learning hard-case export.
- `vision/ui/main_window.py`: PySide6 operator UI for camera/image workflow.

## UI workflow
- Open camera stream or load single image.
- Run continuous/one-shot detection.
- See live mask overlay + confidence panel + fail reason.
- Save snapshot (`runs/ui/<timestamp>/overlay.png`, `meta.json`).

## Logging
Runtime logs are written to `log.txt`.
