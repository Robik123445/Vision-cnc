# Vision-cnc YOLO Segmentation Subsystem

CI-safe repository for CNC Vision segmentation contract and detector orchestration.

## CI mode (no internet, no runtime deps)
```bash
pytest -q
```
Test suite is designed to run even when `numpy/cv2/torch/ultralytics` are unavailable (runtime-heavy tests are auto-skipped).

## Local runtime mode (full YOLO + UI)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
python scripts/vision_detect_smoketest.py --image path/to/image.jpg
python scripts/vision_ui.py
```

## Main components
- `vision/detect/detector_api.py`: detection contract + detector factory.
- `vision/detect/yolo_seg.py`: YOLOv8-seg wrapper.
- `vision/detect/fallback_seg.py`: CPU fallback segmentation.
- `vision/dataset/hard_cases.py`: active-learning hard-case export.
- `vision/ui/main_window.py`: PySide6 operator UI for camera/image workflow.

## Logging
Runtime logs are written to `log.txt`.
