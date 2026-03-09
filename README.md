# Vision-cnc YOLO Segmentation Subsystem

CI-safe repository for CNC Vision segmentation, fallback detection, and local operator tooling.

## CI mode

The default test run does not require internet access or runtime-heavy packages:

```bash
pytest -q
```

`requirements.txt` stays intentionally empty so CI can import modules and run pure-Python tests without `numpy`, `cv2`, `torch`, `ultralytics`, or `PySide6`.

## Local runtime mode

For YOLO inference, OpenCV tools, and the PySide6 operator UI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runtime.txt
python scripts/vision_detect_smoketest.py --image path/to/image.jpg
python scripts/vision_ui.py
```

## Main components

- `vision/detect/detector_api.py`: detector contract, config loader, and hybrid orchestration.
- `vision/detect/yolo_seg.py`: YOLOv8-seg wrapper with lazy runtime imports.
- `vision/detect/fallback_seg.py`: CPU fallback segmentation path.
- `vision/dataset/hard_cases.py`: hard-case export for active-learning review.
- `vision/ui/main_window.py`: PySide6 operator UI for camera/image review.
- `vision/calib/`: camera intrinsics and plane calibration helpers.

## Documentation

- `docs/vision_yolo.md`
- `docs/calibration.md`
- `docs/training.md`

## Logging

Runtime logs are written to `log.txt`.
