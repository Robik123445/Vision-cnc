# Vision-cnc Engine

Production-oriented vision engine for CNC workflows: camera calibration, segmentation, and YOLOv8-seg training.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Main modules
- `vision/calib/` — intrinsic + plane calibration runtime API.
- `vision/detect/` — YOLO-seg + fallback detector with stable `DetectionResult` contract.
- `vision/dataset/hard_cases.py` — hard-case export (`frame.png`, `overlay.png`, `meta.json`).
- `scripts/` — calibration, smoketest, capture and training CLI scripts.

## Documentation
- Calibration guide: `docs/calibration.md`
- Training guide: `docs/training.md`

## Logging
Runtime logs are written to `log.txt`.
