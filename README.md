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
- `scripts/` — calibration, smoketest, healthcheck, dataset validation, capture and training CLI scripts.

## Documentation
- Calibration guide: `docs/calibration.md`
- Training guide: `docs/training.md`

## Logging
Runtime logs are written to `log.txt`.

## Preflight
Run readiness checks before CNC workflow:
```bash
python scripts/vision_healthcheck.py --config config/vision_config.json
```

Validate dataset before training:
```bash
python scripts/validate_yolo_dataset.py --dataset dataset
```


## Vision Bridge API
Run local bridge service (desktop integration layer):
```bash
python -m vision_bridge.server --host 127.0.0.1 --port 8181
```

Alternative:
```bash
uvicorn vision_bridge.server:app --host 127.0.0.1 --port 8181
```

Endpoints v1:
- `GET /v1/health`
- `POST /v1/snapshot`
- `POST /v1/live/start`
- `GET /v1/live/poll?live_id=...`
- `POST /v1/live/stop`
- `POST /v1/replay/save`
- `POST /v1/replay/load`
