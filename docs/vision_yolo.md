# Vision YOLO Segmentačný subsystém

YOLO subsystém je zodpovedný iba za **sémantickú segmentáciu** (workpiece/clamp/hand/tool). Geometrické výpočty (obrys, hull, PCA, safe polygon) robia nadväzné moduly.

## Kľúčové rozhranie
- `detector.detect(frame) -> DetectionResult`
- Výstup vždy obsahuje binárne masky (alebo `None`), confidence mapu, `source`, `inference_ms`, `image_size`, `fail_reason`.

## Konfigurácia
Používa sa `config/vision_config.json`:
- `detector.fallback_enabled` zapína fallback segmentáciu.
- `yolo.model_path` a `yolo.imgsz` pre YOLOv8-seg.
- `yolo.min_conf_by_class` pre confidence gating.
- `dataset.to_label_dir` + `dataset.cooldown_seconds` pre hard-case export.

## Fallback + gating
1. Primárne beží YOLO (`vision/detect/yolo_seg.py`).
2. Ak chýba model, nie sú detekcie, alebo workpiece confidence je pod prahom, použije sa fallback (`vision/detect/fallback_seg.py`).
3. Ak je detegovaná ruka (`hand_mask`), vracia sa `fail_reason="hand_detected"`.

## Hard-case dataset
`vision/dataset/hard_cases.py` ukladá ťažké prípady do:
- `dataset/to_label/<YYYY-MM-DD>/<timestamp>_<reason>/frame.png`
- `dataset/to_label/<YYYY-MM-DD>/<timestamp>_<reason>/meta.json`

Dôvody exportu:
- `no_detections`
- `model_not_loaded`
- `fallback_no_object`
- `hand_detected`
- `low_confidence_workpiece`

## Smoketest
```bash
python scripts/vision_detect_smoketest.py --image sample.jpg
# alebo
python scripts/vision_detect_smoketest.py --camera 0
```
Výstupy sú v `runs/<timestamp>/` (`input.png`, `preview.png`, `summary.json`).
