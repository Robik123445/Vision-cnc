# Vision YOLO Segmentačný subsystém

YOLO subsystém rieši **sémantickú segmentáciu** tried `workpiece`, `clamp`, `hand`, `tool`. Geometria a CNC-safe následné kroky ostávajú v nadväzných moduloch.

## Kľúčové rozhranie

- `detector.detect(frame) -> DetectionResult`
- Výstup vždy vracia masky alebo `None`, confidence mapu, `source`, `inference_ms`, `image_size`, `fail_reason`
- Import modulov ostáva CI-safe aj bez `numpy/cv2/torch/ultralytics`

## Konfigurácia

Používa sa `config/vision_config.json`:

- `detector.type`: `yolo` alebo `fallback`
- `detector.fallback_enabled`: zapína fallback segmentáciu
- `yolo.model_path`, `yolo.imgsz`, `yolo.device`
- `yolo.min_conf_by_class`: confidence gating pre každú triedu
- `yolo.min_area_by_class`: odfiltrovanie malých maskových artefaktov
- `dataset.to_label_dir`, `dataset.cooldown_seconds`: hard-case export

## Fallback + gating

1. Primárne beží YOLO wrapper v `vision/detect/yolo_seg.py`.
2. Ak model chýba, nie sú detekcie, alebo je `workpiece` pod confidence prahom, môže sa použiť `vision/detect/fallback_seg.py`.
3. Ak sa objaví `hand_mask`, výsledok dostane `fail_reason="hand_detected"` a uloží sa ako hard case.

## Hard-case dataset

`vision/dataset/hard_cases.py` zapisuje ťažké prípady do:

- `dataset/to_label/<YYYY-MM-DD>/<timestamp>_<reason>/frame.png`
- `dataset/to_label/<YYYY-MM-DD>/<timestamp>_<reason>/meta.json`

Sledované dôvody:

- `no_detections`
- `model_not_loaded`
- `fallback_no_object`
- `hand_detected`
- `low_confidence_workpiece`

## PySide6 UI

Lokálne spustenie:

```bash
pip install -r requirements-runtime.txt
python scripts/vision_ui.py
```

UI obsahuje:

- live preview s overlay masiek
- otvorenie kamery podľa indexu
- načítanie jedného obrázka z disku
- one-shot detekciu a reload detektora z configu
- status panel so `source`, `fail_reason`, confidence a časom inferencie
- snapshot export do `runs/ui/<timestamp>/`

## Smoketest

```bash
python scripts/vision_detect_smoketest.py --image sample.jpg
python scripts/vision_detect_smoketest.py --camera 0
```

Voliteľne sa dá zapnúť aj projekcia mm mriežky:

```bash
python scripts/vision_detect_smoketest.py --image sample.jpg --with_mm_grid
```

Výstupy sú v `runs/<timestamp>/` ako `input.png`, `preview.png`, `summary.json`.

## Lokálna inštalácia runtime závislostí

CI ostáva bez internetu a bez runtime-heavy balíkov. Pre lokálne spustenie YOLO/OpenCV/UI použi:

```bash
pip install -r requirements-runtime.txt
```

`requirements.txt` ostáva CI-safe a neobsahuje runtime-heavy knižnice.
