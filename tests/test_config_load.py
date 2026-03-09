import json

from vision.detect.detector_api import load_config, save_config


def test_load_config_reads_json(tmp_path):
    path = tmp_path / "vision_config.json"
    payload = {"detector": {"fallback_enabled": True}, "calibration": {"origin": "bottom_left"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_config(str(path))
    assert loaded["detector"]["fallback_enabled"] is True
    assert loaded["calibration"]["origin"] == "bottom_left"


def test_save_config_writes_roundtrip_json(tmp_path):
    path = tmp_path / "vision_config.json"
    payload = {"yolo": {"model_path": "models/demo.pt"}, "detector": {"type": "yolo"}}

    out = save_config(payload, str(path))

    assert out == str(path)
    assert load_config(str(path)) == payload
