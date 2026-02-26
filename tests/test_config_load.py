import json

from vision.detect.detector_api import load_config


def test_load_config_reads_json(tmp_path):
    path = tmp_path / "vision_config.json"
    payload = {"detector": {"fallback_enabled": True}, "calibration": {"origin": "bottom_left"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_config(str(path))
    assert loaded["detector"]["fallback_enabled"] is True
    assert loaded["calibration"]["origin"] == "bottom_left"
