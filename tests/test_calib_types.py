from vision.calib.types import CalibrationMeta, PlaneCalibration


def test_plane_calibration_roundtrip_with_meta():
    payload = {
        "H_px_to_mm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "workspace_mm": {"width": 800, "height": 500},
        "origin_mm": "bottom_left",
        "notes": "n",
        "rmse_mm": 0.25,
        "meta": CalibrationMeta(schema_version="1.1", camera_id="cam01").to_dict(),
    }
    plane = PlaneCalibration.from_dict(payload)
    out = plane.to_dict()
    assert out["rmse_mm"] == 0.25
    assert out["meta"]["camera_id"] == "cam01"
