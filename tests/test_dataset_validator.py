from pathlib import Path

from scripts.validate_yolo_dataset import validate_dataset


def test_validate_dataset_detects_bad_label(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)

    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"x")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 1.2 0.3 0.4 0.5 0.6 0.7\n", encoding="utf-8")

    report = validate_dataset(tmp_path, Path("vision/detect/classes.yaml"))
    assert report["ok"] is False
    assert any("coords_out_of_range" in e for e in report["errors"])
