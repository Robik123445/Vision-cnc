from pathlib import Path

from vision.ui.training_helpers import build_training_command, discover_model_candidates, to_project_relative


def test_build_training_command_contains_expected_flags():
    command = build_training_command(
        "python3",
        data_path="dataset/data.yaml",
        imgsz=640,
        epochs=80,
        batch=8,
        device="cpu",
        project_dir="runs/train",
        run_name="workpiece_seg",
        model_path="models/base.pt",
    )

    assert command == [
        "python3",
        "scripts/train_yolo_seg.py",
        "--data",
        "dataset/data.yaml",
        "--imgsz",
        "640",
        "--epochs",
        "80",
        "--batch",
        "8",
        "--device",
        "cpu",
        "--project",
        "runs/train",
        "--name",
        "workpiece_seg",
        "--model",
        "models/base.pt",
    ]


def test_discover_model_candidates_finds_project_models(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "runs" / "train" / "demo" / "weights").mkdir(parents=True)
    (tmp_path / "models" / "a.pt").write_text("x", encoding="utf-8")
    (tmp_path / "runs" / "train" / "demo" / "weights" / "best.pt").write_text("x", encoding="utf-8")

    found = discover_model_candidates(project_root=tmp_path)

    assert found == ["models/a.pt", "runs/train/demo/weights/best.pt"]


def test_to_project_relative_keeps_external_absolute_paths(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    inside = project_root / "models" / "demo.pt"
    inside.parent.mkdir()
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.pt"
    outside.write_text("x", encoding="utf-8")

    assert to_project_relative(inside, project_root=project_root) == "models/demo.pt"
    assert to_project_relative(outside, project_root=project_root) == str(outside)
