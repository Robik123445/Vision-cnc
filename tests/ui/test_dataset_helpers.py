from datetime import datetime

from vision.ui.dataset_helpers import build_capture_metadata, prepare_session_dir, sanitize_session_name


def test_sanitize_session_name_normalizes_text():
    assert sanitize_session_name(" Doska 800x500 / test ") == "doska_800x500_test"
    assert sanitize_session_name("") == "zber"


def test_prepare_session_dir_builds_date_and_session_folder(tmp_path):
    out = prepare_session_dir(tmp_path / "dataset" / "raw", "Nova Relacia", now=datetime(2026, 3, 9, 10, 0, 0))

    assert out == tmp_path / "dataset" / "raw" / "2026-03-09" / "nova_relacia"
    assert out.exists()


def test_build_capture_metadata_contains_expected_fields():
    payload = build_capture_metadata(
        timestamp="20260309_100000_000000",
        note="preglejka",
        machine_state="idle",
        camera_label="GENERAL WEBCAM | /dev/video2",
        frame_index=42,
        saved_index=5,
        session_name="doska_800x500",
        image_size=(1280, 720),
    )

    assert payload["note"] == "preglejka"
    assert payload["machine_state"] == "idle"
    assert payload["frame_index"] == 42
    assert payload["saved_index"] == 5
    assert payload["image_size"] == {"width": 1280, "height": 720}
