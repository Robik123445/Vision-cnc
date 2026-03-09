from vision.ui.camera_discovery import CameraInfo, clean_device_name, friendly_camera_name, parse_v4l2_devices


def test_parse_v4l2_devices_extracts_video_nodes_only():
    text = """
GENERAL WEBCAM: GENERAL WEBCAM (usb-0000:00:14.0-3):
\t/dev/video2
\t/dev/video3
\t/dev/media1

Integrated Camera: Integrated C (usb-0000:00:14.0-8):
\t/dev/video0
\t/dev/video1
\t/dev/media0
"""

    parsed = parse_v4l2_devices(text)

    assert parsed == [
        ("GENERAL WEBCAM: GENERAL WEBCAM (usb-0000:00:14.0-3)", ["/dev/video2", "/dev/video3"]),
        ("Integrated Camera: Integrated C (usb-0000:00:14.0-8)", ["/dev/video0", "/dev/video1"]),
    ]


def test_friendly_camera_name_strips_usb_suffixes():
    alias = "usb-GENERAL_GENERAL_WEBCAM_JH0319_20210712_v102-video-index0"

    assert friendly_camera_name(alias) == "GENERAL GENERAL WEBCAM JH0319 20210712 v102"


def test_camera_label_includes_device_path():
    camera = CameraInfo(name="GENERAL WEBCAM", device_path="/dev/video2", index=2)

    assert camera.label == "GENERAL WEBCAM  |  /dev/video2"


def test_clean_device_name_removes_repeated_vendor_text():
    raw = "GENERAL WEBCAM: GENERAL WEBCAM (usb-0000:00:14.0-3)"

    assert clean_device_name(raw) == "GENERAL WEBCAM"
