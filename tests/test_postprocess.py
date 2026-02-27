import pytest

pytest.importorskip("cv2", reason="OpenCV runtime not available in this environment", exc_type=ImportError)

import numpy as np

from vision.detect.postprocess import process_yolo_masks


def test_postprocess_selects_largest_workpiece_and_builds_safety_mask():
    image_size = (20, 20)
    wp_small = np.zeros((20, 20), dtype=np.uint8)
    wp_small[0:2, 0:2] = 1
    wp_big = np.zeros((20, 20), dtype=np.uint8)
    wp_big[2:12, 2:12] = 1

    clamp = np.zeros((20, 20), dtype=np.uint8)
    clamp[14:18, 14:18] = 1
    hand = np.zeros((20, 20), dtype=np.uint8)
    hand[15:19, 1:5] = 1

    masks = [wp_small, wp_big, clamp, hand]
    class_ids = [0, 0, 1, 2]
    confs = [0.9, 0.8, 0.7, 0.75]
    classes = {0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"}

    out_masks, out_conf = process_yolo_masks(masks, class_ids, confs, classes, image_size, {"workpiece": 0.5})

    assert out_masks["workpiece"] is not None
    assert int(out_masks["workpiece"].sum()) >= int(wp_big.sum())
    assert out_masks["safety_mask"] is not None
    assert int(out_masks["safety_mask"].sum()) >= int(clamp.sum())
    assert out_conf["hand"] == 0.75


def test_postprocess_applies_thresholds():
    mask = np.ones((4, 4), dtype=np.uint8)
    out_masks, out_conf = process_yolo_masks(
        masks=[mask],
        class_ids=[0],
        confidences=[0.2],
        class_map={0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"},
        image_size=(4, 4),
        min_conf_by_class={"workpiece": 0.8},
    )

    assert out_masks["workpiece"] is None
    assert out_conf["workpiece"] == 0.0
