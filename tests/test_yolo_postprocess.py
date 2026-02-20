import numpy as np

from vision.detect.postprocess import process_yolo_masks


def test_postprocess_selects_largest_workpiece_and_unions_clamps():
    image_size = (10, 10)
    wp_small = np.zeros((10, 10), dtype=np.uint8)
    wp_small[0:2, 0:2] = 1
    wp_big = np.zeros((10, 10), dtype=np.uint8)
    wp_big[0:5, 0:5] = 1

    clamp1 = np.zeros((10, 10), dtype=np.uint8)
    clamp1[6:8, 6:8] = 1
    clamp2 = np.zeros((10, 10), dtype=np.uint8)
    clamp2[7:9, 7:9] = 1

    masks = [wp_small, wp_big, clamp1, clamp2]
    class_ids = [0, 0, 1, 1]
    confs = [0.9, 0.8, 0.7, 0.75]
    classes = {0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"}
    mins = {"workpiece": 0.5, "clamp": 0.5, "hand": 0.5, "tool": 0.5}

    out_masks, out_conf = process_yolo_masks(masks, class_ids, confs, classes, image_size, mins)

    assert out_masks["workpiece"] is not None
    assert int(out_masks["workpiece"].sum()) == int(wp_big.sum())
    assert out_masks["clamp"] is not None
    assert int(out_masks["clamp"].sum()) >= int(clamp1.sum())
    assert out_conf["clamp"] == 0.75


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
