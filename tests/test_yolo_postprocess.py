import pytest

pytest.importorskip("cv2", reason="OpenCV runtime not available in this environment", exc_type=ImportError)

import numpy as np

from vision.detect.postprocess import process_yolo_masks


def test_postprocess_clamp_union_present():
    image_size = (10, 10)
    wp = np.zeros((10, 10), dtype=np.uint8)
    wp[0:5, 0:5] = 1
    clamp1 = np.zeros((10, 10), dtype=np.uint8)
    clamp1[6:8, 6:8] = 1
    clamp2 = np.zeros((10, 10), dtype=np.uint8)
    clamp2[7:9, 7:9] = 1

    masks = [wp, clamp1, clamp2]
    class_ids = [0, 1, 1]
    confs = [0.8, 0.7, 0.75]

    out_masks, out_conf = process_yolo_masks(
        masks=masks,
        class_ids=class_ids,
        confidences=confs,
        class_map={0: "workpiece", 1: "clamp", 2: "hand", 3: "tool"},
        image_size=image_size,
        min_conf_by_class={"workpiece": 0.5, "clamp": 0.5},
    )

    assert out_masks["workpiece"] is not None
    assert out_masks["clamp"] is not None
    assert out_conf["clamp"] == 0.75
