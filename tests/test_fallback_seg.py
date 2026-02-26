import pytest

pytest.importorskip("cv2", reason="OpenCV runtime not available in this environment", exc_type=ImportError)

import numpy as np

from vision.detect.fallback_seg import FallbackSegmenter


def test_fallback_detects_largest_object():
    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255

    result = seg.detect(frame)
    assert result.ok is True
    assert result.debug["source"] == "fallback"
    assert result.masks["workpiece"] is not None
    assert result.fail_reason == ""


def test_fallback_no_object():
    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = seg.detect(frame)
    assert result.ok is False
    assert result.masks["workpiece"] is None
    assert result.fail_reason == "fallback_no_object"
