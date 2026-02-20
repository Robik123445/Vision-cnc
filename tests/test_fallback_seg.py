import numpy as np

from vision.detect.fallback_seg import FallbackSegmenter


def test_fallback_detects_largest_object():
    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255

    result = seg.detect(frame)
    assert result.source == "fallback"
    assert result.workpiece_mask is not None
    assert result.fail_reason is None
    vals = set(np.unique(result.workpiece_mask).tolist())
    assert vals.issubset({0, 1})


def test_fallback_no_object():
    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = seg.detect(frame)
    assert result.workpiece_mask is None
    assert result.fail_reason == "fallback_no_object"
