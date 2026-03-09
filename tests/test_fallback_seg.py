"""Fallback segmentation tests (auto-skipped when runtime deps are missing)."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from vision.detect.fallback_seg import FallbackSegmenter


def test_fallback_detects_largest_object():
    """Fallback should return the largest connected component as workpiece."""

    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255

    result = seg.detect(frame)
    assert result.ok is True
    assert result.source == "fallback"
    assert result.workpiece_mask is not None
    assert result.masks["workpiece"] is not None
    assert result.fail_reason is None


def test_fallback_no_object():
    """Fallback should report no object on a fully empty frame."""

    seg = FallbackSegmenter({})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = seg.detect(frame)
    assert result.ok is False
    assert result.workpiece_mask is None
    assert result.masks["workpiece"] is None
    assert result.fail_reason == "fallback_no_object"
