import pytest

pytest.importorskip("cv2", reason="OpenCV runtime not available in this environment", exc_type=ImportError)

import numpy as np

from vision.calib.plane import compute_homography_px_to_mm, pixel_to_mm


def test_pixel_to_mm_synthetic_homography():
    points_px = [(0, 0), (100, 0), (100, 50), (0, 50)]
    points_mm = [(0, 0), (200, 0), (200, 100), (0, 100)]
    calib = compute_homography_px_to_mm(points_px, points_mm, workspace_mm=(200, 100))

    x_mm, y_mm = pixel_to_mm(50, 25, calib)
    assert np.isclose(x_mm, 100.0, atol=1e-3)
    assert np.isclose(y_mm, 50.0, atol=1e-3)
