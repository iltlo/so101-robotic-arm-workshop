"""Chessboard intrinsics from synthetic views. No camera hardware."""

import numpy as np
import pytest

from so101_utils.intrinsics import chessboard_object, draw_chessboard, find_chessboard, solve_intrinsics


def test_draw_chessboard_accepts_corner_array():
    image = np.zeros((80, 120, 3), np.uint8)
    corners = np.arange(8, dtype=np.float64).reshape(4, 2)
    drawn = draw_chessboard(image, corners, "ok", (2, 2))
    assert drawn.shape == image.shape
    assert not np.array_equal(drawn, image)


def test_blank_image_has_no_chessboard():
    image = np.zeros((48, 64, 3), np.uint8)
    assert find_chessboard(image, (6, 9)) is None


def test_solve_recovers_a_known_pinhole():
    cv2 = pytest.importorskip("cv2")
    board = chessboard_object((6, 9), square=0.02)
    matrix = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([[0.08, -0.04, 0.001, -0.001, 0.0]], dtype=float)
    # The board origin is a corner, so shift it back onto the optical axis.
    center = [-0.05, -0.08]
    poses = (
        ([0.12, -0.18, 0.04], [center[0], center[1], 0.90]),
        ([-0.18, 0.10, -0.08], [center[0] + 0.02, center[1] - 0.01, 0.80]),
        ([0.08, 0.20, 0.15], [center[0] - 0.03, center[1] + 0.02, 1.00]),
        ([-0.04, -0.08, -0.20], [center[0] + 0.03, center[1] + 0.02, 0.75]),
        ([0.16, 0.04, -0.12], [center[0] - 0.02, center[1] - 0.03, 0.95]),
    )
    objects = []
    images = []
    for rvec, tvec in poses:
        projected, _ = cv2.projectPoints(
            board, np.array(rvec, dtype=float), np.array(tvec, dtype=float), matrix, distortion,
        )
        points = projected.reshape(-1, 2)
        assert np.all((points[:, 0] >= 0) & (points[:, 0] < 640))
        assert np.all((points[:, 1] >= 0) & (points[:, 1] < 480))
        objects.append(board.copy())
        images.append(projected.astype(np.float32))
    intr, rms = solve_intrinsics(objects, images, (640, 480))
    assert rms < 0.05
    assert intr.distortion_model == "brown_conrady"
    assert intr.fx == pytest.approx(700, abs=0.5)
    assert intr.fy == pytest.approx(710, abs=0.5)
    assert intr.cx == pytest.approx(320, abs=0.5)
    assert intr.cy == pytest.approx(240, abs=0.5)
    assert len(intr.distortion) == 5
    # k3 is weakly observed when the board stays near the image center.
    assert np.allclose(intr.distortion[:4], distortion.reshape(-1)[:4], atol=1e-3)
    assert intr.distortion[4] == pytest.approx(0.0, abs=0.02)


def test_solve_rejects_too_few_views():
    board = chessboard_object()
    corners = np.zeros((board.shape[0], 1, 2), np.float32)
    with pytest.raises(ValueError, match="3"):
        solve_intrinsics([board], [corners], (64, 48))
