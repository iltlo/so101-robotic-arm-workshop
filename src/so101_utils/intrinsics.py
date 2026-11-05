"""Chessboard intrinsics. Square size does not change fx, fy, or distortion."""

import numpy as np

from so101_utils.transforms import Intrinsics

# Inner corners of the lab board: 9 by 12 squares. OpenCV also accepts the swap.
INNER_CORNERS = (8, 11)
MIN_VIEWS = 3
# The calibration page and script ask for ten tilted views. The solver accepts three.
WORKSHOP_MIN_VIEWS = 10
# Full-frame search on 1920x1080 is what makes the calibration view lag.
_DETECT_WIDTH = 960


def chessboard_object(inner_corners=INNER_CORNERS, square=1.0) -> np.ndarray:
    """One board on z = 0. `square` only scales the later board pose."""
    cols, rows = inner_corners
    if cols < 2 or rows < 2 or square <= 0:
        raise ValueError("Chessboard needs at least 2 by 2 inner corners")
    points = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square)
    return points


def _pattern_sizes(inner_corners):
    cols, rows = int(inner_corners[0]), int(inner_corners[1])
    sizes = [(cols, rows)]
    if cols != rows:
        sizes.append((rows, cols))
    return sizes


def _detect_gray(gray):
    """Gray image to search, and the scale back to full-frame pixels."""
    import cv2

    width = int(gray.shape[1])
    if width <= _DETECT_WIDTH:
        return gray, 1.0
    scale = _DETECT_WIDTH / width
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def _refine(gray, corners, scale):
    import cv2

    points = np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if scale != 1.0:
        points = points / np.float32(scale)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)


def find_chessboard(image_bgr, inner_corners=INNER_CORNERS):
    """Refined corners and the pattern size that matched, or None."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small, scale = _detect_gray(gray)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for size in _pattern_sizes(inner_corners):
        found, corners = cv2.findChessboardCorners(small, size, flags)
        if found and corners is not None:
            return _refine(gray, corners, scale), size
    if hasattr(cv2, "findChessboardCornersSB"):
        for size in _pattern_sizes(inner_corners):
            found, corners = cv2.findChessboardCornersSB(small, size)
            if found and corners is not None:
                return _refine(gray, corners, scale), size
    return None


def draw_chessboard(image_bgr, corners, message, inner_corners=INNER_CORNERS):
    """Copy of the frame with the board outline and one status line."""
    import cv2

    image = np.ascontiguousarray(image_bgr).copy()
    if corners is not None:
        points = np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        pattern = (int(inner_corners[0]), int(inner_corners[1]))
        cv2.drawChessboardCorners(image, pattern, points, True)
    cv2.putText(image, message, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (20, 220, 20), 2, cv2.LINE_AA)
    return image


def capture_chessboard(image_bgr):
    """Object points and refined corners for one frame, or ValueError."""
    found = find_chessboard(image_bgr, INNER_CORNERS)
    if found is None:
        raise ValueError("No chessboard in this frame.")
    corners, pattern = found
    return chessboard_object(pattern), corners.copy()


def solve_captured(samples, image_size, *, minimum: int = WORKSHOP_MIN_VIEWS):
    """Brown-Conrady fit after `minimum` captures. Returns intrinsics and RMS pixels."""
    if len(samples) < minimum:
        raise ValueError(f"Need {minimum} views, have {len(samples)}.")
    return solve_intrinsics(
        [item[0] for item in samples],
        [item[1] for item in samples],
        image_size,
    )


def solve_intrinsics(object_points, image_points, image_size) -> tuple[Intrinsics, float]:
    """Brown-Conrady intrinsics and the RMS reprojection error, in pixels."""
    import cv2

    if len(object_points) != len(image_points) or len(object_points) < MIN_VIEWS:
        raise ValueError(f"Need at least {MIN_VIEWS} chessboard views")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    rms, matrix, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        list(object_points), list(image_points), (width, height), None, None,
    )
    coeffs = np.asarray(dist, dtype=float).reshape(-1)
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(coeffs).all():
        raise RuntimeError("Camera calibration did not converge")
    intr = Intrinsics(
        width, height,
        float(matrix[0, 0]), float(matrix[1, 1]),
        float(matrix[0, 2]), float(matrix[1, 2]),
        tuple(float(value) for value in coeffs),
        "brown_conrady",
    )
    intr.matrix()
    return intr, float(rms)
