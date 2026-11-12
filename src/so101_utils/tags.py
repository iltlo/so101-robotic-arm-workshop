"""AprilTag detection in the camera frame. The tag facts come from robot.yaml."""

from dataclasses import dataclass

import numpy as np

from so101_utils.transforms import Intrinsics, normalized_ray, transform


def tag_rpy_deg(camera_T_tag) -> np.ndarray:
    """Camera-frame XYZ Euler angles of the tag, in degrees."""
    from scipy.spatial.transform import Rotation

    pose = np.asarray(camera_T_tag, dtype=float)
    return Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)


@dataclass(frozen=True)
class Tag:
    """One tag: family, id, black-square edge, and its fixed mount on gripper_link."""

    family: str
    id: int
    size_m: float
    gripper_T_tag: np.ndarray


_LABEL_ORIGIN = (12, 28)
_FONT_SCALE = 0.55
_FONT_THICKNESS = 2


def tag_detector(tag: Tag):
    """pupil_apriltags detector for `tag.family`."""
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise RuntimeError(
            "AprilTags has no wheel for this Python. "
            "Start with: uvx --python 3.13 so101-workshop@latest run"
        ) from exc
    return Detector(families=tag.family)


def _text(image_bgr, message: str, color) -> None:
    import cv2

    cv2.putText(image_bgr, message, _LABEL_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX,
                _FONT_SCALE, color, _FONT_THICKNESS, cv2.LINE_AA)


def draw_tag_overlay(image_bgr, corners, tag: Tag) -> None:
    import cv2

    outline = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_bgr, [outline], True, (0, 255, 0), 2, cv2.LINE_AA)
    center = np.rint(corners.mean(axis=0)).astype(int)
    cv2.circle(image_bgr, tuple(center), 4, (0, 255, 0), -1, cv2.LINE_AA)
    _text(image_bgr, f"{tag.family} ID {tag.id}", (20, 220, 20))


def draw_error(image_bgr, message: str, color=(20, 30, 240)) -> None:
    """Write a detection error over the image."""
    _text(image_bgr, message, color)


def _pose_from_corners(corners, intr: Intrinsics, size_m: float) -> np.ndarray:
    """IPPE on the tag square. Corner order matches pupil_apriltags, not image BL/BR/TR/TL."""
    import cv2

    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Four ordered tag corners required")
    half = size_m / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0],
                       [-half, -half, 0]], dtype=float)
    success, rvec, tvec = cv2.solvePnP(
        points, corners, intr.matrix(),
        np.asarray(intr.distortion or [0, 0, 0, 0, 0], dtype=float),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Tag pose ambiguous or behind the camera")
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec[:, 0]
    return transform(pose)


def find_tag(image_bgr, intr: Intrinsics, tag: Tag, *, detector=None):
    """Return camera_T_tag (metres) and the four RGB corners. No depth."""
    import cv2

    intr.matrix()
    if tag.size_m <= 0 or image_bgr.shape[:2] != (intr.height, intr.width):
        raise ValueError("Tag size or image size does not match the camera")
    if detector is None:
        detector = tag_detector(tag)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = [seen for seen in detector.detect(gray) if seen.tag_id == tag.id]
    if len(tags) != 1:
        raise ValueError(f"Expected one {tag.family} ID {tag.id}, found {len(tags)}")
    corners = np.asarray(tags[0].corners, dtype=float)
    # Native detector order is IPPE (-x,+y), (+x,+y), (+x,-y), (-x,-y).
    undistorted = np.asarray([normalized_ray(tuple(corner), intr) for corner in corners])
    undistorted = undistorted * [intr.fx, intr.fy] + [intr.cx, intr.cy]
    pose = _pose_from_corners(
        undistorted, Intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy),
        tag.size_m,
    )
    return pose, corners
