"""Fixed-camera hand-eye, unknown EE tag mount, and tag-on-table plane.

Notation aTb maps points expressed in b into a; translations always metres.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .config import transform
from .geometry import Intrinsics, normalized_ray


@dataclass
class Sample:
    base_T_gripper: np.ndarray
    camera_T_tag: np.ndarray


def _average(poses):
    """Average translations and incrementally SLERP camera rotations (short arc)."""
    translations = np.array([p[:3, 3] for p in poses])
    rots = Rotation.from_matrix(np.array([p[:3, :3] for p in poses]))
    orientation = rots[0]
    for count in range(1, len(rots)):
        orientation = Slerp([0, 1], Rotation.from_quat(
            [orientation.as_quat(), rots[count].as_quat()]))([1 / (count + 1)])[0]
    result = np.eye(4)
    result[:3, :3] = orientation.as_matrix()
    result[:3, 3] = translations.mean(axis=0)
    return result


def residuals(samples, base_T_camera, gripper_T_tag):
    translation = []
    degrees = []
    for sample in samples:
        predicted = base_T_camera @ sample.camera_T_tag
        measured = sample.base_T_gripper @ gripper_T_tag
        translation.append(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
        degrees.append(np.rad2deg((Rotation.from_matrix(predicted[:3, :3]).inv() *
                                   Rotation.from_matrix(measured[:3, :3])).magnitude()))
    return np.array(translation), np.array(degrees)


def measured_gripper_T_tag() -> np.ndarray:
    """User-measured tag center/frame from gripper_link; provisional metres.

    Translation is expressed along the gripper axes. From the former +90° Y
    pose, turn the tag frame a further 180° about its own Z (postmultiply).
    Re-measure and verify tag-frame axes against camera PnP before acceptance.
    """
    result = np.eye(4)
    result[:3, 3] = [-0.0372, 0.0, -0.0284]
    gripper_R_tag_before = Rotation.from_euler("y", 90, degrees=True)
    tag_R_z_half_turn = Rotation.from_euler("z", 180, degrees=True)
    result[:3, :3] = (gripper_R_tag_before * tag_R_z_half_turn).as_matrix()
    return result


def camera_candidates(samples: list[Sample], gripper_T_tag: np.ndarray) -> list[np.ndarray]:
    """One base_T_camera hypothesis per measured gripper/tag image pair."""
    mount = transform(gripper_T_tag)
    return [transform(s.base_T_gripper) @ mount @ np.linalg.inv(transform(s.camera_T_tag))
            for s in samples]


def solve_eye_to_hand_known_tag(samples: list[Sample], gripper_T_tag: np.ndarray,
                                *, max_error_m=0.020, max_error_deg=8.0) -> dict:
    """Solve fixed camera pose with a measured gripper-link tag mount.

    base_T_camera_i = base_T_gripper_i * gripper_T_tag * inv(camera_T_tag_i).
    """
    if len(samples) < 6:
        raise ValueError("Collect at least 6 diverse settled poses")
    mount = transform(gripper_T_tag)
    checked = [Sample(transform(s.base_T_gripper), transform(s.camera_T_tag)) for s in samples]
    rotations = Rotation.from_matrix([sample.base_T_gripper[:3, :3] for sample in checked])
    spread = (rotations[0].inv() * rotations).as_rotvec()
    if np.linalg.matrix_rank(spread, tol=0.06) < 2:
        raise ValueError("Robot poses lack independent rotation axes")
    candidates = camera_candidates(checked, mount)
    bTc = transform(_average(candidates))
    errs_m, errs_deg = residuals(checked, bTc, mount)
    if max(errs_m) > max_error_m or max(errs_deg) > max_error_deg:
        raise ValueError(f"Calibration inconsistent: max {max(errs_m):.3f} m / {max_error_m:.3f} m, "
                         f"{max(errs_deg):.1f}° / {max_error_deg:.1f}° allowed")
    return {"base_T_camera": bTc, "gripper_T_tag": mount,
            "max_residual_m": float(max(errs_m)), "max_residual_deg": float(max(errs_deg)),
            "sample_count": len(checked)}


def solve_eye_to_hand(samples: list[Sample], *, max_error_m=0.015,
                      max_error_deg=8.0) -> dict:
    """Solve bTg_i * gTtag = bTc * cTtag_i with OpenCV hand-eye role swap."""
    import cv2

    if len(samples) < 6:
        raise ValueError("Collect at least 6 diverse settled poses")
    checked = [Sample(transform(s.base_T_gripper), transform(s.camera_T_tag)) for s in samples]
    base_T_gripper = [s.base_T_gripper for s in checked]
    tag_T_camera = [np.linalg.inv(s.camera_T_tag) for s in checked]
    rotations = Rotation.from_matrix([p[:3, :3] for p in base_T_gripper])
    spread = (rotations[0].inv() * rotations).as_rotvec()
    if np.linalg.matrix_rank(spread, tol=0.06) < 2:
        raise ValueError("Robot poses lack independent rotation axes")
    R, t = cv2.calibrateHandEye([p[:3, :3] for p in base_T_gripper],
                                [p[:3, 3] for p in base_T_gripper],
                                [p[:3, :3] for p in tag_T_camera],
                                [p[:3, 3] for p in tag_T_camera], method=cv2.CALIB_HAND_EYE_PARK)
    gTtag = np.eye(4)
    gTtag[:3, :3], gTtag[:3, 3] = R, np.asarray(t).reshape(3)
    gTtag = transform(gTtag)
    candidates = [s.base_T_gripper @ gTtag @ np.linalg.inv(s.camera_T_tag) for s in checked]
    bTc = transform(_average(candidates))
    errs_m, errs_deg = residuals(checked, bTc, gTtag)
    if max(errs_m) > max_error_m or max(errs_deg) > max_error_deg:
        raise ValueError(f"Calibration inconsistent: max {max(errs_m):.3f} m, {max(errs_deg):.1f}°")
    return {"base_T_camera": bTc, "gripper_T_tag": gTtag,
            "max_residual_m": float(max(errs_m)), "max_residual_deg": float(max(errs_deg)),
            "sample_count": len(checked)}


def _tag_pose_from_corners(corners, intr: Intrinsics, size_m: float) -> np.ndarray:
    """PnP with tagged square corners; OpenCV IPPE expects TL,TR,BR,BL."""
    import cv2

    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Four ordered tag corners required")
    half = size_m / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0],
                       [-half, -half, 0]], dtype=float)
    success, rvec, tvec = cv2.solvePnP(points, corners, intr.matrix(),
                                       np.asarray(intr.distortion or [0, 0, 0, 0, 0], dtype=float),
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Tag pose ambiguous or behind camera")
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec[:, 0]
    return transform(pose)


def find_tag(image_bgr, intr: Intrinsics, *, tag_id=0, tag_size_m=0.04,
             detector=None) -> tuple[np.ndarray, np.ndarray]:
    """Return camera_T_tag in pupil_apriltags' tag frame and ordered RGB corners."""
    import cv2

    intr.matrix()
    if tag_size_m <= 0 or image_bgr.shape[:2] != (intr.height, intr.width):
        raise ValueError("Tag size or RGB resolution mismatch")
    if detector is None:
        from pupil_apriltags import Detector
        detector = Detector(families="tag36h11")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = [t for t in detector.detect(gray) if t.tag_id == tag_id]
    if len(tags) != 1:
        raise ValueError(f"Expected exactly one tag36h11 ID {tag_id}, found {len(tags)}")
    corners = np.asarray(tags[0].corners, dtype=float)
    # Detector corners follow the tag's own orientation, not fixed image BL/BR/TR/TL.
    # Their native order corresponds to IPPE's (-x,+y), (+x,+y), (+x,-y),
    # (-x,-y) model points; reversing it flips the tag axes relative to pose_R.
    corrected = np.asarray([normalized_ray(tuple(corner), intr) for corner in corners])
    undistorted = corrected * [intr.fx, intr.fy] + [intr.cx, intr.cy]
    pose = _tag_pose_from_corners(undistorted,
                                 Intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy),
                                 tag_size_m)
    return pose, corners


def detect_tag(image_bgr, intr: Intrinsics, *, tag_id=0, tag_size_m=0.04) -> np.ndarray:
    """Return exactly one tag36h11 camera pose; no depth is used."""
    pose, _ = find_tag(image_bgr, intr, tag_id=tag_id, tag_size_m=tag_size_m)
    return pose


def table_plane(base_T_camera, camera_T_table_tag) -> dict:
    bTtag = transform(base_T_camera) @ transform(camera_T_table_tag)
    normal = bTtag[:3, 2].copy()
    if normal[2] < 0:  # workshop table top is above its base (+Z)
        normal *= -1
    if normal[2] < 0.8:
        raise ValueError("Tag not nearly flat on a horizontal table in base coordinates")
    return {"point": bTtag[:3, 3].copy(), "normal": normal}