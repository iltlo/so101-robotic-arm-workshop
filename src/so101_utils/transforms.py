"""4x4 checks, the base-plane ray, and eye-to-hand averaging. Metres."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(value) -> np.ndarray:
    """Return a finite 4x4 rigid transform."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("Invalid transform rotation")
    if np.linalg.det(matrix[:3, :3]) < 0.999:
        raise ValueError("Reflection is not a valid rotation")
    return matrix


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()
    distortion_model: str = "brown_conrady"

    def matrix(self) -> np.ndarray:
        if self.width <= 0 or self.height <= 0 or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid RGB camera intrinsics")
        result = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite RGB camera intrinsics")
        return result


def base_link_plane() -> dict:
    """Base-link XY plane at z = 0. The normal points up."""
    return {"point": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0])}


def normalized_ray(pixel: tuple[float, float], intr: Intrinsics) -> np.ndarray:
    """Undistort a pixel into unit-depth camera coordinates (x, y)."""
    import cv2

    u, v = pixel
    intr.matrix()
    coefficients = np.asarray(intr.distortion or [0, 0, 0, 0, 0], dtype=float)
    if intr.distortion_model == "inverse_brown_conrady":
        import pyrealsense2 as rs

        native = rs.intrinsics()
        native.width, native.height = intr.width, intr.height
        native.fx, native.fy, native.ppx, native.ppy = intr.fx, intr.fy, intr.cx, intr.cy
        native.coeffs = coefficients.tolist()
        native.model = rs.distortion.inverse_brown_conrady
        return np.asarray(rs.rs2_deproject_pixel_to_point(native, [float(u), float(v)], 1.0)[:2])
    if intr.distortion_model in ("brown_conrady", "none"):
        return cv2.undistortPoints(np.array([[[u, v]]], dtype=float), intr.matrix(),
                                   coefficients)[0, 0]
    raise ValueError(f"Unsupported camera distortion: {intr.distortion_model}")


def intersect_pixel(pixel, intr: Intrinsics, base_T_camera, point_on_plane, plane_normal) -> np.ndarray:
    """Intersect a camera ray with a plane. Result is a point in the base frame."""
    u, v = pixel
    if not np.isfinite([u, v]).all() or not (0 <= u < intr.width and 0 <= v < intr.height):
        raise ValueError("Click outside RGB image")
    bTc = transform(base_T_camera)
    origin = bTc[:3, 3]
    ray = normalized_ray((u, v), intr)
    direction = bTc[:3, :3] @ np.array([*ray, 1.0])
    point = np.asarray(point_on_plane, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    if point.shape != (3,) or normal.shape != (3,) or not np.isfinite([*point, *normal]).all():
        raise ValueError("Invalid table plane")
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("Degenerate plane")
    normal /= norm
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-8:
        raise ValueError("Camera ray parallel to table")
    distance = float(normal @ (point - origin) / denominator)
    if distance <= 0:
        raise ValueError("Table intersection behind camera")
    return origin + distance * direction


@dataclass
class Sample:
    base_T_gripper: np.ndarray
    camera_T_tag: np.ndarray


def average_transforms(poses) -> np.ndarray:
    """Mean translation and progressive short-arc SLERP of the rotations."""
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
    """Per-sample tag error: metres of translation, degrees of rotation."""
    translation = []
    degrees = []
    for sample in samples:
        predicted = base_T_camera @ sample.camera_T_tag
        measured = sample.base_T_gripper @ gripper_T_tag
        translation.append(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
        degrees.append(np.rad2deg((Rotation.from_matrix(predicted[:3, :3]).inv() *
                                   Rotation.from_matrix(measured[:3, :3])).magnitude()))
    return np.array(translation), np.array(degrees)


def solve_eye_to_hand(samples: list[Sample], gripper_T_tag: np.ndarray,
                      *, max_error_m=0.020, max_error_deg=8.0) -> dict:
    """Fixed camera, known tag mount.

    Each sample is base_T_camera = base_T_gripper @ gripper_T_tag @ inv(camera_T_tag).
    The saved pose is the average of those samples.
    """
    if len(samples) < 6:
        raise ValueError("Collect at least 6 diverse settled poses")
    mount = transform(gripper_T_tag)
    checked = [Sample(transform(s.base_T_gripper), transform(s.camera_T_tag)) for s in samples]
    rotations = Rotation.from_matrix([sample.base_T_gripper[:3, :3] for sample in checked])
    spread = (rotations[0].inv() * rotations).as_rotvec()
    if np.linalg.matrix_rank(spread, tol=0.06) < 2:
        raise ValueError("Robot poses lack independent rotation axes")
    candidates = [s.base_T_gripper @ mount @ np.linalg.inv(s.camera_T_tag) for s in checked]
    bTc = transform(average_transforms(candidates))
    errs_m, errs_deg = residuals(checked, bTc, mount)
    if max(errs_m) > max_error_m or max(errs_deg) > max_error_deg:
        raise ValueError(f"Calibration inconsistent: max {max(errs_m):.3f} m / {max_error_m:.3f} m, "
                         f"{max(errs_deg):.1f}° / {max_error_deg:.1f}° allowed")
    return {"base_T_camera": bTc, "gripper_T_tag": mount,
            "max_residual_m": float(max(errs_m)), "max_residual_deg": float(max(errs_deg)),
            "sample_count": len(checked)}
