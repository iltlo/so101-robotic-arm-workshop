"""Explicit camera/base transforms and RGB-only ray-to-table intersection (metres)."""

from dataclasses import dataclass
import numpy as np
from .config import transform


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
    """Assumed base_link XY plane at z=0; not a measured physical table."""
    return {"point": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0])}


def normalized_ray(pixel: tuple[float, float], intr: Intrinsics) -> np.ndarray:
    """Correct a RGB pixel to unit-depth camera coordinates for supported models."""
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


def intersect_pixel(pixel: tuple[float, float], intr: Intrinsics, base_T_camera,
                    point_on_plane, plane_normal) -> np.ndarray:
    """Return base-frame intersection; plane_normal must point toward safe hover side."""
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


def workspace_check(position, box: dict) -> None:
    if not isinstance(box, dict) or set(box) != {"x", "y", "z"}:
        raise ValueError("Verified workspace bounds required")
    for axis, value in zip("xyz", position):
        bounds = box[axis]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Invalid workspace bounds")
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{axis} outside verified workspace")


def preview_hover_target(pixel, intr, base_T_camera, plane, clearance_m) -> np.ndarray:
    if clearance_m is None or not 0.015 <= clearance_m <= 0.15:
        raise ValueError("Measured hover clearance required (0.015–0.15 m)")
    point = intersect_pixel(pixel, intr, base_T_camera, plane["point"], plane["normal"])
    normal = np.asarray(plane["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    return point + clearance_m * normal


def hover_target(pixel, intr, base_T_camera, plane, clearance_m, workspace,
                 gripper_T_tcp) -> np.ndarray:
    transform(gripper_T_tcp)  # deliberate measurement gate
    target = preview_hover_target(pixel, intr, base_T_camera, plane, clearance_m)
    workspace_check(target, workspace)
    return target