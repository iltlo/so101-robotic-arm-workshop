"""Ray / plane geometry. No arm and no camera hardware."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_utils.transforms import Intrinsics, base_link_plane, intersect_pixel, normalized_ray


def _pose(rvec, xyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    result[:3, 3] = xyz
    return result


def test_center_pixel_hits_base_plane_at_z0():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    # Camera looks down the base -Z axis from 0.4 m above the plane.
    base_T_camera = _pose([np.pi, 0, 0], [0.1, 0.0, 0.4])
    plane = base_link_plane()
    point = intersect_pixel((320, 240), intr, base_T_camera, plane["point"], plane["normal"])
    assert np.allclose(point, [0.1, 0.0, 0.0], atol=1e-6)
    hover = point + np.array([0.0, 0.0, 0.02])
    assert np.allclose(hover, [0.1, 0.0, 0.02])


def test_click_outside_the_image_is_rejected():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    with pytest.raises(ValueError):
        intersect_pixel((-1, 240), intr, np.eye(4), [0, 0, 0], [0, 0, 1])


def test_inverse_brown_ray_matches_realsense():
    rs = pytest.importorskip("pyrealsense2")
    intr = Intrinsics(640, 480, 500, 500, 320, 240,
                      (0.12, -0.04, 0.001, 0.002, 0.0), "inverse_brown_conrady")
    native = rs.intrinsics()
    native.width, native.height = intr.width, intr.height
    native.fx, native.fy, native.ppx, native.ppy = intr.fx, intr.fy, intr.cx, intr.cy
    native.coeffs = list(intr.distortion)
    native.model = rs.distortion.inverse_brown_conrady
    ray = normalized_ray((500, 280), intr)
    expected = rs.rs2_deproject_pixel_to_point(native, [500.0, 280.0], 1.0)
    assert np.allclose(ray, expected[:2], atol=1e-6)
