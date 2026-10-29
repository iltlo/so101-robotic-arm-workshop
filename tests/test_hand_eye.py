"""Eye-to-hand chain: matrix product, then translation mean and rotation SLERP."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from so101_utils.config import load_hand_eye
from so101_utils.transforms import Sample, average_transforms, solve_eye_to_hand

_AUTO = Path(__file__).parents[1] / "calibration" / "auto_hand_eye.py"
_SPEC = importlib.util.spec_from_file_location("auto_hand_eye", _AUTO)
auto_hand_eye = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(auto_hand_eye)


def _pose(rvec, xyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    result[:3, 3] = xyz
    return result


def _mount():
    result = np.eye(4)
    result[:3, 3] = [-0.0372, 0.0, -0.0284]
    result[:3, :3] = (Rotation.from_euler("y", 90, degrees=True)
                      * Rotation.from_euler("z", 180, degrees=True)).as_matrix()
    return result


def test_rotation_average_is_progressive_slerp():
    rotations = [Rotation.from_euler("z", angle, degrees=True) for angle in (0, 50, 100)]
    samples = [_pose(rotation.as_rotvec(), [0.2, 0, 0]) for rotation in rotations]
    assert np.allclose(average_transforms(samples)[:3, :3],
                       Rotation.from_euler("z", 50, degrees=True).as_matrix(), atol=1e-6)
    three = [Rotation.from_euler(axis, angle, degrees=True)
             for axis, angle in (("x", 20), ("y", 60), ("z", 90))]
    step = Slerp([0, 1], Rotation.from_quat([three[0].as_quat(), three[1].as_quat()]))([0.5])[0]
    expected = Slerp([0, 1], Rotation.from_quat([step.as_quat(), three[2].as_quat()]))([1 / 3])[0]
    poses = [_pose(rotation.as_rotvec(), [0, 0, 0]) for rotation in three]
    assert np.allclose(average_transforms(poses)[:3, :3], expected.as_matrix(), atol=1e-6)


def test_known_mount_recovers_the_camera():
    mount = _mount()
    base_T_camera = _pose([0.2, 0.1, -0.1], [0.18, -0.09, 0.45])
    rng = np.random.default_rng(4)
    samples = []
    for _ in range(8):
        base_T_gripper = _pose(rng.uniform(-0.7, 0.7, 3), rng.uniform(-0.2, 0.2, 3))
        # camera_T_tag = inv(base_T_camera) @ base_T_gripper @ gripper_T_tag
        camera_T_tag = np.linalg.inv(base_T_camera) @ base_T_gripper @ mount
        samples.append(Sample(base_T_gripper, camera_T_tag))
    solved = solve_eye_to_hand(samples, mount, max_error_m=1e-5, max_error_deg=1e-4)
    assert np.allclose(solved["base_T_camera"], base_T_camera, atol=1e-6)


def test_too_few_poses_are_rejected():
    mount = _mount()
    with pytest.raises(ValueError, match="4"):
        solve_eye_to_hand([Sample(np.eye(4), np.eye(4))] * 3, mount)


def test_saved_hand_eye_is_a_rigid_transform():
    hand = load_hand_eye()
    assert hand["base_T_camera"].shape == (4, 4)
    assert hand["max_residual_m"] < 0.02


def test_auto_poses_mirror_across_base_x_with_roll_fixed():
    poses = auto_hand_eye.calibration_poses()
    assert len(poses) == 10
    recorded, mirrored = poses[:5], poses[5:]
    for (x, y, z, roll, pitch, yaw), (mx, my, mz, mroll, mpitch, myaw) in zip(recorded, mirrored):
        assert (mx, my, mz, mroll, mpitch, myaw) == (x, -y, z, roll, pitch, -yaw)
        assert roll == -170.0
        assert mroll == -170.0


def test_auto_poses_are_reachable_on_the_mock_arm():
    from so101_control import SO101

    with SO101.connect(mock=True) as arm:
        for x, y, z, roll, pitch, yaw in auto_hand_eye.calibration_poses():
            arm.move_to(x, y, z, roll, pitch, yaw)
            pose = arm.get_pose()
            assert pose.roll == pytest.approx(-170.0, abs=1.0)
