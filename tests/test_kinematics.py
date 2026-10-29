"""Position and pose IK on the real URDF. No serial port."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.kinematics import JOINTS, NEAR_STEP_DEG, Kinematics
from so101_utils.config import load_robot, validated_limits

URDF = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"


@pytest.fixture(scope="module")
def kin():
    if not URDF.is_file():
        pytest.skip("URDF not downloaded")
    return Kinematics(str(URDF))


def test_ik_matches_a_known_base_pose(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    goal_joints = dict(start)
    goal_joints["shoulder_pan"] = 15.0
    goal_joints["wrist_flex"] = -20.0
    goal_joints["wrist_roll"] = 25.0
    matrix = kin.fk(goal_joints)
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    solved = kin.position_ik(start, kin.tool_point(matrix), limits, rpy=rpy)
    pose = kin.fk(solved)
    assert np.linalg.norm(kin.tool_point(pose) - kin.tool_point(matrix)) < 0.015
    error = Rotation.from_matrix(matrix[:3, :3].T @ pose[:3, :3]).magnitude()
    assert np.rad2deg(error) < 1.0


def test_a_short_translation_stays_near_the_start(kin):
    limits = validated_limits(load_robot())
    start = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -90.0,
        "elbow_flex": 90.0,
        "wrist_flex": 45.0,
        "wrist_roll": 0.0,
    }
    target = kin.tool_point(kin.fk(start)) + np.array([0.005, 0.0, 0.0])
    solved = kin.position_ik(start, target, limits)
    assert max(abs(solved[name] - start[name]) for name in JOINTS) <= NEAR_STEP_DEG
    assert np.linalg.norm(kin.tool_point(kin.fk(solved)) - target) < 0.015


def test_ik_returns_to_a_known_tcp(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    goal_joints = dict(start)
    goal_joints["shoulder_pan"] = 20.0
    goal_joints["elbow_flex"] = -25.0
    target = kin.fk(goal_joints)[:3, 3]
    solved = kin.position_ik(start, target, limits)
    reached = kin.fk(solved)[:3, 3]
    assert np.linalg.norm(reached - target) < 0.015


def test_downward_points_the_gripper_z_axis_down(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    solved = kin.position_ik(start, target, limits, downward=True)
    pose = kin.fk(solved)
    assert pose[2, 2] < -0.95
    assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015


def test_ee_tool_offset_follows_the_finger_axis(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = -0.020
    try:
        solved = kin.position_ik(start, target, limits, downward=True)
        pose = kin.fk(solved)
        tool = kin.tool_point(pose)
        assert np.linalg.norm(tool - target) < 0.015
        assert pose[2, 2] < -0.95
        # The tool point sits on gripper −X, the other side of the red arrow.
        offset = tool - pose[:3, 3]
        assert np.dot(offset, pose[:3, 0]) < 0
        assert np.linalg.norm(offset) == pytest.approx(0.020, abs=0.005)
    finally:
        kin.ee_tool_offset = previous


def test_ik_accepts_a_start_just_past_the_stop(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    start["shoulder_lift"] = limits["shoulder_lift"][0] - 0.01
    target = kin.fk({name: 0.0 for name in JOINTS})[:3, 3]
    solved = kin.position_ik(start, target, limits)
    assert np.linalg.norm(kin.fk(solved)[:3, 3] - target) < 0.015
    start["shoulder_lift"] = limits["shoulder_lift"][0] - 5.0
    with pytest.raises(ValueError):
        kin.position_ik(start, target, limits)


def test_ik_recovers_when_the_arm_is_folded(kin):
    """A folded seed can miss a point that the home pose reaches."""
    limits = validated_limits(load_robot())
    folded = {
        "shoulder_pan": 0.0,
        "shoulder_lift": limits["shoulder_lift"][0],
        "elbow_flex": limits["elbow_flex"][0],
        "wrist_flex": limits["wrist_flex"][0],
        "wrist_roll": 0.0,
    }
    target = (0.23428342, 0.0300049, 0.10)
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = -0.020
    try:
        solved = kin.position_ik(folded, target, limits)
        pose = kin.fk(solved)
        assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015
    finally:
        kin.ee_tool_offset = previous


def test_ik_rejects_a_point_outside_reach(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    with pytest.raises(ValueError):
        kin.position_ik(start, (5.0, 5.0, 5.0), limits)
