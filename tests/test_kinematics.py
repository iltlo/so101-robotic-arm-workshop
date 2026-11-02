"""Position and pose IK on the real URDF. No serial port."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.kinematics import NEAR_STEP_DEG, Kinematics
from so101_utils.config import load_robot, validated_limits
from so101_utils.model import JOINTS, gripper_rotation_from_rpy, tool_rpy

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
    rpy = tool_rpy(matrix[:3, :3])
    solved = kin.position_ik(start, kin.tool_point(matrix), limits, rpy=rpy)
    pose = kin.fk(solved)
    assert np.linalg.norm(kin.tool_point(pose) - kin.tool_point(matrix)) < 0.015
    error = Rotation.from_matrix(matrix[:3, :3].T @ pose[:3, :3]).magnitude()
    assert np.rad2deg(error) < 1.0


def test_facing_forward_is_level_in_the_base(kin):
    rpy = tool_rpy(kin.fk({name: 0.0 for name in JOINTS})[:3, :3])
    assert rpy[0] == pytest.approx(2.8, abs=0.2)
    assert rpy[1] == pytest.approx(0.0, abs=0.2)
    assert rpy[2] == pytest.approx(0.0, abs=0.2)


def test_pitch_90_points_the_fingers_down_and_the_wrist_is_yaw(kin):
    fingers = Rotation.from_euler("xyz", [0.0, 90.0, 40.0], degrees=True).apply([1.0, 0.0, 0.0])
    assert fingers == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert tool_rpy(gripper_rotation_from_rpy((0.0, 90.0, 40.0))) == pytest.approx(
        (0.0, 90.0, 40.0), abs=0.1)

    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["wrist_flex"] = 90.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(20.0)
    assert level[0] == pytest.approx(0.0, abs=0.2)
    assert level[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[0] == pytest.approx(0.0, abs=0.2)
    assert twisted[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[2] - level[2] == pytest.approx(20.0, abs=0.5)


def test_a_few_degrees_off_straight_down_still_puts_the_wrist_in_yaw(kin):
    """The live arm sits a few degrees off vertical. Wrist twist is still yaw."""
    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["shoulder_pan"] = -60.0
        joints["wrist_flex"] = 80.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(15.0)
    assert level[0] == pytest.approx(0.0, abs=0.2)
    assert level[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[0] == pytest.approx(0.0, abs=0.2)
    assert twisted[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[2] - level[2] == pytest.approx(15.0, abs=0.5)


def test_a_side_pose_keeps_roll_and_pitch_apart(kin):
    def angles(**overrides):
        joints = {name: 0.0 for name in JOINTS}
        joints["shoulder_pan"] = 90.0
        joints.update(overrides)
        return tool_rpy(kin.fk(joints)[:3, :3])

    base = angles()
    rolled = angles(wrist_roll=20.0)
    flexed = angles(wrist_flex=-30.0)
    assert rolled[0] - base[0] == pytest.approx(-20.0, abs=0.5)
    assert rolled[1:] == pytest.approx(base[1:], abs=0.5)
    assert flexed[1] - base[1] == pytest.approx(-30.0, abs=0.5)
    assert flexed[0] == pytest.approx(base[0], abs=0.5)
    assert flexed[2] == pytest.approx(base[2], abs=0.5)


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


def test_repeated_x_steps_do_not_rotate_the_wrist(kin):
    """Keyboard nudges translate. Position alone leaves five joints a null space."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        start = tool_rpy(kin.fk(state)[:3, :3])
        for _ in range(5):
            target = kin.tool_point(kin.fk(state)) + np.array([0.005, 0.0, 0.0])
            state = kin.position_ik(state, target, limits)
            assert tool_rpy(kin.fk(state)[:3, :3]) == pytest.approx(start, abs=0.5)
            assert state["wrist_roll"] == pytest.approx(0.0, abs=0.5)
    finally:
        kin.ee_tool_offset = previous


def test_downward_steps_keep_the_twist(kin):
    """Fingers down still leaves the twist, so a sideway step must not spin the jaws."""
    limits = validated_limits(load_robot())
    state = kin.position_ik({name: 0.0 for name in JOINTS}, (0.28, -0.05, 0.04),
                            limits, downward=True)
    start = tool_rpy(kin.fk(state)[:3, :3])
    for _ in range(4):
        target = kin.tool_point(kin.fk(state)) + np.array([0.0, 0.005, 0.0])
        state = kin.position_ik(state, target, limits, downward=True)
        assert tool_rpy(kin.fk(state)[:3, :3]) == pytest.approx(start, abs=0.5)


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


def test_downward_points_the_fingers_down(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    solved = kin.position_ik(start, target, limits, downward=True)
    pose = kin.fk(solved)
    assert pose[2, 0] < -0.95
    assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015


def test_ee_tool_offset_follows_the_finger_axis(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        solved = kin.position_ik(start, target, limits, downward=True)
        pose = kin.fk(solved)
        tool = kin.tool_point(pose)
        assert np.linalg.norm(tool - target) < 0.015
        assert pose[2, 0] < -0.95
        # The tool point sits on gripper +Z.
        offset = tool - pose[:3, 3]
        assert np.dot(offset, pose[:3, 2]) > 0
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
    kin.ee_tool_offset = 0.020
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
