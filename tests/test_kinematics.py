"""Position and pose IK on the real URDF. No serial port."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.arm import SO101
from so101_control.kinematics import (
    NEAR_STEP_DEG, TURN_ACCEPT_DEG, TURN_TOLERANCE_M, IKError, Kinematics,
)
from so101_utils.config import load_robot, validated_limits
from so101_utils.model import (
    JOINTS, gripper_rotation_from_rpy, nod_gripper, pitch_about_axis, pitch_axis,
    spin_gripper, tool_rpy,
)

# Recorded from PlaCo before that extra was removed. The live gap was about
# 1e-8 m and 1e-5 deg; these bounds are that rounding, not a frame change.
FK_POSITION_TOL_M = 1e-5
FK_ROTATION_TOL_DEG = 1e-4
# Joint answers of this solver, and the PyBullet answers they replace.
# The known pose is checked against the joints that generated it. PyBullet
# stopped at [14.9980, 0.5335, -1.2774, -19.2381, 24.9915], 1.3 deg short
# on the elbow. The 5 mm step and the downward click have no single answer on
# a five-joint arm; PyBullet's are the second array. This solver stays in that
# family: within 7 deg on the step and 11 deg on the click. PlaCo's soft solve
# of the known pose left the family (about 42 deg at the lift) and still fails.
IK_JOINT_TOL_DEG = 0.2

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


def test_pitch_90_points_the_fingers_down_and_the_angles_rebuild_it(kin):
    fingers = Rotation.from_euler("xyz", [0.0, 90.0, 40.0], degrees=True).apply([1.0, 0.0, 0.0])
    assert fingers == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    # At pitch 90 the wrist twist reads as either roll or yaw. SciPy returns the
    # pair with yaw zero, and that pair still rebuilds the same rotation.
    vertical = gripper_rotation_from_rpy((0.0, 90.0, 40.0))
    led = tool_rpy(vertical)
    assert led == pytest.approx((-40.0, 90.0, 0.0), abs=0.1)
    assert gripper_rotation_from_rpy(led) == pytest.approx(vertical, abs=1e-9)

    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["wrist_flex"] = 90.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(20.0)
    assert level[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[1] == pytest.approx(90.0, abs=0.2)
    assert abs(twisted[0] - level[0]) > 5.0


def test_a_few_degrees_off_straight_down_reports_the_fk_pitch(kin):
    """Near vertical the reported pitch is the FK angle, not a rounded 90."""
    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["shoulder_pan"] = -60.0
        joints["wrist_flex"] = 80.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(15.0)
    assert level[1] == pytest.approx(80.0, abs=0.2)
    assert twisted[1] == pytest.approx(80.0, abs=0.2)
    assert twisted[0] - level[0] == pytest.approx(-15.0, abs=0.5)
    assert twisted[2] == pytest.approx(level[2], abs=0.5)


def test_the_reported_pitch_tracks_the_fk_through_the_vertical_band(kin):
    """A wrist sweep through 70-89° reports each pitch; nothing is rounded up."""
    def pitch(wrist_flex):
        joints = {name: 0.0 for name in JOINTS}
        joints["wrist_flex"] = wrist_flex
        return tool_rpy(kin.fk(joints)[:3, :3])[1]

    seen = [pitch(flex) for flex in (70.0, 75.0, 80.0, 85.0, 89.0)]
    assert seen == pytest.approx([70.0, 75.0, 80.0, 85.0, 89.0], abs=0.1)



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


def test_repeated_y_steps_move_sideways_and_keep_the_tilt(kin):
    """A five-joint arm cannot hold yaw while the pan carries the tool sideways."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        start = tool_rpy(kin.fk(state)[:3, :3])
        for _ in range(5):
            before = kin.tool_point(kin.fk(state))
            target = before + np.array([0.0, 0.005, 0.0])
            state = kin.position_ik(state, target, limits)
            after = kin.tool_point(kin.fk(state))
            assert after[1] - before[1] > 0.0035
            roll, pitch, _ = tool_rpy(kin.fk(state)[:3, :3])
            assert (roll, pitch) == pytest.approx((start[0], start[1]), abs=0.5)
    finally:
        kin.ee_tool_offset = previous


def test_a_turn_holds_the_tool_point(kin):
    """F and V turn the gripper while the reported xyz stays put."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    anchor = kin.tool_point(kin.fk(state))
    for _ in range(4):
        before = kin.fk(state)[:3, :3]
        axis = kin.joint_axes(state)["wrist_flex"]
        goal = nod_gripper(before, axis, 5.0)
        state = kin.position_ik(state, anchor, limits, rotation=goal,
                                follow_yaw=True, local_only=True)
        pose = kin.fk(state)
        assert np.linalg.norm(kin.tool_point(pose) - anchor) <= TURN_TOLERANCE_M
        # A turn owns the whole frame; the full error is what the keys ask for.
        error = Rotation.from_matrix(goal @ pose[:3, :3].T).magnitude()
        assert np.rad2deg(error) < 0.5
    assert tool_rpy(kin.fk(state)[:3, :3])[1] == pytest.approx(20.0, abs=1.0)


def test_a_short_translation_near_vertical_keeps_the_lean(kin):
    """A sideways step near vertical must not throw away the 85° lean."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    state["wrist_flex"] = 85.0
    before = kin.fk(state)
    point = kin.tool_point(before)
    solved = kin.position_ik(state, point + np.array([0.0, 0.005, 0.0]), limits)
    after = kin.fk(solved)
    assert kin.tool_point(after)[1] - point[1] > 0.0035
    assert tool_rpy(after[:3, :3])[:2] == pytest.approx(
        tool_rpy(before[:3, :3])[:2], abs=0.5)


# Live feedback with the tool exactly at the vertical singularity: pitch 90,
# where roll and yaw describe the same rotation.
VERTICAL_POSE = {
    "shoulder_pan": -7.5,
    "shoulder_lift": -1.8,
    "elbow_flex": 5.6,
    "wrist_flex": 86.2,
    "wrist_roll": -8.2,
}

# Live feedback from the pose where the roll and pitch keys stopped moving:
# near vertical, twisted wrist, at the edge of the reach.
USER_POSE = {
    "shoulder_pan": 16.6,
    "shoulder_lift": 9.4,
    "elbow_flex": -5.1,
    "wrist_flex": 89.8,
    "wrist_roll": -138.4,
}


def _press_turn(kin, state, limits, anchor, *, roll=0.0, pitch=0.0):
    """One keyboard turn press, exactly as teleop builds it."""
    before = kin.fk(state)
    turned = spin_gripper(before[:3, :3], roll)
    if pitch:
        turned = nod_gripper(turned, kin.joint_axes(state)["wrist_flex"], pitch)
    solved = kin.position_ik(state, anchor, limits, rotation=turned,
                             follow_yaw=True, local_only=True,
                             tolerance_m=TURN_TOLERANCE_M,
                             tolerance_deg=TURN_ACCEPT_DEG)
    after = kin.fk(solved)
    turned_deg = np.rad2deg(
        Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude())
    return solved, turned_deg, np.linalg.norm(kin.tool_point(after) - anchor)


def test_pitch_steps_at_the_edge_pose_hold_the_point(kin):
    """At the live edge pose a pitch key must turn, not stop at the wrist."""
    limits = validated_limits(load_robot())
    state = dict(USER_POSE)
    anchor = kin.tool_point(kin.fk(state))
    for _ in range(5):
        state, turned, drift = _press_turn(kin, state, limits, anchor, pitch=-5.0)
        assert turned == pytest.approx(5.0, abs=0.5)
        assert drift <= TURN_TOLERANCE_M
    # Holding the point splits each turn across the arm, so the wrist travels
    # further than the 5° frame turn; it must still clear the old 90° stall.
    assert USER_POSE["wrist_flex"] - state["wrist_flex"] > 20.0


def test_roll_steps_at_the_edge_pose_hold_the_point(kin):
    """Both roll directions must turn the tool, not be swallowed by the yaw gate."""
    limits = validated_limits(load_robot())
    for direction in (5.0, -5.0):
        state = dict(USER_POSE)
        anchor = kin.tool_point(kin.fk(state))
        for _ in range(5):
            state, turned, drift = _press_turn(
                kin, state, limits, anchor, roll=direction)
            assert turned == pytest.approx(5.0, abs=0.5)
            assert drift <= TURN_TOLERANCE_M


FLAT_WRIST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 20.0,
    "wrist_roll": 0.0,
}


def _pitch_press(kin, state, limits, anchor, up: bool):
    before = kin.fk(state)
    axis = kin.joint_axes(state)["wrist_flex"]
    goal = nod_gripper(before[:3, :3], pitch_axis(before[:3, :3], axis),
                       5.0 if up else -5.0)
    solved = kin.position_ik(state, anchor, limits, rotation=goal,
                             follow_yaw=True, local_only=True,
                             tolerance_m=TURN_TOLERANCE_M,
                             tolerance_deg=TURN_ACCEPT_DEG)
    after = kin.fk(solved)
    return solved, pitch_about_axis(before[:3, :3], axis), pitch_about_axis(after[:3, :3], axis)


def test_pitch_plus_increases_pitch_below_and_above_vertical(kin):
    """Pitch + nods down through 90, on either side of the base."""
    limits = validated_limits(load_robot())
    poses = (
        FLAT_WRIST_POSE,
        USER_POSE,
        {"shoulder_pan": 0.0, "wrist_flex": 95.0},
        {"shoulder_pan": 110.0, "wrist_flex": 40.0},
        {"shoulder_pan": -110.0, "wrist_flex": 95.0},
    )
    for pose in poses:
        state = {name: float(pose.get(name, 0.0)) for name in JOINTS}
        anchor = kin.tool_point(kin.fk(state))
        state, before, after = _pitch_press(kin, state, limits, anchor, up=True)
        assert after - before > 2.0
        state, before, after = _pitch_press(kin, state, limits, anchor, up=False)
        assert after - before < -2.0


def test_pitch_past_straight_down_is_reported_above_90(kin):
    """The folded Euler triple is pitch 67 with roll and yaw near 180; the nod is 113."""
    joints = {name: 0.0 for name in JOINTS}
    joints["wrist_flex"] = 113.0
    matrix = kin.fk(joints)
    axis = kin.joint_axes(joints)["wrist_flex"]
    reported = tool_rpy(matrix[:3, :3], axis)
    folded = tool_rpy(matrix[:3, :3])
    assert reported[1] == pytest.approx(113.0, abs=0.5)
    assert reported[1] > 90.0
    assert abs(reported[0]) < 20.0
    assert abs(reported[2]) < 20.0
    assert folded[1] == pytest.approx(67.0, abs=1.0)
    assert abs(folded[0]) > 150.0
    assert gripper_rotation_from_rpy(reported) == pytest.approx(matrix[:3, :3], abs=1e-6)

    facing_back = {name: 0.0 for name in JOINTS}
    facing_back["shoulder_pan"] = 180.0
    facing_back["wrist_flex"] = 25.0
    back = kin.fk(facing_back)
    back_axis = kin.joint_axes(facing_back)["wrist_flex"]
    assert tool_rpy(back[:3, :3], back_axis)[1] == pytest.approx(25.0, abs=0.5)

    twisted = {name: 0.0 for name in JOINTS}
    twisted["wrist_flex"] = 30.0
    twisted["wrist_roll"] = 170.0
    twist = kin.fk(twisted)
    twist_rpy = tool_rpy(twist[:3, :3], kin.joint_axes(twisted)["wrist_flex"])
    assert twist_rpy[1] == pytest.approx(30.0, abs=0.5)
    assert abs(twist_rpy[0]) > 150.0


def test_a_pitch_press_at_the_vertical_singularity_leaves_it(kin):
    limits = validated_limits(load_robot())
    state = dict(VERTICAL_POSE)
    matrix = kin.fk(state)
    anchor = kin.tool_point(matrix)
    assert tool_rpy(matrix[:3, :3])[1] == pytest.approx(90.0, abs=0.1)
    turned = nod_gripper(matrix[:3, :3], kin.joint_axes(state)["wrist_flex"], 5.0)
    solved = kin.position_ik(state, anchor, limits,
                             rotation=turned,
                             follow_yaw=True, local_only=True,
                             tolerance_m=TURN_TOLERANCE_M)
    pose = kin.fk(solved)
    # The Euler pitch folds at 90; the frame itself must turn by the whole press.
    turned_deg = np.rad2deg(Rotation.from_matrix(
        matrix[:3, :3].T @ pose[:3, :3]).magnitude())
    assert turned_deg == pytest.approx(5.0, abs=0.5)
    assert tool_rpy(pose[:3, :3])[1] < 89.0
    assert np.linalg.norm(kin.tool_point(pose) - anchor) <= TURN_TOLERANCE_M


def test_a_roll_press_at_the_vertical_singularity_twists_the_tool(kin):
    """At pitch 90 roll is world-Z; a yaw-free solve would swallow the press."""
    limits = validated_limits(load_robot())
    state = dict(VERTICAL_POSE)
    before = kin.fk(state)
    anchor = kin.tool_point(before)
    turned = spin_gripper(before[:3, :3], 5.0)
    solved = kin.position_ik(state, anchor, limits,
                             rotation=turned,
                             follow_yaw=True, local_only=True,
                             tolerance_m=TURN_TOLERANCE_M)
    after = kin.fk(solved)
    turned_deg = np.rad2deg(
        Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude())
    assert turned_deg == pytest.approx(5.0, abs=1.0)
    assert np.linalg.norm(kin.tool_point(after) - anchor) <= TURN_TOLERANCE_M


def test_a_turn_raises_instead_of_jumping_when_it_cannot_hold(kin):
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    anchor = kin.tool_point(kin.fk(state))
    turned = nod_gripper(kin.fk(state)[:3, :3],
                         kin.joint_axes(state)["wrist_flex"], 60.0)
    with pytest.raises(IKError):
        kin.position_ik(state, anchor, limits, rotation=turned,
                        follow_yaw=True, local_only=True)


def test_a_turn_survives_a_mid_workspace_pose(kin):
    """Away from full extension the arm can pitch further while holding the point."""
    limits = validated_limits(load_robot())
    state = kin.position_ik({name: 0.0 for name in JOINTS}, (0.25, 0.0, 0.20),
                            limits)
    anchor = kin.tool_point(kin.fk(state))
    for _ in range(8):
        axis = kin.joint_axes(state)["wrist_flex"]
        turned = nod_gripper(kin.fk(state)[:3, :3], axis, 5.0)
        state = kin.position_ik(state, anchor, limits, rotation=turned,
                                follow_yaw=True, local_only=True)
        assert np.linalg.norm(kin.tool_point(kin.fk(state)) - anchor) <= TURN_TOLERANCE_M
    assert tool_rpy(kin.fk(state)[:3, :3])[1] == pytest.approx(40.0, abs=1.5)


def test_downward_steps_keep_the_twist(kin):
    """Fingers down still leaves the twist, so a sideway step must not spin the jaws."""
    limits = validated_limits(load_robot())
    state = kin.position_ik({name: 0.0 for name in JOINTS}, (0.28, -0.05, 0.04),
                            limits, downward=True)
    start = kin.fk(state)[:3, :3]
    for _ in range(4):
        target = kin.tool_point(kin.fk(state)) + np.array([0.0, 0.005, 0.0])
        state = kin.position_ik(state, target, limits, downward=True)
        # At pitch 90 several roll/yaw pairs describe one frame, so compare the
        # frame itself: the jaws must keep their heading through the step.
        after = kin.fk(state)[:3, :3]
        turned = Rotation.from_matrix(start.T @ after).magnitude()
        assert np.rad2deg(turned) < 0.5


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


def _rotation_error_deg(left, right) -> float:
    delta = Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude()
    return float(np.rad2deg(delta))


# gripper_frame_link, then gripper_link, recorded from PlaCo on this URDF.
_RECORDED_POSES = (
    (
        {name: 0.0 for name in JOINTS},
        np.array([
            [1.0, -0.00001030, -0.00000234, 0.39136147],
            [0.00001017, 0.99881526, -0.04866293, -0.00000921],
            [0.00000284, 0.04866293, 0.99881526, 0.22646971],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [-0.00000601, -0.00001030, -1.0, 0.29323402],
            [-0.04866293, 0.99881526, -0.00001000, -0.00017677],
            [0.99881526, 0.04866293, -0.00000651, 0.23437033],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        {"shoulder_pan": 90.0, "shoulder_lift": -40.0, "elbow_flex": 50.0,
         "wrist_flex": -30.0, "wrist_roll": 20.0},
        np.array([
            [0.00000853, 0.95522288, 0.29588721, 0.03611355],
            [-0.93969250, -0.10119183, 0.32670815, -0.26542692],
            [0.34202047, -0.27804578, 0.89761492, 0.24995594],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [0.29588721, 0.95522288, -0.00000961, 0.03865847],
            [0.32671160, -0.10119183, 0.93969130, -0.17065851],
            [0.89761366, -0.27804578, -0.34202376, 0.22342454],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        {"shoulder_pan": -60.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
         "wrist_flex": 80.0, "wrist_roll": 15.0},
        np.array([
            [0.08682207, -0.95058286, 0.29808415, 0.14731489],
            [0.15039083, 0.30829123, 0.93932908, 0.18377148],
            [-0.98480685, -0.03672537, 0.16972543, 0.07622953],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [0.29808383, -0.95058286, -0.08682316, 0.14094268],
            [0.93932853, 0.30829123, -0.15039428, 0.17650162],
            [0.16972904, -0.03672537, 0.98480622, 0.17419886],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
)


def test_fk_matches_the_recorded_frames(kin):
    for joints, frame, link in _RECORDED_POSES:
        ours = kin.fk(joints)
        assert np.linalg.norm(ours[:3, 3] - frame[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(ours, frame) < FK_ROTATION_TOL_DEG
        got = kin.fk_gripper_link(joints)
        assert np.linalg.norm(got[:3, 3] - link[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(got, link) < FK_ROTATION_TOL_DEG


def test_ik_stays_near_the_recorded_configuration(kin):
    """A reachable pose, a 5 mm step, and a downward click, against recorded joints."""
    limits = validated_limits(load_robot())
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.0
    try:
        seed = {name: 0.0 for name in JOINTS}
        goal = dict(seed)
        goal["shoulder_pan"] = 15.0
        goal["wrist_flex"] = -20.0
        goal["wrist_roll"] = 25.0
        matrix = kin.fk(goal)
        rpy = tool_rpy(matrix[:3, :3])
        cases = [(
            seed, kin.tool_point(matrix),
            {"rpy": tuple(float(angle) for angle in rpy)},
            np.array([15.0, 0.0, 0.0, -20.0, 25.0]),
            np.array([14.9980, 0.5335, -1.2774, -19.2381, 24.9915]),
            2.0,
        )]
        start = {
            "shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
            "wrist_flex": 45.0, "wrist_roll": 0.0,
        }
        pose = kin.fk(start)
        cases.append((
            start, kin.tool_point(pose) + np.array([0.005, 0.0, 0.0]), {},
            np.array([-0.0000, -83.5025, 88.8648, 39.6377, 0.0000]),
            np.array([0.0, -87.8258, 89.5194, 43.2864, 0.0]),
            7.0,
        ))
        cases.append((
            seed, np.array([0.28, -0.05, 0.04]), {"downward": True},
            np.array([11.2623, 35.9458, -37.0912, 91.1456, 14.0516]),
            np.array([11.2604, 30.3480, -26.8648, 86.4439, 14.0504]),
            11.0,
        ))
        for joints, target, options, reference, pybullet, family_deg in cases:
            solved = kin.position_ik(joints, target, limits, **options)
            reached = kin.fk(solved)
            assert np.linalg.norm(kin.tool_point(reached) - target) < 0.015
            if options.get("downward"):
                assert reached[2, 0] < -0.95
            got = np.array([solved[name] for name in JOINTS])
            assert max(abs(got - reference)) <= IK_JOINT_TOL_DEG
            assert max(abs(got - pybullet)) <= family_deg
    finally:
        kin.ee_tool_offset = previous


def test_workshop_entrypoints_keep_the_arm_api():
    source = inspect.getsource(SO101.move_to)
    assert "position_ik" in source
    assert "self._kin.fk" in source
    root = Path(__file__).parents[1]
    for path in ("scripts/keyboard_control.py", "application/click_and_go.py",
                 "application/pick_and_place.py"):
        spec = importlib.util.spec_from_file_location(path.replace("/", "_"), root / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert "move_to(" in (root / "src/so101_control/teleop.py").read_text()
    moves = (root / "src/so101_utils/moves.py").read_text()
    assert "move_to(" in moves
    assert "downward=True" in moves
    assert "move_to(" in (root / "application/pick_and_place.py").read_text()
