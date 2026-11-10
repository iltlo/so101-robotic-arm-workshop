"""In-memory SO101: connect, move, grip, and the context manager. No serial port."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control import SO101
from so101_control.arm import DEFAULT_EE_TOOL_OFFSET_M, OPEN_PERCENT
from so101_control.backend import MockBackend, OffsetBackend
from so101_control.kinematics import TURN_TOLERANCE_M
from so101_utils.model import (
    JOINTS, gripper_rotation_from_rpy, nod_gripper, tool_pitch, tool_rpy,
)


def test_context_manager_and_speed_clamp():
    with SO101.connect(mock=True) as arm:
        assert arm.mock
        assert arm.ee_tool_offset == DEFAULT_EE_TOOL_OFFSET_M
        assert arm._kin.ee_tool_offset == DEFAULT_EE_TOOL_OFFSET_M
        pose = arm.get_pose()
        assert np.isfinite([pose.x, pose.y, pose.z]).all()
        arm.set_speed(10)
        with pytest.raises(ValueError):
            arm.set_speed(200)
    assert arm._backend.connected is False


def test_move_to_reports_its_goal_pose():
    with SO101.connect(mock=True) as arm:
        seen = []
        arm.on_target(seen.append)
        pose = arm.get_pose()
        goal_xyz = (pose.x, pose.y, pose.z + 0.01)
        arm.move_to(*goal_xyz, wait=False)
        assert len(seen) == 1
        goal = seen[0]
        assert goal.shape == (4, 4)
        assert abs(np.linalg.det(goal[:3, :3]) - 1.0) < 1e-6
        tool = arm._kin.tool_point(goal)
        assert np.linalg.norm(tool - goal_xyz) < 0.015


def test_move_to_continues_from_the_last_command_when_encoders_lag():
    with SO101.connect(mock=True) as arm:
        arm.set_speed(40)
        pose = arm.get_pose()
        arm.move_to(pose.x + 0.03, pose.y, pose.z, wait=False)
        sent = {name: arm._sent[name] for name in JOINTS}
        for name in JOINTS:
            arm._backend.positions[f"{name}.pos"] = 0.0
        packets = []
        send = arm._backend.send_action

        def spy(action):
            packets.append(dict(action))
            return send(action)

        arm._backend.send_action = spy
        arm.move_to(pose.x + 0.03, pose.y, pose.z, wait=False)
        name = max(JOINTS, key=lambda joint: abs(sent[joint]))
        assert abs(sent[name]) > 1.0
        assert abs(packets[0][f"{name}.pos"] - sent[name]) < abs(sent[name])


def test_move_to_seeds_ik_from_the_last_command_without_a_read():
    with SO101.connect(mock=True) as arm:
        pose = arm.get_pose()
        arm.move_to(pose.x + 0.005, pose.y, pose.z, wait=False)

        def no_read():
            raise AssertionError("move_to read the encoders to seed IK")

        arm.get_joints = no_read
        arm.move_to(pose.x + 0.010, pose.y, pose.z, wait=False)


def test_torque_off_forgets_the_last_command():
    with SO101.connect(mock=True) as arm:
        arm.move_joint("shoulder_pan", 1.0)
        assert arm._sent is not None
        arm.relax()
        assert arm._sent is None
        assert arm._goal is None
        # The arm moves while torque is off, so the next step must start there.
        arm._backend.positions["shoulder_pan.pos"] = 20.0
        arm.hold()
        packets = []
        send = arm._backend.send_action

        def spy(action):
            packets.append(dict(action))
            return send(action)

        arm._backend.send_action = spy
        arm.move_joint("shoulder_pan", 1.0)
        assert packets[0]["shoulder_pan.pos"] == pytest.approx(21.0)


def test_hold_point_turns_keep_the_tcp():
    """A pitch key must not walk the reported xyz while the gripper turns."""
    with SO101.connect(mock=True) as arm:
        anchor = arm.get_pose()
        for _ in range(4):
            pose = arm.get_pose()
            frame = arm.gripper_frame_pose()
            turned = nod_gripper(frame[:3, :3], arm.nod_axis(), 5.0)
            arm.move_to(anchor.x, anchor.y, anchor.z, rotation=turned,
                        hold_point=True, wait=False)
            moved = arm.get_pose()
            assert np.linalg.norm(
                [moved.x - anchor.x, moved.y - anchor.y, moved.z - anchor.z]
            ) <= TURN_TOLERANCE_M
            assert moved.pitch - pose.pitch == pytest.approx(5.0, abs=0.5)


def test_move_to_accepts_an_orientation_matrix():
    with SO101.connect(mock=True) as arm:
        anchor = arm.get_pose()
        frame = arm.gripper_frame_pose()
        turned = nod_gripper(frame[:3, :3], arm.nod_axis(), -5.0)
        arm.move_to(anchor.x, anchor.y, anchor.z, rotation=turned,
                    hold_point=True, wait=False)
        moved = arm.get_pose()
        assert np.linalg.norm(
            [moved.x - anchor.x, moved.y - anchor.y, moved.z - anchor.z]
        ) <= TURN_TOLERANCE_M
        degree = np.rad2deg(Rotation.from_matrix(
            frame[:3, :3].T @ arm.gripper_frame_pose()[:3, :3]).magnitude())
        assert degree == pytest.approx(5.0, abs=0.5)


def test_the_pose_quaternion_is_one_value_and_matches_fk():
    with SO101.connect(mock=True) as arm:
        frame = arm.gripper_frame_pose()
        pose = arm.get_pose()
        quaternion = pose.quaternion
        assert quaternion.shape == (4,)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0)
        first = next(value for value in quaternion if abs(value) > 1e-12)
        assert first > 0.0
        w, x, y, z = quaternion
        rebuilt = Rotation.from_quat([x, y, z, w]).as_matrix()
        assert np.rad2deg(
            Rotation.from_matrix(frame[:3, :3].T @ rebuilt).magnitude()) < 1e-6


def test_spin_wrist_turns_the_gripper_and_pitch_passes_90():
    with SO101.connect(mock=True) as arm:
        assert arm.spin_wrist(5.0) is True
        assert arm.get_joints()["wrist_roll"] == pytest.approx(5.0)
        assert arm.spin_wrist(-5.0) is True
        assert arm.get_joints()["wrist_roll"] == pytest.approx(0.0, abs=1e-6)
        arm._backend.positions["wrist_flex.pos"] = 113.0
        assert arm.get_pose().pitch == pytest.approx(113.0, abs=0.5)


def test_the_pose_reports_the_tool_pitch_the_keys_drive():
    with SO101.connect(mock=True) as arm:
        pose = arm.get_pose()
        frame = arm.gripper_frame_pose()
        assert pose.tool_pitch == pytest.approx(tool_pitch(frame[:3, :3]))
    # At Euler pitch 90 the RPY triple folds; the finger elevation does not.
    leaned = gripper_rotation_from_rpy((10.0, 90.0, 20.0))
    assert tool_pitch(leaned) == pytest.approx(-90.0, abs=1e-6)


def test_target_rotation_matches_the_reported_target_pose():
    with SO101.connect(mock=True) as arm:
        pose = arm.get_pose()
        arm.move_to(pose.x + 0.01, pose.y, pose.z, wait=False)
        rotation = arm.target_rotation()
        assert rotation.shape == (3, 3)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        target = arm.target_pose()
        assert (target.roll, target.pitch, target.yaw) == pytest.approx(
            tool_rpy(rotation), abs=1e-6)


def test_move_to_changes_pitch_in_the_base_frame():
    with SO101.connect(mock=True) as arm:
        pose = arm.get_pose()
        start = Rotation.from_euler("xyz", [pose.roll, pose.pitch, pose.yaw], degrees=True)
        goal = Rotation.from_euler("xyz", [pose.roll, pose.pitch + 5.0, pose.yaw], degrees=True)
        arm.move_to(pose.x, pose.y, pose.z, pose.roll, pose.pitch + 5.0, pose.yaw, wait=False)
        reached = arm.get_pose()
        got = Rotation.from_euler("xyz", [reached.roll, reached.pitch, reached.yaw], degrees=True)
        assert np.rad2deg((goal.inv() * got).magnitude()) < 1.0
        assert np.rad2deg((start.inv() * got).magnitude()) == pytest.approx(5.0, abs=1.0)
        assert np.linalg.norm(
            [reached.x - pose.x, reached.y - pose.y, reached.z - pose.z]) < 0.015


def test_connect_can_override_the_tool_offset():
    with SO101.connect(mock=True, ee_tool_offset=0.0) as arm:
        assert arm.ee_tool_offset == 0.0
        assert arm._kin.ee_tool_offset == 0.0


def test_move_grip_and_home():
    with SO101.connect(mock=True) as arm:
        arm.set_speed(30)
        arm.move_joint("shoulder_pan", 1.0)
        assert arm.get_joints()["shoulder_pan"] == pytest.approx(1.0)
        with pytest.raises(ValueError):
            arm.move_joint("gripper", 1.0)
        home = arm.get_pose()
        arm.move_joint("elbow_flex", -2.0)
        target = arm.get_pose()
        arm.home()
        assert arm.get_joints()["elbow_flex"] == pytest.approx(0.0, abs=0.05)
        arm.set_speed(90)
        assert arm.rest() is True
        rested = arm.get_joints()
        assert rested["shoulder_pan"] == pytest.approx(0.0, abs=0.05)
        assert rested["shoulder_lift"] == pytest.approx(-90.0, abs=0.05)
        assert rested["elbow_flex"] == pytest.approx(90.0, abs=0.05)
        assert rested["wrist_flex"] == pytest.approx(45.0, abs=0.05)
        assert rested["wrist_roll"] == pytest.approx(0.0, abs=0.05)
        assert rested["gripper"] == pytest.approx(0.0)
        arm.move_to(target.x, target.y, target.z)
        reached = arm.get_pose()
        assert np.linalg.norm([reached.x - target.x, reached.y - target.y, reached.z - target.z]) < 0.015
        arm.release()
        assert arm.get_joints()["gripper"] == pytest.approx(OPEN_PERCENT)
        arm.grip()
        assert arm.get_joints()["gripper"] == pytest.approx(0.0)
        arm.move_to(home.x, home.y, home.z, wait=False)
        arm.wait()


def test_tracker_sees_each_motion_step():
    with SO101.connect(mock=True) as arm:
        seen = []
        arm.track(lambda: seen.append(round(arm.get_joints()["shoulder_pan"], 3)))
        arm.move_joint("shoulder_pan", 1.0)
    assert seen[0] == 0.0
    assert seen[-1] == pytest.approx(1.0)
    assert len(seen) > 2


def test_offsets_are_added_on_read_and_removed_on_command():
    raw = MockBackend()
    offsets = {name: 0.0 for name in JOINTS}
    offsets["shoulder_lift"] = -4.0
    arm = OffsetBackend(raw, offsets)
    arm.send_action({"shoulder_lift.pos": 0.0})
    assert raw.positions["shoulder_lift.pos"] == pytest.approx(4.0)
    assert arm.get_observation()["shoulder_lift.pos"] == pytest.approx(0.0)
