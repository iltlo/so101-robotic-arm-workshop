"""In-memory SO101: connect, move, grip, and the context manager. No serial port."""

import numpy as np
import pytest

from so101_control import SO101
from so101_control.backend import MockBackend, OffsetBackend
from so101_control.kinematics import JOINTS


def test_context_manager_and_speed_clamp():
    with SO101.connect(mock=True) as arm:
        assert arm.mock
        pose = arm.get_pose()
        assert np.isfinite([pose.x, pose.y, pose.z]).all()
        arm.set_speed(10)
        with pytest.raises(ValueError):
            arm.set_speed(200)
    assert arm._backend.connected is False


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
        assert arm.get_joints()["gripper"] == pytest.approx(100.0)
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
