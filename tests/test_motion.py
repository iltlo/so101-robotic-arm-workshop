"""A newer goal must not wait out a settle that the arm never reaches."""

import time

from so101_control.backend import MockBackend
from so101_control.kinematics import JOINTS
from so101_control.motion import move_joints

LIMITS = {name: (-180.0, 180.0) for name in JOINTS}


class _Held:
    """Accepts goals but never moves. This is the live settle hang."""

    def __init__(self):
        self.positions = {f"{name}.pos": 0.0 for name in (*JOINTS, "gripper")}
        self.sent = 0

    def get_observation(self):
        return dict(self.positions)

    def send_action(self, action):
        self.sent += 1
        return dict(action)


def test_cancel_stops_between_steps():
    robot = MockBackend()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    done = move_joints(
        robot, {"shoulder_pan": 30.0}, LIMITS,
        speed_deg_s=10, period_s=0.01, cancel=cancel,
    )
    assert done is None
    assert 0.0 < robot.positions["shoulder_pan.pos"] < 30.0


def test_cancel_returns_while_feedback_never_settles():
    robot = _Held()
    started = time.monotonic()

    def cancel():
        # 5 deg at 20 deg/s and 0.02 s is 13 steps. Cancel only once those
        # goals are sent, so this is the settle wait and not the trajectory.
        return robot.sent >= 13

    done = move_joints(
        robot, {"elbow_flex": 5.0}, LIMITS,
        speed_deg_s=20, period_s=0.02, timeout_s=5.0, cancel=cancel,
    )
    elapsed = time.monotonic() - started
    assert done is None
    assert robot.sent == 13
    assert elapsed < 1.5
    assert robot.positions["elbow_flex.pos"] == 0.0
