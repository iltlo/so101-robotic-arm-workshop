"""A stalled jaw must drop its goal, and quit must close the port anyway."""

import pytest

from so101_control.arm import SO101
from so101_control.backend import connect_follower, release_overload
from so101_control.kinematics import JOINTS


class _Stall:
    """Closes until 18% and then stops, short of a fully closed goal."""

    def __init__(self):
        self.goals = []
        self.gripper = 40.0
        self.goal = 40.0

    def send_action(self, action):
        self.goal = action["gripper.pos"]
        self.goals.append(self.goal)
        return action

    def get_observation(self):
        if self.goal <= 0.0:
            self.gripper = max(18.0, self.gripper - 8.0)
        else:
            self.gripper = self.goal
        obs = {f"{name}.pos": 0.0 for name in (*JOINTS, "gripper")}
        obs["gripper.pos"] = self.gripper
        return obs


class _Bus:
    def __init__(self):
        self.is_connected = True
        self.present = {"gripper": 18.0}
        self.goals = []
        self.closed_without_torque_off = False

    def sync_read(self, name):
        assert name == "Present_Position"
        return dict(self.present)

    def sync_write(self, name, values):
        assert name == "Goal_Position"
        self.goals.append(dict(values))

    def disconnect(self, disable_torque=True):
        self.is_connected = False
        self.closed_without_torque_off = disable_torque is False


class _Live:
    def __init__(self):
        self.bus = _Bus()

    def disconnect(self):
        raise RuntimeError(
            "Failed to write 'Torque_Enable' on id_=6 with '0' after 6 tries. "
            "[RxPacketError] Overload error!"
        )


def test_stalled_gripper_holds_where_it_stopped():
    arm = SO101(_Stall(), None, {}, mock=False)
    arm.grip()
    assert arm._backend.goals[0] == pytest.approx(0.0)
    assert arm._backend.goals[-1] == pytest.approx(18.0)
    assert arm.get_joints()["gripper"] == pytest.approx(18.0)


def test_release_overload_rewrites_the_goal_to_the_measured_position():
    bus = _Bus()
    release_overload(bus)
    assert bus.goals == [{"gripper": 18.0}]


def test_connect_clears_a_latched_overload_and_configures_again():
    class Follower:
        def __init__(self):
            self.bus = _Bus()
            self.configured = False

        def connect(self, calibrate=False):
            raise RuntimeError(
                "Failed to write 'Torque_Enable' on id_=6 with '0' after 6 tries. "
                "[RxPacketError] Overload error!"
            )

        def configure(self):
            self.configured = True

    follower = Follower()
    connect_follower(follower)
    assert follower.bus.goals == [{"gripper": 18.0}]
    assert follower.configured is True


def test_close_unloads_an_overloaded_gripper_and_still_closes_the_port(capsys):
    backend = _Live()
    arm = SO101(backend, None, {}, mock=False)
    arm.close()
    assert backend.bus.goals == [{"gripper": 18.0}]
    assert backend.bus.is_connected is False
    assert backend.bus.closed_without_torque_off is True
    assert "Overload error" in capsys.readouterr().out
