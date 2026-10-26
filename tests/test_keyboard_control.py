"""WASD must keep taking steps when the arm has not settled on the previous one."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "keyboard_control.py"


def _load():
    spec = importlib.util.spec_from_file_location("keyboard_control", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pose:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Arm:
    """Measured pose never moves, which is what a live settle wait hangs on."""

    def __init__(self):
        self.pose = _Pose(0.2, 0.0, 0.2)
        self.calls = []
        self.homes = []
        self.rests = []

    def get_pose(self):
        return self.pose

    def get_joints(self):
        return {"wrist_roll": 0.0}

    def move_to(self, x, y, z, r=None, wait=True):
        self.calls.append((x, y, z, r, wait))

    def home(self, *, wait=True, cancel=None):
        self.homes.append((wait, bool(cancel and cancel())))
        return True

    def rest(self, *, wait=True, cancel=None):
        self.rests.append((wait, bool(cancel and cancel())))
        return True


def test_wasd_presses_apply_without_waiting_for_settle():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("w"))
    teleop.on_release(key("w"))
    teleop.on_press(key("a"))
    teleop.on_release(key("a"))
    assert arm.calls == []
    teleop.poll()
    teleop.poll()
    assert [call[4] for call in arm.calls] == [False, False]
    assert arm.calls[0][0] == pytest.approx(0.21)
    assert arm.calls[0][1] == pytest.approx(0.0)
    assert arm.calls[1][0] == pytest.approx(0.21)
    assert arm.calls[1][1] == pytest.approx(0.01)
    assert arm.pose.x == 0.2


def test_held_key_steps_again_from_the_command():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("w"))
    teleop.poll()
    teleop.poll()
    assert [call[0] for call in arm.calls] == pytest.approx([0.21, 0.22])
    assert [call[4] for call in arm.calls] == [False, False]


def test_held_key_does_not_walk_the_goal_away_from_a_stuck_arm():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("w"))
    for _ in range(6):
        teleop.poll()
    assert len(arm.calls) == 5
    assert arm.calls[-1][0] == pytest.approx(0.25)
    assert arm.calls[-1][4] is False


def test_home_does_not_wait_for_settle():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("h"))
    teleop.on_release(module.keyboard.KeyCode.from_char("h"))
    teleop.poll()
    assert arm.homes == [(False, False)]


def test_r_goes_to_rest_without_waiting_for_settle():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("r"))
    teleop.on_release(module.keyboard.KeyCode.from_char("r"))
    teleop.poll()
    assert arm.rests == [(False, False)]
    assert teleop.closed is True
