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
    def __init__(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw


class _Arm:
    """Measured pose never moves, which is what a live settle wait hangs on."""

    def __init__(self):
        self.pose = _Pose(0.2, 0.0, 0.2)
        self.target = self.pose
        self.calls = []
        self.homes = []
        self.rests = []

    def get_pose(self):
        return self.pose

    def pose_at(self, joints):
        return self.pose

    def get_joints(self):
        return {"wrist_roll": 0.0, "wrist_flex": 0.0}

    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None, wait=True,
                single_goal=False, hold_point=False, cancel=None):
        self.calls.append((x, y, z, roll, pitch, yaw, wait))
        self.single_goal = single_goal
        self.hold_point = hold_point
        self.target = _Pose(x, y, z, *(value or 0.0 for value in (roll, pitch, yaw)))
        return True

    def target_pose(self):
        return self.target

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
    assert arm.calls[0][0] == pytest.approx(0.2 + teleop.step)
    assert arm.calls[0][1] == pytest.approx(0.0)
    assert arm.calls[0][3] is None
    assert arm.calls[1][0] == pytest.approx(0.2 + teleop.step)
    assert arm.calls[1][1] == pytest.approx(teleop.step)
    assert arm.pose.x == 0.2


def test_held_key_steps_again_from_the_command():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("w"))
    teleop.poll()
    teleop.poll()
    assert [call[0] for call in arm.calls] == pytest.approx(
        [0.2 + teleop.step, 0.2 + 2 * teleop.step])


def test_held_key_is_not_paused_while_the_arm_catches_up():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("w"))
    for _ in range(8):
        teleop.poll()
    assert len(arm.calls) == 8
    assert arm.calls[-1][0] == pytest.approx(0.2 + 8 * teleop.step)
    assert arm.calls[-1][-1] is False


def test_f_and_v_step_wrist_pitch():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("f"))
    teleop.on_release(key("f"))
    teleop.poll()
    teleop.on_press(key("v"))
    teleop.on_release(key("v"))
    teleop.poll()
    assert arm.calls[0][4] == pytest.approx(teleop.pitch_step)
    assert arm.calls[1][4] == pytest.approx(0.0)
    assert arm.calls[0][3] == pytest.approx(0.0)
    assert arm.calls[0][5] == pytest.approx(0.0)
    assert arm.calls[0][0] == pytest.approx(0.2)
    assert arm.hold_point is True


def test_home_does_not_wait_for_settle():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("h"))
    teleop.on_release(module.keyboard.KeyCode.from_char("h"))
    teleop.poll()
    assert arm.homes == [(False, False)]


def test_the_viewer_keeps_reading_joints_while_no_key_is_held():
    module = _load()
    arm = _Arm()
    arm.shoulder = 0.0
    teleop = module.Teleop(arm)
    draws = []

    def refresh():
        draws.append(arm.shoulder)
        arm.shoulder += 15.0

    steps = {"n": 0}

    def alive():
        steps["n"] += 1
        return steps["n"] <= 2

    module.run(teleop, alive, refresh)
    assert draws == [0.0, 15.0]


def test_r_goes_to_rest_without_waiting_for_settle():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("r"))
    teleop.on_release(module.keyboard.KeyCode.from_char("r"))
    teleop.poll()
    assert arm.rests == [(False, False)]
    assert teleop.closed is True
