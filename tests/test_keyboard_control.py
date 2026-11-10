"""WASD must keep taking steps when the arm has not settled on the previous one."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.kinematics import IKError
from so101_utils.model import (
    gripper_rotation_from_rpy, nod_gripper, pitch_about_axis, pitch_axis, tool_rpy,
)

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
        self.spins = []
        self._target_rotation = np.eye(3)

    def get_pose(self):
        return self.pose

    def pose_at(self, joints):
        return self.pose

    def get_joints(self):
        return {"wrist_roll": 0.0, "wrist_flex": 0.0}

    def gripper_frame_pose(self, joints=None):
        matrix = np.eye(4)
        matrix[:3, :3] = gripper_rotation_from_rpy(
            (self.pose.roll, self.pose.pitch, self.pose.yaw))
        return matrix

    def nod_axis(self, joints=None):
        return np.asarray(getattr(self, "_axis", (0.0, 1.0, 0.0)), dtype=float)

    def spin_wrist(self, delta_deg, *, cancel=None):
        self.spins.append(float(delta_deg))
        self.wrist_roll = getattr(self, "wrist_roll", 0.0) + float(delta_deg)
        self._target_rotation = self._target_rotation @ Rotation.from_euler(
            "x", float(delta_deg), degrees=True).as_matrix()
        return True

    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None, wait=True,
                single_goal=False, hold_point=False, cancel=None, rotation=None):
        if rotation is not None:
            roll, pitch, yaw = tool_rpy(rotation)
        self.calls.append((x, y, z, roll, pitch, yaw, wait))
        self.single_goal = single_goal
        self.hold_point = hold_point
        if rotation is not None:
            self._target_rotation = np.asarray(rotation, dtype=float)
        elif roll is not None:
            self._target_rotation = gripper_rotation_from_rpy((roll, pitch, yaw))
        self.target = _Pose(x, y, z, *(value or 0.0 for value in (roll, pitch, yaw)))
        return True

    def target_pose(self):
        return self.target

    def target_rotation(self):
        return self._target_rotation

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
    assert arm.calls == []
    teleop.poll()
    teleop.on_press(key("a"))
    teleop.on_release(key("a"))
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


def test_f_and_v_increase_and_decrease_pitch():
    module = _load()
    arm = _Arm()
    start = pitch_about_axis(arm.gripper_frame_pose()[:3, :3], arm.nod_axis())
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("f"))
    teleop.on_release(key("f"))
    teleop.poll()
    raised = pitch_about_axis(arm._target_rotation, arm.nod_axis())
    teleop.on_press(key("v"))
    teleop.on_release(key("v"))
    teleop.poll()
    lowered = pitch_about_axis(arm._target_rotation, arm.nod_axis())
    assert raised - start == pytest.approx(teleop.pitch_step, abs=0.5)
    assert lowered - raised == pytest.approx(-teleop.pitch_step, abs=0.5)
    assert arm.calls[0][0] == pytest.approx(0.2)
    assert arm.hold_point is True


@pytest.mark.parametrize("yaw", [0.0, 180.0])
def test_f_increases_pitch_whichever_way_the_arm_faces(yaw):
    """Across the base the flexion axis flips with the arm; pitch + must not."""
    module = _load()
    arm = _Arm()
    arm.pose = _Pose(0.2, 0.0, 0.2, 0.0, 20.0, yaw)
    arm._axis = (0.0, -1.0, 0.0) if yaw == 180.0 else (0.0, 1.0, 0.0)
    start = pitch_about_axis(arm.gripper_frame_pose()[:3, :3], arm.nod_axis())
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("f"))
    teleop.on_release(module.keyboard.KeyCode.from_char("f"))
    teleop.poll()
    assert pitch_about_axis(arm._target_rotation, arm.nod_axis()) == pytest.approx(start + 5.0, abs=0.5)


class _TightArm(_Arm):
    """Only a quarter of a pitch press can hold the point; bigger steps are refused."""

    def __init__(self):
        super().__init__()
        self.attempts = []

    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None, **kwargs):
        rotation = kwargs.get("rotation")
        if rotation is not None:
            pitch = float(tool_rpy(rotation)[1])
        self.attempts.append(abs(pitch or 0.0))
        if rotation is not None and abs(pitch) > 1.25:
            raise IKError("Cannot hold the tool point through that turn")
        return super().move_to(x, y, z, roll, pitch, yaw, **kwargs)


def test_a_blocked_turn_retries_at_a_quarter_of_a_press():
    module = _load()
    arm = _TightArm()
    teleop = module.Teleop(arm)
    teleop.on_press(module.keyboard.KeyCode.from_char("v"))
    teleop.on_release(module.keyboard.KeyCode.from_char("v"))
    teleop.poll()
    assert len(arm.attempts) == 3
    start = pitch_about_axis(arm.gripper_frame_pose()[:3, :3], arm.nod_axis())
    assert pitch_about_axis(arm._target_rotation, arm.nod_axis()) - start == pytest.approx(-1.25, abs=0.05)


class _StuckArm(_Arm):
    def move_to(self, x, y, z, roll=None, pitch=None, yaw=None, **kwargs):
        if kwargs.get("rotation") is not None:
            raise IKError("Cannot hold the tool point through that turn")
        return super().move_to(x, y, z, roll, pitch, yaw, **kwargs)


def test_a_blocked_turn_is_reported_once():
    module = _load()
    teleop = module.Teleop(_StuckArm())
    teleop.on_press(module.keyboard.KeyCode.from_char("v"))
    teleop.on_release(module.keyboard.KeyCode.from_char("v"))
    teleop.poll()
    assert teleop.notice == "Turn blocked: the wrist cannot turn that way here"
    assert teleop.notice is None


def test_a_press_steps_from_the_solved_frame_not_the_reported_angles():
    """At pitch 90 the reported roll/yaw pair is one of many; the step needs the frame."""
    module = _load()
    arm = _Arm()
    lean = gripper_rotation_from_rpy((10.0, 85.0, 20.0))
    arm._target_rotation = lean
    arm.pose = _Pose(0.2, 0.0, 0.2, 0.0, 90.0, -15.0)
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("w"))
    teleop.on_release(key("w"))
    teleop.poll()
    teleop.on_press(key("v"))
    teleop.on_release(key("v"))
    teleop.poll()
    asked = arm._target_rotation
    axis = pitch_axis(lean, arm.nod_axis())
    expected = nod_gripper(lean, axis, -5.0)
    angle = Rotation.from_matrix(expected.T @ asked).magnitude()
    assert np.rad2deg(angle) < 1e-6


def test_z_and_c_spin_the_gripper_when_a_pose_solve_cannot():
    """Roll is the wrist joint, so a blocked hold-point solve cannot swallow it."""
    module = _load()
    arm = _StuckArm()
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("z"))
    teleop.on_release(key("z"))
    teleop.poll()
    teleop.on_press(key("c"))
    teleop.on_release(key("c"))
    teleop.poll()
    assert arm.spins == pytest.approx([teleop.roll_step, -teleop.roll_step])
    assert arm.calls == []


def test_a_new_key_replaces_one_that_has_not_run():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("f"))
    teleop.on_release(key("f"))
    teleop.on_press(key("w"))
    teleop.on_release(key("w"))
    teleop.poll()
    teleop.poll()
    assert len(arm.calls) == 1
    assert arm.calls[0][0] == pytest.approx(0.2 + teleop.step)
    assert arm.calls[0][1] == pytest.approx(0.0)


def test_only_the_latest_home_or_rest_runs():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    key = module.keyboard.KeyCode.from_char
    teleop.on_press(key("h"))
    teleop.on_release(key("h"))
    teleop.on_press(key("r"))
    teleop.on_release(key("r"))
    teleop.on_press(key("h"))
    teleop.on_release(key("h"))
    teleop.poll()
    teleop.poll()
    assert arm.homes == [(False, False)]
    assert arm.rests == []


def test_rest_cuts_off_a_home_that_is_already_moving():
    module = _load()
    arm = _Arm()
    teleop = module.Teleop(arm)
    seen = []

    def home(*, wait=True, cancel=None):
        teleop.press_token("r")
        seen.append(bool(cancel and cancel()))
        return not seen[-1]

    arm.home = home
    teleop.press_token("h")
    teleop.release_token("h")
    teleop.poll()
    assert seen == [True]
    assert arm.rests == []
    teleop.poll()
    assert arm.rests == [(False, False)]


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
