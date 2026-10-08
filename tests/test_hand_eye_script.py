"""Guided torque and capture tests never touch the physical follower."""

import importlib.util
from pathlib import Path
from collections import deque

import numpy as np
import pytest


def load_script():
    path = Path(__file__).parents[1] / "calibration" / "hand_eye.py"
    spec = importlib.util.spec_from_file_location("workshop_hand_eye", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBus:
    def __init__(self):
        self.calls = []

    def disable_torque(self):
        self.calls.append("off")

    def sync_read(self, register):
        assert register == "Present_Position"
        self.calls.append("read")
        return {"shoulder_pan": 12.5, "gripper": 70.0}

    def sync_write(self, register, positions):
        assert register == "Goal_Position"
        self.calls.append(("goal", positions.copy()))

    def enable_torque(self):
        self.calls.append("on")


def test_unlock_and_lock_hold_measured_position_before_torque_on():
    from so101_control.torque import ArmTorque

    robot = type("Robot", (), {"bus": FakeBus()})()
    controller = ArmTorque(robot)
    assert controller.toggle() is False
    assert controller.toggle() is True
    assert robot.bus.calls == ["off", "read", ("goal", {"shoulder_pan": 12.5, "gripper": 70.0}), "on"]


def test_capture_rejected_unlocked_no_limits_needed(monkeypatch):
    script = load_script()
    robot = type("Robot", (), {"bus": FakeBus()})()
    commands = deque(("t", "enter", "q"))
    samples = []
    state = {name: 0.0 for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    monkeypatch.setattr(script, "joint_state", lambda _: state.copy())
    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    monkeypatch.setattr(script, "solve_eye_to_hand_known_tag",
                        lambda samples, mount: pytest.fail("must not solve while unlocked"))

    class Kin:
        def fk(self, state):
            return np.eye(4)

        def fk_gripper_link(self, state):
            return np.eye(4)

    class View:
        def update(self, *args):
            samples.append("preview")
            return np.eye(4)

        def show_camera_estimate(self, *args, **kwargs):
            pytest.fail("capture should not happen with torque off")

    cam = type("Cam", (), {"read": lambda self: type("Frame", (), {"rgb_bgr": None})()})()
    keys = type("Keys", (), {"consume": lambda self: commands.popleft()})()
    script.run_capture(robot, cam, Kin(), {}, "config/station.yaml", view=View(), keys=keys)
    assert robot.bus.calls == ["off"]
    assert len(samples) == 3  # preview continues even while torque is disabled


def test_t_key_unlocked_then_locked_before_capture(monkeypatch):
    script = load_script()
    robot = type("Robot", (), {"bus": FakeBus()})()
    commands = deque(("t", "t", "q"))
    state = {name: 0.0 for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    monkeypatch.setattr(script, "joint_state", lambda _: state.copy())

    class Kin:
        def fk(self, state):
            return np.eye(4)

        def fk_gripper_link(self, state):
            return np.eye(4)

    class View:
        def update(self, *args):
            return np.eye(4)

    cam = type("Cam", (), {"read": lambda self: object()})()
    keys = type("Keys", (), {"consume": lambda self: commands.popleft()})()
    script.run_capture(robot, cam, Kin(), {}, "config/station.yaml", view=View(), keys=keys)
    assert robot.bus.calls == ["off", "read", ("goal", {"shoulder_pan": 12.5, "gripper": 70.0}), "on"]


def test_calibration_requires_explicit_live_and_fk_validation(monkeypatch):
    script = load_script()
    with pytest.raises(SystemExit):
        script.main([])
    monkeypatch.setattr(script, "load_station", lambda _: {"fk_tcp_verified": False})
    with pytest.raises(ValueError, match="gripper_link FK"):
        script.main(["--enable-live"])


def test_calibration_uses_no_stdin():
    script = load_script()
    assert not hasattr(script, "_read_command")


def test_enter_shows_each_provisional_pose_and_c_saves_validated(monkeypatch):
    script = load_script()
    robot = type("Robot", (), {"bus": FakeBus()})()
    commands = deque(("enter", "enter", "c", "q"))
    state = {name: 0.0 for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    monkeypatch.setattr(script, "joint_state", lambda _: state.copy())
    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    solved = []
    saved = []
    camera = np.eye(4)
    camera[:3, 3] = [0.2, 0, 0.3]

    def solve(samples, mount):
        solved.append(len(samples))
        return {"base_T_camera": camera, "gripper_T_tag": mount,
                "max_residual_m": 0.0, "max_residual_deg": 0.0}

    monkeypatch.setattr(script, "solve_eye_to_hand_known_tag", solve)
    monkeypatch.setattr(script, "save_calibration", lambda *args: saved.append(args))
    monkeypatch.setattr(script, "intrinsics_data", lambda _: {})

    class Kin:
        def fk(self, state):
            return np.eye(4)

        def fk_gripper_link(self, state):
            return np.eye(4)

    class View:
        def __init__(self):
            self.estimates = []

        def update(self, *args):
            return np.eye(4)

        def show_camera_estimate(self, pose, *, sample_count, confirmed=False):
            self.estimates.append((sample_count, confirmed, pose.copy()))

    view = View()
    cam = type("Cam", (), {"intrinsics": object(), "read": lambda self: object()})()
    keys = type("Keys", (), {"consume": lambda self: commands.popleft()})()
    script.run_capture(robot, cam, Kin(), {"urdf": "robot.urdf"}, "station.yaml", view=view, keys=keys)
    assert [(count, confirmed) for count, confirmed, _ in view.estimates] == [
        (1, False), (2, False), (2, True)]
    assert solved == [2] and len(saved) == 1
    assert saved[0][1]["accepted"] is True


def test_c_rejects_insufficient_samples_without_saving(monkeypatch):
    script = load_script()
    robot = type("Robot", (), {"bus": FakeBus()})()
    commands = deque(("c", "q"))
    state = {name: 0.0 for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    monkeypatch.setattr(script, "joint_state", lambda _: state.copy())
    monkeypatch.setattr(script, "save_calibration", lambda *args: pytest.fail("must not save"))

    class Kin:
        def fk(self, state):
            return np.eye(4)

        fk_gripper_link = fk

    class View:
        def update(self, *args):
            return None

    keys = type("Keys", (), {"consume": lambda self: commands.popleft()})()
    cam = type("Cam", (), {"read": lambda self: object()})()
    script.run_capture(robot, cam, Kin(), {}, "station.yaml", view=View(), keys=keys)


def test_q_after_capture_discards_without_saving_and_explains_c(monkeypatch, capsys):
    script = load_script()
    robot = type("Robot", (), {"bus": FakeBus()})()
    commands = deque(("enter", "q"))
    state = {name: 0.0 for name in
             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    monkeypatch.setattr(script, "joint_state", lambda _: state.copy())
    monkeypatch.setattr(script.time, "sleep", lambda _: None)
    monkeypatch.setattr(script, "save_calibration", lambda *args: pytest.fail("Q must not save"))

    class Kin:
        def fk(self, state):
            return np.eye(4)

        fk_gripper_link = fk

    class View:
        def update(self, *args):
            return np.eye(4)

        def show_camera_estimate(self, *args, **kwargs):
            pass

    keys = type("Keys", (), {"consume": lambda self: commands.popleft()})()
    cam = type("Cam", (), {"read": lambda self: object()})()
    script.run_capture(robot, cam, Kin(), {}, "station.yaml", view=View(), keys=keys)
    assert "1 unsaved samples discarded" in capsys.readouterr().out


def test_contour_detection_returns_original_rgb_corners(monkeypatch):
    from so101_control import calibration as cal
    from so101_control.geometry import Intrinsics

    corners = np.array([[10.0, 30.0], [30.0, 30.0], [30.0, 10.0], [10.0, 10.0]])

    class Detector:
        def detect(self, image):
            return [type("Tag", (), {"tag_id": 0, "corners": corners})()]

    intr = Intrinsics(40, 40, 100, 100, 20, 20)
    pose, detected = cal.find_tag(np.zeros((40, 40, 3), dtype=np.uint8), intr, detector=Detector())
    assert pose.shape == (4, 4)
    assert np.allclose(detected, corners)


def test_camera_candidate_after_one_sample_needs_no_full_solve():
    from so101_control.calibration import Sample, measured_gripper_T_tag, camera_candidates

    mount = measured_gripper_T_tag()
    camera = np.eye(4)
    camera[:3, 3] = [0.2, -0.1, 0.4]
    gripper = np.eye(4)
    gripper[:3, 3] = [0.12, 0, 0.2]
    tag = np.linalg.inv(camera) @ gripper @ mount
    candidates = camera_candidates([Sample(gripper, tag)], mount)
    assert len(candidates) == 1
    assert np.allclose(candidates[0], camera)


def test_calibration_controls_expose_t_enter_c_without_prompt(monkeypatch):
    from so101_control.torque import keyboard_requests

    assert callable(keyboard_requests)
    script = load_script()
    assert not hasattr(script, "_read_command")


def test_keyboard_controls_debounce_and_queue_without_motor_io(monkeypatch):
    from pynput import keyboard
    from so101_control.torque import keyboard_requests

    listeners = []

    class Listener:
        def __init__(self, on_press, on_release):
            self.press, self.release = on_press, on_release
            listeners.append(self)

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(keyboard, "Listener", Listener)
    with keyboard_requests() as requests:
        listener = listeners[0]
        for key in (keyboard.KeyCode.from_char("T"), keyboard.KeyCode.from_char("T"),
                    keyboard.Key.enter, keyboard.KeyCode.from_char("C"),
                    keyboard.KeyCode.from_char("Q")):
            listener.press(key)
        assert [requests.consume() for _ in range(5)] == ["t", "enter", "c", "q", None]
        listener.release(keyboard.KeyCode.from_char("t"))
        listener.press(keyboard.KeyCode.from_char("t"))
        assert requests.consume() == "t"