"""One click picks at PICK_HEIGHT_M, then releases only after the arm is at rest."""

import importlib.util
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

from so101_control import SO101
from so101_control.arm import OPEN_PERCENT, REST_JOINTS
from so101_control.kinematics import JOINTS
from so101_utils.click_view import click_view

_SCRIPT = Path(__file__).resolve().parents[1] / "application" / "click_and_pick.py"


def _load():
    spec = importlib.util.spec_from_file_location("click_and_pick", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _View:
    def __init__(self):
        self.messages = []
        self.pending = False

    def report(self, message):
        self.messages.append(message)

    def has_pending(self):
        return self.pending


class _Frame:
    def __init__(self, image):
        self.rgb_bgr = image


class _Cam:
    def __init__(self, image):
        self.image = image
        self.polls = 0

    def read(self):
        return _Frame(self.image)

    def poll(self):
        self.polls += 1
        return _Frame(self.image)


def test_camera_frame_is_up_before_home_and_keeps_updating():
    module = _load()
    image = np.zeros((8, 8, 3), np.uint8)
    shown = []
    with SO101.connect(mock=True) as arm, click_view(8, 8) as (view, url):
        module.attach_camera(arm, _Cam(image), view, shown.append)
        assert urlopen(url + "frame").status == 200
        assert _get_status(url) == "Waiting for a click"
        before = len(shown)
        arm.home()
        assert len(shown) > before


def _get_status(url):
    import json
    with urlopen(url + "status") as response:
        return json.load(response)["message"]


def test_page_states_the_pick_height():
    module = _load()
    assert f"{module.PICK_HEIGHT_M * 1000:.0f} mm" in module._page_text()
    assert f"{module.APPROACH_M * 1000:.0f} mm" in module._page_text()


def test_a_click_during_rest_waits_until_rest_is_sent():
    module = _load()
    image = np.zeros((8, 8, 3), np.uint8)
    with SO101.connect(mock=True) as arm, click_view(8, 8) as (view, url):
        arm.move_joint("elbow_flex", -2.0)
        view.publish(image)
        request = Request(
            url + "click", data=json.dumps({"x": 1, "y": 1}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request):
            pass
        module.ready(arm, view)
        joints = arm.get_joints()
        for name, angle in REST_JOINTS.items():
            assert joints[name] == pytest.approx(angle, abs=0.05)
        assert joints["gripper"] == pytest.approx(OPEN_PERCENT)
        assert view.consume() == (1, 1)


def test_ready_folds_to_rest():
    module = _load()
    assert module.EE_TOOL_OFFSET_M == 0.025
    with SO101.connect(mock=True) as arm:
        arm.move_joint("elbow_flex", -2.0)
        arm.release()
        module.ready(arm)
        joints = arm.get_joints()
        assert joints["gripper"] == pytest.approx(OPEN_PERCENT)
        for name, angle in REST_JOINTS.items():
            assert joints[name] == pytest.approx(angle, abs=0.05)


def test_pick_stops_above_then_comes_straight_down():
    module = _load()
    with SO101.connect(mock=True) as arm:
        module.ready(arm)
        pick = (0.28, -0.05, 0.03)
        seen = []
        calls = []
        grip = arm.grip
        release = arm.release
        move_to = arm.move_to

        def moved(x, y, z, roll=None, pitch=None, yaw=None, wait=True, cancel=None,
                  downward=False):
            calls.append((x, y, z, roll, downward))
            return move_to(
                x, y, z, roll, pitch, yaw, wait=wait, cancel=cancel, downward=downward)

        def closed():
            pose = arm.get_pose()
            fingers = arm._kin.fk(arm.get_joints())[:3, 0]
            seen.append(("grip", pose.x, pose.y, pose.z, fingers[2]))
            grip()

        def opened():
            joints = arm.get_joints()
            seen.append(("release", joints["gripper"], {name: joints[name] for name in JOINTS}))
            release()

        arm.move_to = moved
        arm.grip = closed
        arm.release = opened
        view = _View()
        module.DWELL_S = 0.0
        module.pick_and_return(arm, view, pick)
        hover_z = pick[2] + module.APPROACH_M
        assert calls[0][:3] == pytest.approx((pick[0], pick[1], hover_z))
        assert calls[0][3] is None and calls[0][4] is True
        assert calls[1][:3] == pytest.approx(pick)
        assert calls[1][3] is None and calls[1][4] is True
        assert view.messages.index("Holding the pick") < view.messages.index("Closing gripper")
        assert view.messages.index("Holding the grasp") < view.messages.index("Returning to rest")
        assert seen[0][0] == "grip"
        assert np.linalg.norm([seen[0][1] - pick[0], seen[0][2] - pick[1], seen[0][3] - pick[2]]) < 0.015
        assert seen[0][4] < -0.95
        opened = seen[-1]
        assert opened[0] == "release"
        assert opened[1] == pytest.approx(0.0)
        for name, angle in REST_JOINTS.items():
            assert opened[2][name] == pytest.approx(angle, abs=0.05)
        assert arm.get_joints()["gripper"] == pytest.approx(OPEN_PERCENT)
        assert view.messages[-1] == "Released at rest"


def test_hold_keeps_the_camera_updating():
    import time

    module = _load()
    refreshes = []

    class _Arm:
        def _on_motion(self):
            refreshes.append(time.monotonic())

    started = time.monotonic()
    module._hold(_Arm(), 0.12)
    elapsed = time.monotonic() - started
    assert elapsed == pytest.approx(0.12, abs=0.08)
    assert len(refreshes) >= 2


def test_a_missed_pick_does_not_close_or_go_to_rest():
    module = _load()

    class _Arm:
        def move_to(self, x, y, z, roll=None, pitch=None, yaw=None, wait=True, cancel=None,
                    downward=False):
            return False

        def grip(self):
            raise AssertionError("gripper closed")

        def rest(self, wait=True, cancel=None):
            raise AssertionError("went to rest")

        def release(self):
            raise AssertionError("gripper opened")

    view = _View()
    module.pick_and_return(_Arm(), view, (0.2, 0.0, 0.02))
    assert view.messages[-1] == "Did not reach the pick"
