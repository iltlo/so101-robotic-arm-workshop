"""One click picks at PICK_HEIGHT_M, then releases only after the arm is at rest."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rerun._image_clicks import note_viewer_event, reset_image_clicks
from rerun.experimental import EntitySelectionItem, SelectionChangeEvent

from so101_control import SO101
from so101_control.arm import OPEN_PERCENT, REST_JOINTS
from so101_utils.click_app import attach_camera
from so101_utils.model import JOINTS
from so101_utils.views import CameraClicks

_SCRIPT = Path(__file__).resolve().parents[1] / "application" / "click_and_pick.py"


def _load():
    spec = importlib.util.spec_from_file_location("click_and_pick", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clicks:
    def __init__(self):
        self.pending = False

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


def _selection(x, y):
    return SelectionChangeEvent(
        application_id="so101_control",
        recording_id="recording",
        segment_id=None,
        items=[EntitySelectionItem(entity_path="camera/rgb", position=[x, y, 0.0])],
    )


def test_camera_frames_keep_updating_during_moves():
    image = np.zeros((8, 8, 3), np.uint8)
    shown = []
    with SO101.connect(mock=True) as arm:
        attach_camera(arm, _Cam(image), shown.append)
        before = len(shown)
        arm.home()
        assert len(shown) > before


def test_a_click_during_rest_waits_until_rest_is_sent():
    module = _load()
    with SO101.connect(mock=True) as arm:
        reset_image_clicks()
        clicks = CameraClicks(8, 8)
        clicks.mark_ready()
        arm.move_joint("elbow_flex", -2.0)
        note_viewer_event(_selection(1, 1))
        module.ready(arm)
        joints = arm.get_joints()
        for name, angle in REST_JOINTS.items():
            assert joints[name] == pytest.approx(angle, abs=0.05)
        assert joints["gripper"] == pytest.approx(OPEN_PERCENT)
        assert clicks.poll() == (1, 1)


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


def test_pick_stops_above_then_comes_straight_down(capsys):
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
            fingers = arm.gripper_frame_pose()[:3, 0]
            seen.append(("grip", pose.x, pose.y, pose.z, fingers[2]))
            grip()

        def opened():
            joints = arm.get_joints()
            seen.append(("release", joints["gripper"], {name: joints[name] for name in JOINTS}))
            release()

        arm.move_to = moved
        arm.grip = closed
        arm.release = opened
        module.DWELL_S = 0.0
        module.pick_and_return(arm, _Clicks(), pick)
        hover_z = pick[2] + module.APPROACH_M
        assert calls[0][:3] == pytest.approx((pick[0], pick[1], hover_z))
        assert calls[0][3] is None and calls[0][4] is True
        assert calls[1][:3] == pytest.approx(pick)
        assert calls[1][3] is None and calls[1][4] is True
        lines = capsys.readouterr().out.splitlines()
        assert lines.index("Holding the pick") < lines.index("Closing gripper")
        assert lines.index("Holding the grasp") < lines.index("Returning to rest")
        assert seen[0][0] == "grip"
        assert np.linalg.norm([seen[0][1] - pick[0], seen[0][2] - pick[1], seen[0][3] - pick[2]]) < 0.015
        assert seen[0][4] < -0.95
        opened = seen[-1]
        assert opened[0] == "release"
        assert opened[1] == pytest.approx(0.0)
        for name, angle in REST_JOINTS.items():
            assert opened[2][name] == pytest.approx(angle, abs=0.05)
        assert arm.get_joints()["gripper"] == pytest.approx(OPEN_PERCENT)
        assert lines[-1] == "Released at rest"


def test_hold_keeps_the_camera_updating():
    import time

    module = _load()
    refreshes = []

    class _Arm:
        def refresh(self):
            refreshes.append(time.monotonic())

    started = time.monotonic()
    module._hold(_Arm(), 0.12)
    elapsed = time.monotonic() - started
    assert elapsed == pytest.approx(0.12, abs=0.08)
    assert len(refreshes) >= 2


def test_a_missed_pick_does_not_close_or_go_to_rest(capsys):
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

    module.pick_and_return(_Arm(), _Clicks(), (0.2, 0.0, 0.02))
    assert capsys.readouterr().out.splitlines()[-1] == "Did not reach the pick"
