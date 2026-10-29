"""A new click must take over, and the page must show that without another click."""

import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from so101_control import SO101
from so101_control.arm import REST_JOINTS
from so101_utils.click_view import click_view

_SCRIPT = Path(__file__).resolve().parents[1] / "application" / "click_and_go.py"


def _load():
    spec = importlib.util.spec_from_file_location("click_and_go", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post(url, payload):
    data = json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.load(response)


def _get(url):
    with urlopen(url) as response:
        return json.load(response)


def test_page_stays_blank_until_a_frame_arrives_and_ignores_that_click():
    with click_view(8, 8) as (view, url):
        assert _get(url + "status")["message"] == "Waiting for camera..."
        page = urlopen(url).read().decode()
        assert "Camera image is not ready" in page
        try:
            urlopen(url + "frame")
            raise AssertionError("frame was served before the camera published")
        except HTTPError as exc:
            assert exc.code == 503
        try:
            _post(url + "click", {"x": 0, "y": 0})
            raise AssertionError("click was queued with no image")
        except HTTPError as exc:
            assert exc.code == 400
            assert json.load(exc)["message"] == "Camera image is not ready"
        assert view.consume() is None
        view.publish(np.zeros((8, 8, 3), np.uint8))
        assert _get(url + "status")["message"] == "Waiting for a click"
        assert urlopen(url + "frame").status == 200


def _show(view):
    view.publish(np.zeros((8, 8, 3), np.uint8))


def test_status_updates_after_the_click_returns():
    with click_view(640, 480) as (view, url):
        _show(view)
        view.report("Moved to 0.282, -0.082, 0.020 m")
        body = _post(url + "click", {"x": 323, "y": 85})
        assert body["message"] == "Queued (323, 85)"
        view.report("Moving to 0.264, 0.048, 0.020 m")
        assert _get(url + "status")["message"] == "Moving to 0.264, 0.048, 0.020 m"
        page = urlopen(url).read().decode()
        assert "/status" in page


def test_newer_click_replaces_one_still_waiting():
    with click_view(64, 48) as (view, url):
        _show(view)
        _post(url + "click", {"x": 1, "y": 2})
        _post(url + "click", {"x": 3, "y": 4})
        assert view.consume() == (3, 4)
        assert view.has_pending() is False
        assert view.consume() is None


def test_ready_folds_to_rest_and_a_click_waits():
    module = _load()
    assert module.EE_TOOL_OFFSET_M == 0.0
    with SO101.connect(mock=True, ee_tool_offset=module.EE_TOOL_OFFSET_M) as arm, click_view(8, 8) as (view, url):
        arm.move_joint("elbow_flex", -2.0)
        arm.release()
        _show(view)
        _post(url + "click", {"x": 1, "y": 1})
        module.ready(arm, view)
        joints = arm.get_joints()
        for name, angle in REST_JOINTS.items():
            assert joints[name] == pytest.approx(angle, abs=0.05)
        assert joints["gripper"] == pytest.approx(0.0)
        assert view.consume() == (1, 1)


def test_waiting_click_stops_the_move_before_it_runs():
    module = _load()
    with SO101.connect(mock=True) as arm, click_view(64, 48) as (view, url):
        arm.move_joint("elbow_flex", -2.0)
        goal = arm.get_pose()
        arm.home()
        start = arm.get_joints()["elbow_flex"]
        _show(view)
        _post(url + "click", {"x": 1, "y": 1})
        arrived = module.go_to(arm, view, (goal.x, goal.y, goal.z))
        assert arrived is False
        assert arm.get_joints()["elbow_flex"] == pytest.approx(start)
        assert _get(url + "status")["message"].startswith("Moving to")
