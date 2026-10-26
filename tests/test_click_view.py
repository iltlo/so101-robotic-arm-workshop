"""A new click must take over, and the page must show that without another click."""

import importlib.util
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from so101_control import SO101
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


def test_status_updates_after_the_click_returns():
    with click_view(640, 480) as (view, url):
        view.report("Moved to 0.282, -0.082, 0.020 m")
        body = _post(url + "click", {"x": 323, "y": 85})
        assert body["message"] == "Queued (323, 85)"
        view.report("Moving to 0.264, 0.048, 0.020 m")
        assert _get(url + "status")["message"] == "Moving to 0.264, 0.048, 0.020 m"
        page = urlopen(url).read().decode()
        assert "/status" in page


def test_newer_click_replaces_one_still_waiting():
    with click_view(64, 48) as (view, url):
        _post(url + "click", {"x": 1, "y": 2})
        _post(url + "click", {"x": 3, "y": 4})
        assert view.consume() == (3, 4)
        assert view.has_pending() is False
        assert view.consume() is None


def test_waiting_click_stops_the_move_before_it_runs():
    module = _load()
    with SO101.connect(mock=True) as arm, click_view(64, 48) as (view, url):
        arm.move_joint("elbow_flex", -2.0)
        goal = arm.get_pose()
        arm.home()
        start = arm.get_joints()["elbow_flex"]
        _post(url + "click", {"x": 1, "y": 1})
        arrived = module.go_to(arm, view, (goal.x, goal.y, goal.z))
        assert arrived is False
        assert arm.get_joints()["elbow_flex"] == pytest.approx(start)
        assert _get(url + "status")["message"].startswith("Moving to")
