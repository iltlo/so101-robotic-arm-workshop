"""Camera clicks read from `rr.poll_image_click`."""

import importlib.util
from pathlib import Path

import pytest
from rerun._image_clicks import note_viewer_event, reset_image_clicks
from rerun.experimental import EntitySelectionItem, SelectionChangeEvent

from so101_control import SO101
from so101_control.arm import REST_JOINTS
from so101_utils.views import CameraClicks

_SCRIPT = Path(__file__).resolve().parents[1] / "application" / "click_and_go.py"


def _load_click_and_go():
    spec = importlib.util.spec_from_file_location("click_and_go", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(x, y, *, path="camera/rgb"):
    return SelectionChangeEvent(
        application_id="so101_control",
        recording_id="recording",
        segment_id=None,
        items=[EntitySelectionItem(entity_path=path, position=[x, y, 0.0])],
    )


def _deliver(x, y, *, path="camera/rgb"):
    note_viewer_event(_selection(x, y, path=path))


def _clicks(width=8, height=8, *, ready=True):
    reset_image_clicks()
    clicks = CameraClicks(width, height)
    if ready:
        clicks.mark_ready()
    return clicks


def test_newer_click_replaces_one_still_waiting():
    clicks = _clicks(64, 48)
    _deliver(1, 2)
    _deliver(3, 4)
    assert clicks.poll() == (3, 4)
    assert clicks.has_pending() is False
    assert clicks.poll() is None


def test_click_before_the_camera_is_ready_is_ignored():
    clicks = _clicks(ready=False)
    _deliver(1, 1)
    assert clicks.has_pending() is False
    clicks.mark_ready()
    _deliver(2, 2)
    assert clicks.poll() == (2, 2)


def test_non_camera_selection_is_ignored():
    clicks = _clicks()
    _deliver(1, 1, path="/world/mesh")
    assert clicks.has_pending() is False


def test_ready_still_folds_when_a_click_is_waiting():
    module = _load_click_and_go()
    clicks = _clicks()
    _deliver(1, 1)
    with SO101.connect(mock=True, ee_tool_offset=module.EE_TOOL_OFFSET_M) as arm:
        arm.move_joint("elbow_flex", -2.0)
        arm.release()
        module.ready(arm)
        joints = arm.get_joints()
        for name, angle in REST_JOINTS.items():
            assert joints[name] == pytest.approx(angle, abs=0.05)
        assert clicks.poll() == (1, 1)


def test_mark_ready_prompts_once(capsys):
    clicks = CameraClicks(8, 8)
    clicks.mark_ready()
    clicks.mark_ready()
    assert capsys.readouterr().out.count("Camera image is ready") == 1


def test_camera_clicks_close_is_idempotent():
    clicks = _clicks()
    clicks.close()
    clicks.close()
