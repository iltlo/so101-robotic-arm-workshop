"""Offline checks for camera-optical -> AprilTag diagnostics."""

import numpy as np

from so101_control.geometry import Intrinsics
from so101_control.tag_view import TagPoseView, format_camera_T_tag


def test_tag_pose_text_reports_camera_frame_metres_and_rotation():
    pose = np.eye(4)
    pose[:3, 3] = [0.01, -0.02, 0.3]
    text = format_camera_T_tag(pose, tag_id=0)
    assert "camera_T_tag" in text
    assert "metres" in text
    assert "0.3000" in text
    assert "Tag ID: 0" in text


def test_view_logs_detection_and_clears_it_when_tag_disappears(monkeypatch):
    from so101_control import tag_view

    pose = np.eye(4)
    pose[:3, 3] = [0.04, 0.01, 0.3]
    corners = np.array([[100, 100], [140, 100], [140, 140], [100, 140]])
    observations = iter([(pose, corners), ValueError("No tag")])

    def detect(*args, **kwargs):
        result = next(observations)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(tag_view, "find_tag", detect)

    class Recording:
        def __init__(self):
            self.calls = []

        def log(self, path, *items, **kwargs):
            self.calls.append((path, items))

        def __getattr__(self, name):
            return lambda *args, **kwargs: (name, args, kwargs)

    rr = Recording()
    view = TagPoseView.__new__(TagPoseView)
    view.rr, view.detector = rr, object()
    view.intrinsics = Intrinsics(640, 480, 500, 500, 320, 240)
    view.tag_id, view.tag_size_m = 0, 0.04
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert np.allclose(view.update(image), pose)
    paths = [path for path, _ in rr.calls]
    assert "camera/tag" in paths
    assert "camera/tag_axes" in paths
    assert "camera/pose_text" in paths
    assert "camera/rgb" in paths
    assert view.update(image) is None
    assert ("camera/tag", (("Clear", (), {"recursive": True}),)) in rr.calls
    assert ("camera/tag_axes", (("Clear", (), {"recursive": False}),)) in rr.calls