"""Camera backend choice and the demo camera. No camera hardware."""

import time

import numpy as np
import pytest

from so101_utils.camera import BACKENDS, DemoCamera, camera_backend, open_camera
from so101_utils.config import CONFIG, load_camera, save_camera_intrinsics


def test_only_realsense_and_demo_are_supported():
    assert BACKENDS == ("realsense", "demo")
    assert camera_backend({}) == "realsense"
    assert camera_backend({"backend": "demo"}) == "demo"
    for removed in ("recamera", "webcamera", "usb"):
        with pytest.raises(ValueError, match="workshop uses"):
            camera_backend({"backend": removed})


def test_default_camera_config_is_a_realsense():
    cfg = load_camera()
    assert cfg["backend"] == "realsense"
    assert cfg["serial"] is None
    assert (cfg["width"], cfg["height"], int(cfg["fps"])) == (640, 480, 30)


def test_realsense_intrinsics_come_from_the_device_not_the_profile(tmp_path):
    profile = tmp_path / "camera.yaml"
    profile.write_text(
        "backend: realsense\nwidth: 640\nheight: 480\nintrinsics:\n"
        "  fx: 1.0\n  fy: 1.0\n  cx: 1.0\n  cy: 1.0\n"
    )
    assert load_camera(profile)["intrinsics"] is None
    with pytest.raises(ValueError, match="from the device"):
        save_camera_intrinsics(load_camera(CONFIG / "camera_demo.yaml")["intrinsics"], path=profile)


def test_demo_profile_has_working_intrinsics():
    cfg = load_camera(CONFIG / "camera_demo.yaml")
    assert cfg["backend"] == "demo"
    assert (cfg["width"], cfg["height"]) == (640, 480)
    intr = cfg["intrinsics"]
    assert intr.width == 640 and intr.height == 480
    assert 0 < intr.cx < intr.width and 0 < intr.cy < intr.height


def test_demo_camera_returns_frames():
    cfg = load_camera(CONFIG / "camera_demo.yaml")
    cam = DemoCamera(cfg["intrinsics"], width=cfg["width"], height=cfg["height"], fps=30)
    try:
        frame = cam.read()
        assert frame.image.shape == (480, 640, 3)
        assert frame.rgb_bgr is frame.image
        time.sleep(0.05)
        assert cam.poll() is not None
    finally:
        cam.close()


def test_open_camera_uses_the_demo_profile():
    cfg = load_camera(CONFIG / "camera_demo.yaml")
    with open_camera(cfg) as cam:
        assert isinstance(cam, DemoCamera)
        assert cam.read().image.dtype == np.uint8
