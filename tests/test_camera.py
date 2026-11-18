"""Camera backend choice and the demo camera. No camera hardware."""

import time

import numpy as np
import pytest

from so101_utils.camera import (
    BACKENDS, DemoCamera, camera_backend, open_camera, realsense_uses_device_intrinsics,
)
from so101_utils.config import CONFIG, D405_FX, load_camera, save_camera_intrinsics


def test_the_main_camera_backends_are_supported():
    assert BACKENDS == ("recamera", "webcamera", "realsense", "demo")
    assert camera_backend({}) == "webcamera"
    assert camera_backend({"backend": "recamera"}) == "recamera"
    assert camera_backend({"backend": "realsense"}) == "realsense"
    assert camera_backend({"backend": "demo"}) == "demo"
    with pytest.raises(ValueError, match="usb"):
        camera_backend({"backend": "usb"})


def test_a_d405_keeps_the_profile_and_another_realsense_reads_the_device():
    assert realsense_uses_device_intrinsics("Intel RealSense D405") is False
    assert realsense_uses_device_intrinsics("Intel RealSense D435") is True


def test_a_realsense_profile_without_numbers_uses_the_d405(tmp_path):
    profile = tmp_path / "camera.yaml"
    profile.write_text("backend: realsense\nwidth: 640\nheight: 480\nfps: 30\n")
    intr = load_camera(profile)["intrinsics"]
    assert intr.fx == pytest.approx(D405_FX)
    assert (intr.cx, intr.cy) == (320.0, 240.0)
    with pytest.raises(ValueError, match="nothing to save"):
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
