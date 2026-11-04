"""Camera backend choice and frame checks. No camera hardware."""

import time

import numpy as np
import pytest

from so101_utils.camera import Recamera, WebCamera, _open_rtsp, camera_backend
from so101_utils.config import load_camera
from so101_utils.transforms import Intrinsics


class _Frames:
    """Capture stand-in. `read` returns the listed images, then holds or ends."""

    def __init__(self, images, *, hold_last=False):
        self._images = list(images)
        self._index = 0
        self._hold_last = hold_last
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        if self.released:
            return False, None
        if self._index < len(self._images):
            image = self._images[self._index]
            self._index += 1
            return True, image
        if self._hold_last and self._images:
            time.sleep(0.01)
            return True, self._images[-1]
        time.sleep(0.01)
        return False, None

    def release(self):
        self.released = True


def _intrinsics(width, height):
    return Intrinsics(width, height, 1000, 1000, width / 2, height / 2, (), "none")


def test_closed_rtsp_port_is_a_clear_error():
    with pytest.raises(RuntimeError, match="not accepting connections"):
        _open_rtsp("rtsp://127.0.0.1:9/live")


def test_camera_backend_defaults_to_webcamera():
    assert camera_backend({}) == "webcamera"
    assert camera_backend({"backend": "recamera"}) == "recamera"
    assert camera_backend({"backend": "realsense"}) == "realsense"
    with pytest.raises(ValueError, match="usb"):
        camera_backend({"backend": "usb"})


def test_default_camera_config_is_webcamera():
    cfg = load_camera()
    assert cfg["backend"] == "webcamera"
    assert isinstance(cfg["device"], int)
    assert cfg["url"] == "rtsp://admin:admin@192.168.42.1:554/live"
    assert (cfg["width"], cfg["height"], int(cfg["fps"])) == (1280, 720, 30)
    intr = cfg["intrinsics"]
    assert intr.distortion_model == "none"
    assert intr.distortion == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (1000.0, 1000.0, 640.0, 360.0)


def test_webcam_device_must_be_an_index():
    with pytest.raises(ValueError, match="index"):
        WebCamera("C270", _intrinsics(4, 2), width=4, height=2, fps=30)


def test_missing_video_node_is_a_clear_error():
    with pytest.raises(RuntimeError, match="not present"):
        WebCamera(99, _intrinsics(4, 2), width=4, height=2, fps=30)


def test_webcamera_read_returns_the_bgr_frame():
    image = np.zeros((2, 4, 3), np.uint8)
    image[0, 0] = (4, 5, 6)
    cam = WebCamera("unused", _intrinsics(4, 2), width=4, height=2, fps=30,
                    capture=_Frames([image], hold_last=True))
    try:
        frame = cam.read()
        assert frame.image.shape == (2, 4, 3)
        assert frame.rgb_bgr[0, 0].tolist() == [4, 5, 6]
        assert cam.poll() is not None
    finally:
        cam.close()


def test_webcamera_rejects_the_wrong_frame_size():
    bad = np.zeros((3, 4, 3), np.uint8)
    cam = WebCamera("unused", _intrinsics(4, 2), width=4, height=2, fps=30,
                    capture=_Frames([bad]))
    try:
        with pytest.raises(RuntimeError, match="frame is 4x3"):
            cam.read()
    finally:
        cam.close()


def test_recamera_read_returns_the_bgr_frame():
    image = np.zeros((2, 4, 3), np.uint8)
    image[0, 0] = (1, 2, 3)
    cam = Recamera("rtsp://unused", _intrinsics(4, 2), capture=_Frames([image], hold_last=True))
    try:
        frame = cam.read()
        assert frame.image.shape == (2, 4, 3)
        assert frame.rgb_bgr[0, 0].tolist() == [1, 2, 3]
        latest = cam.poll()
        assert latest is not None and latest.image.shape == (2, 4, 3)
        assert cam.intrinsics.width == 4
    finally:
        cam.close()


def test_recamera_skips_frames_already_buffered():
    class _Burst:
        def __init__(self):
            self._n = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            if self.released:
                return False, None
            self._n += 1
            image = np.zeros((2, 4, 3), np.uint8)
            image[0, 0, 0] = self._n
            if self._n < 5:
                return True, image
            time.sleep(0.01)
            return True, image

        def release(self):
            self.released = True

    cam = Recamera("rtsp://unused", _intrinsics(4, 2), capture=_Burst())
    try:
        frame = cam.read()
        assert int(frame.image[0, 0, 0]) >= 5
    finally:
        cam.close()


def test_recamera_rejects_the_wrong_frame_size():
    bad = np.zeros((3, 4, 3), np.uint8)
    cam = Recamera("rtsp://unused", _intrinsics(4, 2), capture=_Frames([bad]))
    try:
        with pytest.raises(RuntimeError, match="frame is 4x3"):
            cam.read()
    finally:
        cam.close()
