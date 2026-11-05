"""Camera backend choice and frame checks. No camera hardware."""

import base64
import json
import time

import cv2
import numpy as np
import pytest

from so101_utils.camera import (
    Recamera, WebCamera, _image_from_recamera_message, camera_backend, recamera_ws_url,
)
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


def test_closed_websocket_is_a_clear_error():
    with pytest.raises(RuntimeError, match="Could not open reCamera"):
        Recamera("ws://127.0.0.1:9/ws/boxes", _intrinsics(4, 2), timeout=1.0)


def test_rtsp_url_uses_the_supervisor_websocket():
    assert recamera_ws_url("rtsp://admin:admin@192.168.42.1:554/live") == (
        "ws://192.168.42.1:8090/ws/boxes"
    )
    assert recamera_ws_url("ws://192.168.42.1:8090/ws/boxes") == "ws://192.168.42.1:8090/ws/boxes"
    with pytest.raises(ValueError, match="no host"):
        recamera_ws_url("rtsp://")


def test_recamera_message_decodes_the_jpeg():
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = json.dumps({"data": {"image": base64.b64encode(encoded).decode(), "resolution": [8, 8]}})
    decoded = _image_from_recamera_message(payload)
    assert decoded is not None
    assert decoded.shape == (8, 8, 3)
    assert decoded[0, 0].tolist() == [10, 20, 30]
    assert _image_from_recamera_message('{"data": {"boxes": []}}') is None


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
    assert cfg["url"] == "ws://192.168.42.1:8090/ws/boxes"
    assert (cfg["width"], cfg["height"], int(cfg["fps"])) == (1280, 720, 30)
    intr = cfg["intrinsics"]
    assert intr.distortion_model == "brown_conrady"
    assert len(intr.distortion) == 5
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (
        1421.9603421676707, 1421.440722507306, 623.0756651086508, 358.594869293321,
    )


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


def test_recamera_reads_the_supervisor_websocket():
    from threading import Thread

    from websockets.sync.server import serve

    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = json.dumps({
        "data": {
            "image": base64.b64encode(encoded.tobytes()).decode(),
            "resolution": [8, 8],
        }
    })

    def handler(ws):
        try:
            while True:
                ws.send(payload)
                time.sleep(0.05)
        except Exception:
            return

    server = serve(handler, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.socket.getsockname()[1]
        cam = Recamera(f"ws://127.0.0.1:{port}/ws/boxes", _intrinsics(8, 8), timeout=2)
        try:
            frame = cam.read()
            assert frame.image.shape == (8, 8, 3)
            assert frame.image[0, 0].tolist() == [10, 20, 30]
        finally:
            cam.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_recamera_read_returns_the_bgr_frame():
    image = np.zeros((2, 4, 3), np.uint8)
    image[0, 0] = (1, 2, 3)
    cam = Recamera("ws://192.168.42.1:8090/ws/boxes", _intrinsics(4, 2), frames=[image])
    try:
        frame = cam.read()
        assert frame.image.shape == (2, 4, 3)
        assert frame.rgb_bgr[0, 0].tolist() == [1, 2, 3]
        latest = cam.poll()
        assert latest is not None and latest.image.shape == (2, 4, 3)
        assert cam.intrinsics.width == 4
        assert cam.url == "ws://192.168.42.1:8090/ws/boxes"
    finally:
        cam.close()


def test_recamera_rejects_the_wrong_frame_size():
    bad = np.zeros((3, 4, 3), np.uint8)
    cam = Recamera("ws://unused/ws/boxes", _intrinsics(4, 2), frames=[bad])
    try:
        with pytest.raises(RuntimeError, match="frame is 4x3"):
            cam.read()
    finally:
        cam.close()
