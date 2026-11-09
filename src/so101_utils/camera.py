"""RGB capture from a webcam, a reCamera websocket, a RealSense, or a demo scene.

Depth is not used.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

import numpy as np

from so101_utils.config import load_camera
from so101_utils.transforms import Intrinsics

DEFAULT_BACKEND = "webcamera"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
FRAME_TIMEOUT_MS = 2000
# How long open waits for the first supervisor JPEG.
_STARTUP_TIMEOUT_S = 10
# A frame already in the webcam queue returns in about a millisecond.
# A frame from the camera waits out a frame interval.
_QUEUED_S = 0.008
# Supervisor socket used by recamera_client.Recamera.frames.
_RECAMERA_WS_PORT = 8090
_RECAMERA_WS_PATH = "/ws/boxes"
# Short enough that close() unblocks, long enough to wait for the next JPEG.
_RECV_TIMEOUT_S = 0.5


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float

    @property
    def rgb_bgr(self) -> np.ndarray:
        return self.image


BACKENDS = ("recamera", "webcamera", "realsense", "demo")


def list_video_devices() -> list[int]:
    """Video indices the OS exposes. Windows and macOS use the index directly."""
    if not sys.platform.startswith("linux"):
        return []
    found = []
    for path in Path("/dev").glob("video*"):
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def camera_backend(cfg: dict) -> str:
    """`webcamera`, `recamera`, `realsense`, or `demo`. A missing backend is the webcam."""
    backend = str(cfg.get("backend") or DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown camera backend {backend}")
    return backend


def recamera_ws_url(url: str) -> str:
    """Supervisor websocket. An rtsp:// URL keeps its host and uses this socket."""
    text = str(url or "").strip()
    if text.startswith(("ws://", "wss://")):
        return text
    host = urlparse(text).hostname
    if not host:
        raise ValueError(f"reCamera URL has no host: {url}")
    return f"ws://{host}:{_RECAMERA_WS_PORT}{_RECAMERA_WS_PATH}"


def _image_from_recamera_message(raw) -> np.ndarray | None:
    """BGR image from one supervisor message, or None when it has no JPEG."""
    import base64
    import json

    import cv2

    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None
    encoded = data.get("image") or ""
    if not encoded:
        return None
    try:
        jpeg = base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


class Recamera:
    """Latest BGR frame from the reCamera supervisor websocket.

    Intrinsics come from config, not the stream. `frames` yields BGR images
    and skips the socket; tests use it.
    """

    def __init__(self, url: str, intrinsics: Intrinsics, *, frames=None, timeout: float = 10.0,
                 on_payload=None):
        self.url = recamera_ws_url(url)
        self.intrinsics = intrinsics
        intrinsics.matrix()
        self._timeout = timeout
        self._frames = frames
        # The supervisor message carries the JPEG and, when the on-device model
        # runs, the YOLO boxes. One socket feeds both so the agent is still the
        # only camera client.
        self._on_payload = on_payload
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._latest: Frame | None = None
        self._seq = 0
        self._delivered = 0
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None
        try:
            if frames is None:
                print(f"Opening the reCamera stream at {self.url}.", flush=True)
            self._thread = threading.Thread(target=self._run, name="recamera", daemon=True)
            self._thread.start()
            if frames is None:
                self._await_first_frame()
                frame = self._latest
                if frame is not None:
                    height, width = frame.image.shape[:2]
                    print(f"reCamera stream is up, {width}x{height}.", flush=True)
        except Exception:
            self.close()
            raise

    def read(self) -> Frame:
        """Block until a frame newer than the last `read`, or raise at the timeout."""
        deadline = time.monotonic() + FRAME_TIMEOUT_MS / 1000
        with self._lock:
            seen = self._delivered
            while self._seq == seen and self._failed is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ready.wait(remaining)
            self._raise_if_stopped()
            if self._seq == seen:
                raise RuntimeError("No reCamera frame")
            self._delivered = self._seq
            return self._latest

    def poll(self) -> Frame | None:
        """Latest RGB frame, or None. Does not block."""
        with self._lock:
            self._raise_if_stopped()
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._ready.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_STARTUP_TIMEOUT_S)
            if thread.is_alive():
                return

    def _await_first_frame(self) -> None:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        with self._lock:
            while self._seq == 0 and self._failed is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ready.wait(remaining)
            self._raise_if_stopped()
            if self._seq == 0:
                raise RuntimeError("No reCamera frame")

    def _run(self) -> None:
        try:
            source = self._frames if self._frames is not None else self._websocket_frames()
            for image in source:
                if self._stop.is_set():
                    return
                if image is None:
                    continue
                self._publish(image)
                if self._failed is not None:
                    return
        except Exception as exc:
            self._fail(exc)
        finally:
            with self._lock:
                self._ready.notify_all()

    def _websocket_frames(self):
        """JPEGs from the supervisor socket. Reconnect after the first picture."""
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        delay = 1.0
        connected = False
        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=self._timeout, max_size=None) as ws:
                    delay = 1.0
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=_RECV_TIMEOUT_S)
                        except TimeoutError:
                            continue
                        if self._on_payload is not None:
                            try:
                                self._on_payload(raw)
                            except Exception:
                                pass
                        image = _image_from_recamera_message(raw)
                        if image is None:
                            continue
                        connected = True
                        yield image
            except (ConnectionClosed, TimeoutError, OSError) as exc:
                if self._stop.is_set():
                    return
                if not connected:
                    raise RuntimeError(
                        f"Could not open reCamera stream {self.url} ({exc})"
                    ) from exc
                if self._stop.wait(delay):
                    return
                delay = min(5.0, delay * 1.5)

    def _publish(self, image) -> None:
        height, width = int(image.shape[0]), int(image.shape[1])
        if width != self.intrinsics.width or height != self.intrinsics.height:
            self._fail(RuntimeError(
                f"reCamera frame is {width}x{height}, "
                f"config is {self.intrinsics.width}x{self.intrinsics.height}"))
            return
        frame = Frame(np.array(image, copy=True), time.monotonic())
        with self._lock:
            self._latest = frame
            self._seq += 1
            self._ready.notify_all()

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._failed is None:
                self._failed = exc
            self._ready.notify_all()

    def _raise_if_stopped(self) -> None:
        if self._failed is not None:
            raise self._failed
        if self._stop.is_set():
            raise RuntimeError("reCamera closed")


def _webcam_index(device) -> int:
    """OpenCV camera index. Linux uses it as `/dev/videoN`."""
    try:
        return int(device)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera.yaml device must be a camera index, got {device!r}") from exc


def _webcam_api():
    import cv2

    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _webcam_holders(path: str) -> str:
    """Process names that already have the device open, if `fuser` can say so."""
    import subprocess

    try:
        result = subprocess.run(
            ["fuser", "-v", path], capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    names = []
    for line in (result.stdout + result.stderr).splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if any(part[:1] == "F" and "." in part for part in parts):
            names.append(parts[-1])
    if not names:
        return ""
    return "In use by " + ", ".join(dict.fromkeys(names)) + "."


def _open_webcam(device, width: int, height: int, fps: int):
    import cv2

    index = _webcam_index(device)
    path = f"/dev/video{index}" if sys.platform.startswith("linux") else ""
    if path and not os.path.exists(path):
        raise RuntimeError(f"Webcam {path} is not present.")
    label = path or f"index {index}"
    print(f"Opening webcam {label}.", flush=True)
    # CAP_V4L2 rejects a path ("can't be used to capture by name"), so Linux
    # opens the index. Windows uses that same index with DirectShow, macOS with AVFoundation.
    capture = cv2.VideoCapture(index, _webcam_api())
    if not capture.isOpened():
        capture.release()
        holders = _webcam_holders(path) if path else ""
        detail = f" {holders}" if holders else ""
        raise RuntimeError(
            f"Could not open webcam {label}.{detail} "
            "Close the other viewer and try again."
        )
    # 720p30 on this webcam is MJPEG. Uncompressed YUYV at this size is under 8 fps.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_FPS, int(fps))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class WebCamera:
    """Latest BGR frame from a V4L2 webcam. Intrinsics come from config, not the device."""

    def __init__(self, device, intrinsics: Intrinsics, *, width, height, fps, capture=None):
        self.device = str(device)
        self.intrinsics = intrinsics
        intrinsics.matrix()
        self._capture = capture if capture is not None else _open_webcam(device, width, height, fps)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._latest: Frame | None = None
        self._seq = 0
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open webcam {device}")
            if capture is None:
                self._await_first_frame()
                frame = self._latest
                if frame is not None:
                    got_h, got_w = frame.image.shape[:2]
                    print(f"Webcam is up, {got_w}x{got_h}.", flush=True)
            self._thread = threading.Thread(target=self._run, name="webcamera", daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    def read(self) -> Frame:
        """Block until a frame newer than the last `read`, or raise at the timeout."""
        deadline = time.monotonic() + FRAME_TIMEOUT_MS / 1000
        with self._lock:
            seen = self._seq
            while self._seq == seen and self._failed is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._ready.wait(remaining)
            self._raise_if_stopped()
            if self._seq == seen:
                raise RuntimeError("No webcam frame")
            return self._latest

    def poll(self) -> Frame | None:
        """Latest RGB frame, or None. Does not block."""
        with self._lock:
            self._raise_if_stopped()
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._ready.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
            if thread.is_alive():
                return
        self._release()

    def _await_first_frame(self) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self._stop.is_set():
            image = self._newest()
            if image is None:
                continue
            self._publish(image)
            if self._failed is not None:
                raise self._failed
            return
        raise RuntimeError("No webcam frame")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                image = self._newest()
                if self._stop.is_set():
                    return
                if image is None:
                    time.sleep(0.01)
                    continue
                self._publish(image)
                if self._failed is not None:
                    return
        except Exception as exc:
            self._fail(exc)
        finally:
            self._release()
            with self._lock:
                self._ready.notify_all()

    def _newest(self):
        """Newest frame. Pictures that return immediately were already buffered."""
        latest = None
        while not self._stop.is_set():
            ok, image, waited = self._read_capture()
            if not ok or image is None:
                return latest
            latest = image
            if waited >= _QUEUED_S:
                return latest
        return latest

    def _read_capture(self):
        capture = self._capture
        if capture is None:
            return False, None, 0.0
        started = time.monotonic()
        ok, image = capture.read()
        return bool(ok), image, time.monotonic() - started

    def _release(self) -> None:
        with self._lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()

    def _publish(self, image) -> None:
        height, width = int(image.shape[0]), int(image.shape[1])
        if width != self.intrinsics.width or height != self.intrinsics.height:
            self._fail(RuntimeError(
                f"Webcam frame is {width}x{height}, "
                f"config is {self.intrinsics.width}x{self.intrinsics.height}"))
            return
        frame = Frame(np.array(image, copy=True), time.monotonic())
        with self._lock:
            self._latest = frame
            self._seq += 1
            self._ready.notify_all()

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._failed is None:
                self._failed = exc
            self._ready.notify_all()

    def _raise_if_stopped(self) -> None:
        if self._failed is not None:
            raise self._failed
        if self._stop.is_set():
            raise RuntimeError("Webcam closed")


class DemoCamera:
    """Synthetic tabletop frames. Lets the page be exercised without a camera.

    The image is not a calibration target and its intrinsics are whatever the
    active profile says. It is for trying the UI and the mock arm.
    """

    def __init__(self, intrinsics: Intrinsics, *, width=None, height=None, fps=30):
        self.intrinsics = intrinsics
        intrinsics.matrix()
        self.width = int(width or intrinsics.width)
        self.height = int(height or intrinsics.height)
        self.fps = max(1, int(fps))
        self._stop = threading.Event()
        self._next = 0.0
        self._started = time.monotonic()

    def poll(self) -> Frame | None:
        if self._stop.is_set():
            return None
        now = time.monotonic()
        if now < self._next:
            return None
        self._next = now + 1.0 / self.fps
        return Frame(self._render(now), now)

    def read(self) -> Frame:
        deadline = time.monotonic() + FRAME_TIMEOUT_MS / 1000
        while time.monotonic() < deadline:
            frame = self.poll()
            if frame is not None:
                return frame
            time.sleep(0.005)
        raise RuntimeError("No demo frame")

    def close(self) -> None:
        self._stop.set()

    def _render(self, now: float) -> np.ndarray:
        import cv2

        width, height = self.width, self.height
        image = np.zeros((height, width, 3), np.uint8)
        top = np.array([58, 62, 66], np.float32)
        bottom = np.array([22, 24, 27], np.float32)
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        image[:] = np.clip(top * (1.0 - ramp) + bottom * ramp, 0, 255).astype(np.uint8)
        step = max(40, width // 20)
        for x in range(0, width, step):
            cv2.line(image, (x, height // 2), (x, height), (40, 46, 52), 1)
        for y in range(height // 2, height, step):
            cv2.line(image, (0, y), (width, y), (40, 46, 52), 1)
        # Three "objects" on the table for the pick panels to describe.
        scale = width / 1280.0
        objects = (
            (int(360 * scale), int(520 * scale), (60, 150, 240), "cube"),
            (int(620 * scale), int(560 * scale), (120, 210, 120), "cup"),
            (int(880 * scale), int(500 * scale), (230, 160, 80), "block"),
        )
        for cx, cy, color, label in objects:
            half = int(42 * scale)
            cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), color, -1)
            cv2.putText(image, label, (cx - half, cy + half + int(26 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (220, 226, 232),
                        max(1, int(2 * scale)), cv2.LINE_AA)
        sweep = (math.sin((now - self._started) * 1.4) + 1.0) / 2.0
        x = int(width * (0.1 + 0.8 * sweep))
        cv2.circle(image, (x, int(height * 0.2)), int(14 * scale), (80, 220, 140), -1)
        cv2.putText(image, "SO-101 demo camera", (int(24 * scale), int(44 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (235, 240, 245),
                    max(1, int(2 * scale)), cv2.LINE_AA)
        return image


class RGBRealSense:
    def __init__(self, *, serial=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        setup = rs.config()
        if serial:
            setup.enable_device(str(serial))
        setup.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(setup)
        raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        if raw.model not in (rs.distortion.none, rs.distortion.brown_conrady,
                             rs.distortion.inverse_brown_conrady):
            self.pipeline.stop()
            raise ValueError(f"Unsupported distortion model {raw.model}")
        self.intrinsics = Intrinsics(
            raw.width, raw.height, raw.fx, raw.fy, raw.ppx, raw.ppy,
            tuple(raw.coeffs), str(raw.model).removeprefix("distortion."),
        )
        self.intrinsics.matrix()

    def read(self) -> Frame:
        frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
        return self._frame(frames)

    def poll(self) -> Frame | None:
        """Latest RGB frame, or None. Does not block."""
        try:
            ready, frames = self.pipeline.try_wait_for_frames(0)
        except RuntimeError:
            return None
        if not ready:
            return None
        image = frames.get_color_frame()
        if not image:
            return None
        return Frame(np.asanyarray(image.get_data()).copy(), time.monotonic())

    def _frame(self, frames) -> Frame:
        image = frames.get_color_frame()
        if not image:
            raise RuntimeError("No RealSense RGB image")
        return Frame(np.asanyarray(image.get_data()).copy(), time.monotonic())

    def close(self) -> None:
        self.pipeline.stop()


@contextmanager
def open_camera(cfg: dict | None = None, *, on_payload=None):
    """Open the camera in `cfg`, or the active config/camera.yaml."""
    cfg = load_camera() if cfg is None else cfg
    backend = camera_backend(cfg)
    if backend == "demo":
        cam = DemoCamera(
            cfg["intrinsics"], width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    elif backend == "webcamera":
        device = cfg.get("device")
        if device is None or device == "":
            raise ValueError("camera.yaml backend webcamera needs a device")
        cam = WebCamera(
            device, cfg["intrinsics"],
            width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    elif backend == "recamera":
        url = cfg.get("url")
        if not url:
            raise ValueError("camera.yaml backend recamera needs a url")
        cam = Recamera(str(url), cfg["intrinsics"], on_payload=on_payload)
    else:
        serial = cfg.get("serial") or None
        cam = RGBRealSense(
            serial=serial, width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    try:
        yield cam
    finally:
        cam.close()
