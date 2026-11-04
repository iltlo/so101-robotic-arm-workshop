"""RGB capture from a webcam, a reCamera RTSP stream, or a RealSense. Depth is not used."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
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
# The RTSP demuxer blocks through the first keyframe. Later frames are quick.
_STARTUP_TIMEOUT_S = 20
# Camera input flags from the reCamera low-latency ffmpeg command.
# `fflags;nobuffer` and `flags;low_delay` drop SPS/PPS on this device, and
# `max_delay;0` leaves the default queue (about three seconds) to be played
# back in order. A short max_delay and no reorder queue make buffered frames
# return immediately, so the grab loop can keep the newest one.
_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|max_delay;500000|reorder_queue_size;0|rw_timeout;5000000"
)
# A frame already in the demuxer returns in about a millisecond.
# A frame from the camera waits out a frame interval.
_QUEUED_S = 0.008


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float

    @property
    def rgb_bgr(self) -> np.ndarray:
        return self.image


def camera_backend(cfg: dict) -> str:
    """`webcamera`, `recamera`, or `realsense`. A missing backend is the webcam."""
    backend = str(cfg.get("backend") or DEFAULT_BACKEND)
    if backend not in ("webcamera", "recamera", "realsense"):
        raise ValueError(f"Unknown camera backend {backend}")
    return backend


def _rtsp_endpoint(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise RuntimeError(f"reCamera URL has no host: {url}")
    return parsed.hostname, parsed.port or 554


def _require_rtsp_port(host: str, port: int) -> None:
    import socket

    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, port))
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise RuntimeError(
            f"reCamera RTSP port {port} on {host} is not accepting connections ({reason}). "
            "The page on port 80 can still load. The video service is not running; "
            "power-cycle the camera or start the dashboard preview, then try again."
        ) from exc
    finally:
        sock.close()


def _open_rtsp(url: str):
    import cv2

    host, port = _rtsp_endpoint(url)
    _require_rtsp_port(host, port)
    print(f"Opening the reCamera RTSP stream at {host}. "
          "The pictures on the camera web page are a separate preview. "
          "Waiting for the next H.264 keyframe takes about half a minute.",
          flush=True)
    # Joining mid-stream logs "non-existing PPS" for every slice before the keyframe.
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_CAPTURE_OPTIONS
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open reCamera stream {url}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class Recamera:
    """Latest BGR frame from an RTSP URL. Intrinsics come from config, not the stream."""

    def __init__(self, url: str, intrinsics: Intrinsics, *, capture=None):
        self.url = url
        self.intrinsics = intrinsics
        intrinsics.matrix()
        self._capture = capture if capture is not None else _open_rtsp(url)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._latest: Frame | None = None
        self._seq = 0
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open reCamera stream {url}")
            if capture is None:
                self._await_first_frame()
                frame = self._latest
                if frame is not None:
                    height, width = frame.image.shape[:2]
                    print(f"reCamera stream is up, {width}x{height}.", flush=True)
            self._thread = threading.Thread(target=self._run, name="recamera", daemon=True)
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
                raise RuntimeError("No reCamera frame")
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
        self._release()

    def _await_first_frame(self) -> None:
        """Publish the live picture, not the oldest frame in the RTSP queue."""
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        latest = None
        while time.monotonic() < deadline and not self._stop.is_set():
            ok, image, waited = self._read_capture()
            if not ok or image is None:
                time.sleep(0.01)
                continue
            latest = image
            if waited < _QUEUED_S:
                continue
            # The first paced read can still have the rest of the queue behind it.
            latest = self._take_queued(latest, deadline)
            self._publish(latest)
            if self._failed is not None:
                raise self._failed
            return
        raise RuntimeError("No reCamera frame")

    def _take_queued(self, latest, deadline: float):
        while time.monotonic() < deadline and not self._stop.is_set():
            ok, image, waited = self._read_capture()
            if not ok or image is None:
                return latest
            if waited >= _QUEUED_S:
                return image
            latest = image
        return latest

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
def open_camera():
    """Open the camera described in config/camera.yaml."""
    cfg = load_camera()
    backend = camera_backend(cfg)
    if backend == "webcamera":
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
        cam = Recamera(str(url), cfg["intrinsics"])
    else:
        serial = cfg.get("serial") or None
        cam = RGBRealSense(
            serial=serial, width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    try:
        yield cam
    finally:
        cam.close()
