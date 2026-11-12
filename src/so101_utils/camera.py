"""RGB capture from an Intel RealSense or a synthetic demo scene.

Depth is not used. RealSense intrinsics come from the colour stream when the
camera opens; the demo profile carries fixed intrinsics.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from so101_utils.config import load_camera
from so101_utils.transforms import Intrinsics

BACKENDS = ("realsense", "demo")
DEFAULT_BACKEND = "realsense"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
FRAME_TIMEOUT_MS = 2000


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float

    @property
    def rgb_bgr(self) -> np.ndarray:
        return self.image


def camera_backend(cfg: dict) -> str:
    """`realsense` or `demo`. A missing backend is a RealSense."""
    backend = str(cfg.get("backend") or DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown camera backend {backend}; this workshop uses {', '.join(BACKENDS)}"
        )
    return backend


class DemoCamera:
    """Synthetic tabletop frames. Lets the page be exercised without a camera.

    The image is not a calibration target and its intrinsics are whatever the
    active profile says. It is for trying the UI and the mock arm.
    """

    intrinsics_from_device = False

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
    """Colour stream from an Intel RealSense. Intrinsics come from the device."""

    intrinsics_from_device = True

    def __init__(self, *, serial=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
        import pyrealsense2 as rs

        self.serial = serial
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
def open_camera(cfg: dict | None = None):
    """Open the camera in `cfg`, or the active config/camera.yaml."""
    cfg = load_camera() if cfg is None else cfg
    backend = camera_backend(cfg)
    if backend == "demo":
        cam = DemoCamera(
            cfg["intrinsics"], width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    else:
        cam = RGBRealSense(
            serial=cfg.get("serial") or None,
            width=cfg["width"], height=cfg["height"], fps=int(cfg["fps"]),
        )
    try:
        yield cam
    finally:
        cam.close()
