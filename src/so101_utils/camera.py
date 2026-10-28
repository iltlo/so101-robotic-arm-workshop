"""RGB-only RealSense capture. Depth is not used."""

from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np

from so101_utils.config import load_camera
from so101_utils.transforms import Intrinsics


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float

    @property
    def rgb_bgr(self) -> np.ndarray:
        return self.image


class RGBRealSense:
    def __init__(self, *, serial=None, width=640, height=480, fps=30):
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
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
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
def camera():
    """Open the RealSense described in config/camera.yaml."""
    cfg = load_camera()
    serial = cfg.get("serial") or None
    cam = RGBRealSense(
        serial=serial, width=cfg["width"], height=cfg["height"], fps=int(cfg.get("fps", 30)),
    )
    try:
        yield cam
    finally:
        cam.close()


open_camera = camera
