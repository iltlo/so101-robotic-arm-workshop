"""RGB-only RealSense capture; no depth stream or depth-based projection."""

from dataclasses import dataclass
import time
import numpy as np
from .geometry import Intrinsics


@dataclass
class Frame:
    rgb_bgr: np.ndarray
    timestamp: float  # local monotonic capture time


class RGBRealSense:
    def __init__(self, *, serial=None, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
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
            raise ValueError(f"Unsupported distortion model {raw.model}; calibrate/rectify RGB camera")
        self.intrinsics = Intrinsics(raw.width, raw.height, raw.fx, raw.fy, raw.ppx, raw.ppy,
                                     tuple(raw.coeffs), str(raw.model).removeprefix("distortion."))
        self.intrinsics.matrix()

    def read(self) -> Frame:
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        image = frames.get_color_frame()
        if not image:
            raise RuntimeError("No RealSense RGB image")
        return Frame(np.asanyarray(image.get_data()).copy(), time.monotonic())

    def close(self):
        self.pipeline.stop()


def require_fresh(frame: Frame, max_age_s=0.5):
    if time.monotonic() - frame.timestamp > max_age_s:
        raise ValueError("Stale RGB frame")