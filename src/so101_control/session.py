"""Context-managed follower and RGB camera sessions; offline robot by default."""

from contextlib import contextmanager

from .robot import OfflineRobot, connect_live
from .joint_offsets import StationRobot


@contextmanager
def station_robot(config, live):
    from .joint_offsets import station_offsets

    if live:
        station_offsets(config)  # reject invalid offsets before opening serial port
    robot = connect_live(config, enable_live=True) if live else OfflineRobot()
    try:
        yield StationRobot(robot, config) if live else robot
    finally:
        robot.disconnect()


@contextmanager
def rgb_camera(config):
    from .camera import RGBRealSense
    cam = RGBRealSense(serial=config.get("camera_serial"), width=config.get("camera_width", 640),
                       height=config.get("camera_height", 480), fps=config.get("camera_fps", 30))
    try:
        yield cam
    finally:
        cam.close()


def capture_tag(cam):
    """Discard buffered RGB frames before detecting a tag in the settled pose."""
    from .calibration import detect_tag

    for _ in range(4):
        frame = cam.read()
    return detect_tag(frame.rgb_bgr, cam.intrinsics)