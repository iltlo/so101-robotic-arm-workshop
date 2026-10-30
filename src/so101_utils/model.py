"""Joint names and the base-frame pose conventions shared by both packages."""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

# The five arm joints, in URDF order. The gripper is not one of them.
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Servo feedback for the jaw, percent. 0 is closed and 100 fully open.
GRIPPER_CLOSED_PERCENT = 0.0
GRIPPER_OPEN_PERCENT = 100.0

# Fingers within this many degrees of vertical are pitch ±90. The wrist then
# spins about base Z, which is yaw. A few degrees of lean still counts: the
# XYZ chart parks that spin in roll until the fingers are exactly vertical.
VERTICAL_DEG = 15.0


def _vertical_rpy(rotation: np.ndarray) -> np.ndarray | None:
    """Base yaw of a straight up or down gripper. None when it is not vertical."""
    fingers = rotation[:, 0]
    if float(np.hypot(fingers[0], fingers[1])) > np.sin(np.deg2rad(VERTICAL_DEG)):
        return None
    pitch = 90.0 if fingers[2] < 0.0 else -90.0
    side = rotation[:, 1]
    yaw = float(np.rad2deg(np.arctan2(-side[0], side[1])))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return np.array([0.0, pitch, yaw], dtype=float)


def tool_rpy(gripper_rotation) -> np.ndarray:
    """Roll, pitch, and yaw in degrees. XYZ Euler about the base axes.

    gripper_frame_link has X out the fingers. Roll is about base X, pitch about
    base Y, yaw about base Z. Pitch +90 points the fingers straight down.
    Within 15° of that pose a wrist twist is a yaw and roll stays 0.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    vertical = _vertical_rpy(rotation)
    if vertical is not None:
        return vertical
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return np.array([roll, pitch, yaw], dtype=float)


def gripper_rotation_from_rpy(rpy) -> np.ndarray:
    """gripper_frame_link rotation for base-frame XYZ Euler degrees."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float), degrees=True).as_matrix()
