"""Joint names and the base-frame pose conventions shared by both packages."""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

# The five arm joints, in URDF order. The gripper is not one of them.
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Servo feedback for the jaw, percent. 0 is closed and 100 fully open.
GRIPPER_CLOSED_PERCENT = 0.0
GRIPPER_OPEN_PERCENT = 100.0


def _wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def pitch_about_axis(gripper_rotation, axis) -> float:
    """Pitch about the flexion axis, in degrees.

    0 is level, +90 points the fingers straight down, and the angle keeps
    climbing as the fingers nod past vertical. The base XYZ Euler pitch cannot:
    past straight down it folds back under 90 and parks the extra half-turn on
    roll and yaw.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    axis = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Turn axis must be a non-zero direction")
    forward = np.cross(axis / length, np.array([0.0, 0.0, 1.0]))
    flat = float(np.linalg.norm(forward))
    if flat < 1e-9:
        return tool_pitch(rotation)
    finger = rotation[:, 0]
    return float(np.rad2deg(np.arctan2(-finger[2], float(np.dot(finger, forward / flat)))))


def tool_rpy(gripper_rotation, axis=None) -> np.ndarray:
    """Roll, pitch, and yaw in degrees, straight from the rotation matrix.

    gripper_frame_link has X out the fingers. Roll is about base X, pitch about
    base Y, yaw about base Z. Pitch +90 points the fingers straight down.
    The angles rebuild the pose exactly, so a pitch step is never hidden: they
    are the FK frame, not a display convenience. At pitch ±90 roll and yaw are
    not unique; SciPy reports one valid pair (third angle zero) that still
    rebuilds the same rotation.

    Pass the flexion axis to keep pitch continuous past straight down. SciPy's
    pitch stays inside ±90, so a nod to 113° is reported as pitch 67° with roll
    and yaw near ±180. The other Euler triple of the same rotation is the one
    whose pitch matches the nod about `axis`.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    primary = np.array([roll, pitch, yaw], dtype=float)
    if axis is None:
        return primary
    geometric = pitch_about_axis(rotation, axis)
    alternate = np.array([
        _wrap_deg(roll + 180.0),
        _wrap_deg(180.0 - pitch),
        _wrap_deg(yaw + 180.0),
    ], dtype=float)
    if abs(_wrap_deg(alternate[1] - geometric)) + 1e-9 < abs(_wrap_deg(primary[1] - geometric)):
        chosen = alternate
    else:
        chosen = primary
    rebuilt = gripper_rotation_from_rpy(chosen)
    if Rotation.from_matrix(rotation.T @ rebuilt).magnitude() > 1e-6:
        return primary
    return chosen


def gripper_rotation_from_rpy(rpy) -> np.ndarray:
    """gripper_frame_link rotation for base-frame XYZ Euler degrees."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float), degrees=True).as_matrix()


def spin_gripper(gripper_rotation, angle_deg: float) -> np.ndarray:
    """Rotation after turning about the gripper's own X axis, the fingers.

    A matrix in and a matrix out keeps every degree of lean. Control code
    carries this rotation between presses; near vertical the Euler triple is
    not unique, so rebuilding it from reported angles can move the lean.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    if not angle_deg:
        return rotation
    return rotation @ Rotation.from_euler("x", float(angle_deg), degrees=True).as_matrix()


def nod_gripper(gripper_rotation, axis, angle_deg: float) -> np.ndarray:
    """Rotation after a pitch turn about `axis`, a base-frame direction.

    `axis` is the arm's flexion axis: shoulder_lift, elbow_flex, and wrist_flex
    all turn about the same line, so the arm can hold the tool point through a
    turn about it. A pitch turn about the gripper's own Y is not the same
    rotation: away from a level tool the two axes differ by tens of degrees,
    and the five joints cannot hold the point through that turn.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    axis = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Turn axis must be a non-zero direction")
    if not angle_deg:
        return rotation
    turn = Rotation.from_rotvec(np.deg2rad(float(angle_deg)) * (axis / length))
    return turn.as_matrix() @ rotation


def tool_pitch(gripper_rotation) -> float:
    """Degrees the fingers point above the horizontal: +90 up, -90 down.

    This elevation folds at vertical. F and V drive the nod about the flexion
    axis (`pitch_about_axis`), which is the pitch `tool_rpy` reports when that
    axis is passed in. +pitch nods the fingers down and keeps going past 90.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    return float(np.rad2deg(np.arctan2(
        rotation[2, 0], float(np.hypot(rotation[0, 0], rotation[1, 0])))))


def pitch_axis(gripper_rotation, axis) -> np.ndarray:
    """`axis` signed so a positive turn increases pitch.

    Pitch + nods the fingers down: 0 is level, +90 is straight down, and the
    same direction continues past vertical. Signing it from the finger's
    horizontal part reverses exactly there, because that part flips as the
    fingers pass straight down, so pitch + lowered the gripper on one side of
    90 and raised it on the other.
    """
    rotation = np.asarray(gripper_rotation, dtype=float)
    axis = np.asarray(axis, dtype=float)
    before = pitch_about_axis(rotation, axis)
    after = pitch_about_axis(nod_gripper(rotation, axis, 1.0), axis)
    if _wrap_deg(after - before) < 0.0:
        return -axis
    return axis


def canonical_quaternion(gripper_rotation) -> np.ndarray:
    """Orientation as w, x, y, z with one sign, so a pose has one value.

    Euler angles are not unique at pitch ±90: several roll/yaw pairs rebuild
    the same tool. A quaternion has two signs for one rotation; this picks the
    one whose first non-zero component is positive.
    """
    x, y, z, w = Rotation.from_matrix(np.asarray(gripper_rotation, dtype=float)).as_quat()
    quaternion = np.array([w, x, y, z], dtype=float)
    for value in quaternion:
        if abs(value) > 1e-12:
            if value < 0.0:
                quaternion = -quaternion
            break
    return quaternion
