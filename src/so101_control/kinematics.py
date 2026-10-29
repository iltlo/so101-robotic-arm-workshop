"""URDF forward kinematics (LeRobot Placo) and pose IK. Joints in degrees, TCP in metres."""

import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from so101_utils.transforms import transform

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Feedback can rest slightly past a calibrated stop. Commanded goals stay inside.
LIMIT_MARGIN_DEG = 2.0
# One radian of orientation error weighs the same as this many metres of position error.
_ORIENTATION_M_PER_RAD = 0.05
# A short translation is solved inside this many degrees of the start. Outside it,
# the five-joint arm can spin a wrist through a long path that does not change the tool point.
NEAR_STEP_DEG = 8.0
# An orientation goal that does not fit in that window is solved again on the full
# limits. The window used to keep a pitch that was still a degree off.
NEAR_ORIENTATION_DEG = 1.0

# Fingers within this many degrees of vertical are pitch ±90. The wrist then
# spins about base Z, which is yaw. A few degrees of lean still counts: the
# XYZ chart parks that spin in roll until the fingers are exactly vertical.
_VERTICAL_DEG = 15.0


def _vertical_rpy(rotation: np.ndarray) -> np.ndarray | None:
    """Base yaw of a straight up or down gripper. None when it is not vertical."""
    fingers = rotation[:, 0]
    if float(np.hypot(fingers[0], fingers[1])) > np.sin(np.deg2rad(_VERTICAL_DEG)):
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


class Kinematics:
    def __init__(self, urdf: str, ee_tool_offset: float = 0.0):
        from lerobot.model import RobotKinematics

        offset = float(ee_tool_offset)
        if not np.isfinite(offset):
            raise ValueError("ee_tool_offset must be finite")
        # Metres along gripper_frame_link +Z. Positive follows the blue axis.
        self.ee_tool_offset = offset
        self.solver = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link",
                                      joint_names=list(JOINTS))

    def fk(self, state: dict[str, float]) -> np.ndarray:
        """4x4 pose of gripper_frame_link in the base frame."""
        return transform(self.solver.forward_kinematics(
            np.array([state[name] for name in JOINTS], dtype=float)))

    def tool_point(self, pose: np.ndarray) -> np.ndarray:
        """Point `ee_tool_offset` along gripper_frame +Z."""
        pose = np.asarray(pose, dtype=float)
        return pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, self.ee_tool_offset])

    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of gripper_link, where the AprilTag is mounted. Not the TCP."""
        self.fk(state)
        return self.gripper_link()

    def gripper_link(self) -> np.ndarray:
        """gripper_link from the model last updated by fk()."""
        return transform(self.solver.robot.get_T_world_frame("gripper_link"))

    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = 0.015, tolerance_deg: float = 8.0,
                    downward: bool = False, rpy=None) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_tool_offset` along gripper_frame_link +Z.
        `rpy` is roll, pitch, and yaw in degrees about the base axes, the same
        angles a pose reports. Pitch +90 points the fingers straight down.
        Orientation is free when `rpy` is omitted.
        `downward` points +X (the fingers) straight down and leaves the twist
        about that axis free. An `rpy` target can be rejected when the
        orientation is outside `tolerance_deg`.

        The first attempt stays within `NEAR_STEP_DEG` of `state`, so a 5 mm
        step does not spin a joint. A point outside that window, or an
        orientation that misses by more than `NEAR_ORIENTATION_DEG`, is solved
        again on the full limits. If that still misses, one more solve starts from
        the home pose. A folded arm can sit in a local minimum that hides a
        point the home pose reaches. The mock arm is created already there.
        """
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Invalid TCP target")
        if downward and rpy is not None:
            raise ValueError("downward replaces an rpy target")
        goal_rotation = None
        if rpy is not None:
            angles = np.asarray(rpy, dtype=float)
            if angles.shape != (3,) or not np.isfinite(angles).all():
                raise ValueError("Invalid RPY target")
            goal_rotation = gripper_rotation_from_rpy(angles)
        if set(limits) != set(JOINTS):
            raise ValueError("Limits required for all arm joints")
        lower = np.array([limits[name][0] for name in JOINTS], dtype=float)
        upper = np.array([limits[name][1] for name in JOINTS], dtype=float)
        initial = np.array([state[name] for name in JOINTS], dtype=float)
        if not np.isfinite(initial).all() or not np.all(lower - LIMIT_MARGIN_DEG <= initial) \
                or not np.all(initial <= upper + LIMIT_MARGIN_DEG):
            raise ValueError("Starting pose outside joint limits")
        initial = np.clip(initial, lower, upper)

        def error(angles):
            joints = dict(zip(JOINTS, angles))
            pose = self.fk(joints)
            position = self.tool_point(pose) - goal
            if goal_rotation is not None:
                delta = Rotation.from_matrix(goal_rotation.T @ pose[:3, :3]).as_rotvec()
                return np.hstack([position, _ORIENTATION_M_PER_RAD * delta])
            if not downward:
                return position
            # +X (fingers) straight down. Twist about that axis stays free.
            return np.hstack([position, 0.05 * (pose[:3, 0] - np.array([0.0, 0.0, -1.0]))])

        def position_error(angles) -> float:
            return float(np.linalg.norm(self.tool_point(self.fk(dict(zip(JOINTS, angles)))) - goal))

        def orientation_error_deg(angles) -> float:
            pose = self.fk(dict(zip(JOINTS, angles)))
            delta = Rotation.from_matrix(goal_rotation.T @ pose[:3, :3]).magnitude()
            return float(np.rad2deg(delta))

        def near_enough(angles) -> bool:
            if position_error(angles) > tolerance_m:
                return False
            if goal_rotation is not None and orientation_error_deg(angles) > tolerance_deg:
                return False
            if downward and self.fk(dict(zip(JOINTS, angles)))[2, 0] >= -0.95:
                return False
            return True

        near_lower = np.maximum(lower, initial - NEAR_STEP_DEG)
        near_upper = np.minimum(upper, initial + NEAR_STEP_DEG)
        solved = None
        if np.all(near_lower < near_upper):
            near = least_squares(error, initial, bounds=(near_lower, near_upper), max_nfev=100)
            orientation_fits = goal_rotation is None or orientation_error_deg(near.x) <= NEAR_ORIENTATION_DEG
            if orientation_fits and near_enough(near.x):
                solved = near.x
        if solved is None:
            result = least_squares(error, initial, bounds=(lower, upper), max_nfev=400)
            # A click during homing used to leave the seed on a halfway pose.
            # The same point still solves from the home joints.
            if position_error(result.x) > tolerance_m:
                home = np.clip(np.zeros_like(initial), lower, upper)
                alt = least_squares(error, home, bounds=(lower, upper), max_nfev=400)
                if position_error(alt.x) < position_error(result.x):
                    result = alt
            if position_error(result.x) > tolerance_m:
                if downward:
                    raise ValueError(
                        "IK cannot reach that point with the fingers pointing down")
                raise ValueError("IK did not reach the point within joint limits")
            if goal_rotation is not None and orientation_error_deg(result.x) > tolerance_deg:
                raise ValueError("IK did not match the orientation within joint limits")
            solved = result.x
        return {name: float(value) for name, value in zip(JOINTS, solved)}
