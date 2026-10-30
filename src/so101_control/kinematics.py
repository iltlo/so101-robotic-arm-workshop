"""URDF forward kinematics (LeRobot Placo) and pose IK. Joints in degrees, TCP in metres."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from so101_utils.model import JOINTS, gripper_rotation_from_rpy
from so101_utils.transforms import transform

# Feedback can rest slightly past a calibrated stop. Commanded goals stay inside.
LIMIT_MARGIN_DEG = 2.0
# One radian of orientation error weighs the same as this many metres of position error.
_ORIENTATION_M_PER_RAD = 0.05
# A short translation is solved inside this many degrees of the start. Outside it,
# the five-joint arm can spin a wrist through a long path that does not change the tool point.
NEAR_STEP_DEG = 8.0
# An orientation goal that does not fit in that window is solved again on the full limits.
NEAR_ORIENTATION_DEG = 1.0
# Residual weight that pulls gripper +X toward straight down, and the +X.z it accepts.
DOWNWARD_WEIGHT = 0.05
DOWNWARD_Z = -0.95
# Least-squares iteration caps for the near and full-limit solves.
NEAR_NFEV = 100
FULL_NFEV = 400
# A solved pose counts once its tool point is this close, and orientation this close.
DEFAULT_TOLERANCE_M = 0.015
DEFAULT_TOLERANCE_DEG = 8.0


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
                    *, tolerance_m: float = DEFAULT_TOLERANCE_M,
                    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
                    downward: bool = False, rpy=None) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_tool_offset` along gripper_frame_link +Z. `rpy` is
        base-frame XYZ Euler degrees, the angles a pose reports; omit it for free
        orientation. `downward` points the fingers (+X) straight down and leaves
        the twist about that axis free.

        The first attempt stays within `NEAR_STEP_DEG` of `state` so a 5 mm step
        does not spin a joint. A point outside that window, or an orientation off
        by more than `NEAR_ORIENTATION_DEG`, is solved again on the full limits.
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
            return np.hstack([position, DOWNWARD_WEIGHT * (pose[:3, 0] - np.array([0.0, 0.0, -1.0]))])

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
            if downward and self.fk(dict(zip(JOINTS, angles)))[2, 0] >= DOWNWARD_Z:
                return False
            return True

        near_lower = np.maximum(lower, initial - NEAR_STEP_DEG)
        near_upper = np.minimum(upper, initial + NEAR_STEP_DEG)
        solved = None
        if np.all(near_lower < near_upper):
            near = least_squares(error, initial, bounds=(near_lower, near_upper), max_nfev=NEAR_NFEV)
            orientation_fits = goal_rotation is None or orientation_error_deg(near.x) <= NEAR_ORIENTATION_DEG
            if orientation_fits and near_enough(near.x):
                solved = near.x
        if solved is None:
            result = least_squares(error, initial, bounds=(lower, upper), max_nfev=FULL_NFEV)
            # A folded or mid-move seed can sit in a local minimum that the home
            # joints escape, so retry from home before giving up.
            if position_error(result.x) > tolerance_m:
                home = np.clip(np.zeros_like(initial), lower, upper)
                alt = least_squares(error, home, bounds=(lower, upper), max_nfev=FULL_NFEV)
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
