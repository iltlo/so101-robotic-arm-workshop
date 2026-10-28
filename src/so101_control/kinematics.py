"""URDF forward kinematics (LeRobot Placo) and position IK. Joints in degrees, TCP in metres."""

import numpy as np
from scipy.optimize import least_squares

from so101_utils.transforms import transform

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Feedback can rest slightly past a calibrated stop. Commanded goals stay inside.
LIMIT_MARGIN_DEG = 2.0


class Kinematics:
    def __init__(self, urdf: str, ee_x_m: float = 0.0):
        from lerobot.model import RobotKinematics

        offset = float(ee_x_m)
        if not np.isfinite(offset):
            raise ValueError("ee_x_m must be finite")
        # Metres along gripper_frame_link +X. The solver tracks this point, not the frame origin.
        self.ee_x_m = offset
        self.solver = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link",
                                      joint_names=list(JOINTS))

    def fk(self, state: dict[str, float]) -> np.ndarray:
        """4x4 pose of gripper_frame_link in the base frame."""
        return transform(self.solver.forward_kinematics(
            np.array([state[name] for name in JOINTS], dtype=float)))

    def tool_point(self, pose: np.ndarray) -> np.ndarray:
        """Point `ee_x_m` along the EE X axis of a gripper_frame_link pose."""
        pose = np.asarray(pose, dtype=float)
        return pose[:3, 3] + pose[:3, :3] @ np.array([self.ee_x_m, 0.0, 0.0])

    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of gripper_link, where the AprilTag is mounted. Not the TCP."""
        self.fk(state)  # Update the Placo model with these joint angles.
        return transform(self.solver.robot.get_T_world_frame("gripper_link"))

    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = 0.015, downward: bool = False) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_x_m` along gripper_frame_link X, set when this
        object is built. Orientation is otherwise free. `downward` points
        gripper_frame +Z, the finger direction, straight down along the plane
        normal. That term is soft: only the tool point can reject the solve.
        """
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Invalid TCP target")
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
            if not downward:
                return position
            # Finger direction is gripper_frame +Z. Straight down is world -Z.
            return np.hstack([position, 0.05 * (pose[:3, 2] - np.array([0.0, 0.0, -1.0]))])

        result = least_squares(error, initial, bounds=(lower, upper), max_nfev=100)
        if np.linalg.norm(error(result.x)[:3]) > tolerance_m:
            raise ValueError("IK did not reach the point within joint limits")
        return {name: float(value) for name, value in zip(JOINTS, result.x)}
