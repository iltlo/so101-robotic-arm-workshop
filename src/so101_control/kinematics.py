"""URDF forward kinematics (LeRobot Placo) and position-only IK. Joints in degrees, TCP in metres."""

import numpy as np
from scipy.optimize import least_squares

from so101_utils.transforms import transform

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class Kinematics:
    def __init__(self, urdf: str):
        from lerobot.model import RobotKinematics

        self.solver = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link",
                                      joint_names=list(JOINTS))

    def fk(self, state: dict[str, float]) -> np.ndarray:
        """4x4 pose of gripper_frame_link in the base frame."""
        return transform(self.solver.forward_kinematics(
            np.array([state[name] for name in JOINTS], dtype=float)))

    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of gripper_link, where the AprilTag is mounted. Not the TCP."""
        self.fk(state)  # Update the Placo model with these joint angles.
        return transform(self.solver.robot.get_T_world_frame("gripper_link"))

    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = 0.015) -> dict[str, float]:
        """Solve the five arm joints so the TCP reaches target xyz. Orientation is ignored."""
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Invalid TCP target")
        if set(limits) != set(JOINTS):
            raise ValueError("Limits required for all arm joints")
        lower = np.array([limits[name][0] for name in JOINTS], dtype=float)
        upper = np.array([limits[name][1] for name in JOINTS], dtype=float)
        initial = np.array([state[name] for name in JOINTS], dtype=float)
        if not np.isfinite(initial).all() or not np.all(lower <= initial) or not np.all(initial <= upper):
            raise ValueError("Starting pose outside joint limits")

        def error(angles):
            return self.fk(dict(zip(JOINTS, angles)))[:3, 3] - goal

        result = least_squares(error, initial, bounds=(lower, upper), max_nfev=100)
        if np.linalg.norm(error(result.x)) > tolerance_m:
            raise ValueError("IK did not reach the point within joint limits")
        return {name: float(value) for name, value in zip(JOINTS, result.x)}
