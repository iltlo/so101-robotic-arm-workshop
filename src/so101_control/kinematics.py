"""Pinned LeRobot Placo FK/IK; body joints in degrees, TCP poses in metres."""

import numpy as np
from scipy.optimize import least_squares
from .config import JOINTS, transform


class Kinematics:
    def __init__(self, urdf: str):
        from lerobot.model import RobotKinematics

        self.solver = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link",
                                      joint_names=list(JOINTS))

    def fk(self, state: dict[str, float], gripper_T_tcp=None) -> np.ndarray:
        frame = transform(self.solver.forward_kinematics(
            np.array([state[name] for name in JOINTS], dtype=float)))
        return frame if gripper_T_tcp is None else frame @ transform(gripper_T_tcp)

    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of URDF gripper_link, distinct from gripper_frame_link TCP."""
        self.fk(state)  # Update Placo model with these measured arm joint angles.
        return transform(self.solver.robot.get_T_world_frame("gripper_link"))

    def simulate_tcp_ik(self, state: dict[str, float], target, gripper_T_tcp,
                        limits: dict[str, tuple[float, float]], *, tolerance_m: float = 0.015) -> dict[str, float]:
        """Offline position-only IK for the selected TCP; never used to authorize motion."""
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Invalid simulated TCP target")
        tcp = transform(gripper_T_tcp)
        if set(limits) != set(JOINTS):
            raise ValueError("Verified limits required for all arm joints")
        lower = np.array([limits[name][0] for name in JOINTS], dtype=float)
        upper = np.array([limits[name][1] for name in JOINTS], dtype=float)
        initial = np.array([state[name] for name in JOINTS], dtype=float)
        if not np.isfinite(initial).all() or not np.all(lower <= initial) or not np.all(initial <= upper):
            raise ValueError("Simulated starting pose outside joint limits")

        def error(angles):
            return self.fk(dict(zip(JOINTS, angles)), tcp)[:3, 3] - goal

        result = least_squares(error, initial, bounds=(lower, upper), max_nfev=100)
        if np.linalg.norm(error(result.x)) > tolerance_m:
            raise ValueError("IK did not reach requested TCP position within joint limits")
        return {name: float(value) for name, value in zip(JOINTS, result.x)}

    def hover_ik(self, state: dict[str, float], base_T_tcp: np.ndarray, gripper_T_tcp: np.ndarray,
                 limits: dict[str, tuple[float, float]], *, tolerance_m: float = 0.015) -> dict[str, float]:
        target = transform(base_T_tcp)[:3, 3]
        return self.simulate_tcp_ik(state, target, gripper_T_tcp, limits, tolerance_m=tolerance_m)