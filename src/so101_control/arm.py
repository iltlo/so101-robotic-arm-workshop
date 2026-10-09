"""Public arm API. Serial, offsets, and IK stay behind this class."""

import math
import time

from scipy.spatial.transform import Rotation

from .backend import MockBackend, OffsetBackend, connect_live
from .kinematics import JOINTS, Kinematics
from .motion import MotionError, joint_state, move_joints

MAX_SPEED_DEG_S = 30.0


class Pose:
    """Gripper pose in the robot base frame."""

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.roll, self.pitch, self.yaw = float(roll), float(pitch), float(yaw)

    def __iter__(self):
        return iter((self.x, self.y, self.z, self.roll, self.pitch, self.yaw))

    def __repr__(self) -> str:
        return (f"Pose(x={self.x:.4f} m, y={self.y:.4f} m, z={self.z:.4f} m, "
                f"roll={self.roll:.1f} deg, pitch={self.pitch:.1f} deg, yaw={self.yaw:.1f} deg)")


class SO101:
    """Connect, move, and read the follower. Use `with` so the port is closed.

    `mock=True` keeps an in-memory arm: same IK, no serial.
    """

    def __init__(self, backend, kinematics: Kinematics, limits, *, mock: bool, speed_deg_s: float = 10.0):
        self._backend = backend
        self._kin = kinematics
        self._limits = limits
        self.mock = mock
        self._speed = 10.0
        self._goal = None
        self.torque_on = True
        self.set_speed(speed_deg_s)

    @classmethod
    def connect(cls, mock: bool = False, config_path: str | None = None) -> "SO101":
        from so101_utils.config import joint_offsets, load_robot, validated_limits

        config = load_robot(config_path)
        limits = validated_limits(config)
        kinematics = Kinematics(config["urdf"])
        if mock:
            backend = MockBackend()
        else:
            backend = OffsetBackend(connect_live(config), joint_offsets(config))
        return cls(backend, kinematics, limits, mock=mock)

    def __enter__(self) -> "SO101":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self._backend.disconnect()

    def set_speed(self, deg_per_s: float) -> None:
        """Joint speed for later moves, in degrees per second (1 to 30)."""
        speed = float(deg_per_s)
        if not math.isfinite(speed) or not 1.0 <= speed <= MAX_SPEED_DEG_S:
            raise ValueError(f"Speed must be from 1 to {MAX_SPEED_DEG_S:.0f} deg/s")
        self._speed = speed

    def get_joints(self) -> dict[str, float]:
        """Arm joints in degrees. Gripper is percent: 0 closed, 100 open."""
        return joint_state(self._backend)

    def gripper_link_pose(self):
        """4x4 pose of gripper_link, where the AprilTag is mounted."""
        return self._kin.fk_gripper_link(self.get_joints())

    def get_pose(self) -> Pose:
        """TCP pose: gripper_frame_link, metres and XYZ Euler degrees."""
        matrix = self._kin.fk(self.get_joints())
        roll, pitch, yaw = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
        x, y, z = matrix[:3, 3]
        return Pose(x, y, z, roll, pitch, yaw)

    def move_to(self, x: float, y: float, z: float, r: float | None = None, *, wait: bool = True) -> None:
        """Move the TCP to x, y, z metres. `r` is optional wrist_roll in degrees.

        The other joints come from position IK. Orientation other than `r` is not tracked.
        """
        state = self.get_joints()
        limits = dict(self._limits)
        if r is not None:
            low, high = limits["wrist_roll"]
            roll = float(r)
            if not math.isfinite(roll) or not low <= roll <= high:
                raise ValueError("wrist_roll outside joint limits")
            # Keep this joint at r. A tiny window satisfies the solver's strict bounds.
            state = dict(state)
            state["wrist_roll"] = roll
            limits["wrist_roll"] = (roll - 1e-3, roll + 1e-3)
        targets = self._kin.position_ik(state, (float(x), float(y), float(z)), limits)
        if r is not None:
            targets["wrist_roll"] = float(r)
        self._goal = targets
        self._move(targets, wait=wait)

    def move_joint(self, name: str, delta_deg: float) -> None:
        """Step one arm joint. The step must be non-zero and at most 2 degrees."""
        delta = float(delta_deg)
        if name not in JOINTS or not math.isfinite(delta) or delta == 0 or abs(delta) > 2:
            raise ValueError("Use an arm joint and a step of at most 2 degrees")
        state = self.get_joints()
        self._move({name: state[name] + delta})

    def home(self) -> None:
        """Slow move to the calibrated midpoint: every arm joint at 0 degrees."""
        self._move({name: 0.0 for name in JOINTS})

    def grip(self) -> None:
        """Close the gripper."""
        self._gripper(0.0)

    def release(self) -> None:
        """Open the gripper."""
        self._gripper(100.0)

    def wait(self) -> None:
        """Block until the last move is within the settle tolerance."""
        if self._goal:
            self._move(self._goal, wait=True)

    def relax(self) -> None:
        """Torque off. Support the arm; it can fall. Not an emergency stop."""
        if not self.mock:
            self._backend.bus.disable_torque()
        self.torque_on = False
        print("Torque OFF. Support the arm; it can fall.")

    def hold(self) -> None:
        """Torque on, holding the position the arm is in right now."""
        if not self.mock:
            bus = self._backend.bus
            bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
            bus.enable_torque()
        self.torque_on = True
        print("Torque ON at the measured position.")

    def _move(self, targets: dict[str, float], *, wait: bool = True) -> None:
        self._goal = targets
        move_joints(
            self._backend, targets, self._limits,
            speed_deg_s=self._speed, period_s=0.01 if self.mock else 0.05,
            timeout_s=60.0, wait=wait,
        )

    def _gripper(self, percent: float) -> None:
        # Step by 5% so a live relative-target limit does not clip the goal.
        start = self.get_joints()["gripper"]
        steps = max(1, math.ceil(abs(percent - start) / 5.0))
        for index in range(1, steps + 1):
            goal = start + (percent - start) * index / steps
            actual = self._backend.send_action({"gripper.pos": goal})
            if abs(actual.get("gripper.pos", float("nan")) - goal) > 1e-3:
                raise MotionError("Driver clipped a gripper goal")
            if index < steps and not self.mock:
                time.sleep(0.05)
