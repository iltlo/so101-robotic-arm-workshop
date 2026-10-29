"""Public arm API. Serial, offsets, and IK stay behind this class."""

import math
import time

from .backend import MockBackend, OffsetBackend, connect_live, release_overload
from .kinematics import JOINTS, Kinematics, tool_rpy
from .motion import MotionError, joint_state, move_joints

MAX_SPEED_DEG_S = 90.0

# Metres along gripper_frame_link +Z. Positive follows the blue axis. 20 mm.
DEFAULT_EE_TOOL_OFFSET_M = 0.020

# Gripper percent. 0 is closed and 100 is fully open. Opening stops halfway.
OPEN_PERCENT = 50.0

# Folded pose, degrees. The gripper is closed separately (0%).
REST_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 45.0,
    "wrist_roll": 0.0,
}


class Pose:
    """Gripper pose in the robot base frame."""

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.roll, self.pitch, self.yaw = float(roll), float(pitch), float(yaw)

    def __repr__(self) -> str:
        return (f"Pose(x={self.x:.4f} m, y={self.y:.4f} m, z={self.z:.4f} m, "
                f"roll={self.roll:.1f} deg, pitch={self.pitch:.1f} deg, yaw={self.yaw:.1f} deg)")


class SO101:
    """Connect, move, and read the follower. Use `with` so the port is closed.

    `mock=True` keeps an in-memory arm: same IK, no serial.
    """

    def __init__(self, backend, kinematics: Kinematics, limits, *, mock: bool,
                 speed_deg_s: float = 90.0, ee_tool_offset: float = DEFAULT_EE_TOOL_OFFSET_M):
        self._backend = backend
        self._kin = kinematics
        self._limits = limits
        self.mock = mock
        self._speed = 90.0
        self._goal = None
        self._sent = None
        self._on_motion = None
        self._on_target = None
        self.torque_on = True
        offset = float(ee_tool_offset)
        if not math.isfinite(offset):
            raise ValueError("ee_tool_offset must be finite")
        self.ee_tool_offset = offset
        if kinematics is not None:
            kinematics.ee_tool_offset = offset
        self.set_speed(speed_deg_s)

    @classmethod
    def connect(cls, mock: bool = False, config_path: str | None = None, *,
                ee_tool_offset: float = DEFAULT_EE_TOOL_OFFSET_M) -> "SO101":
        from so101_utils.config import joint_offsets, load_robot, validated_limits

        config = load_robot(config_path)
        limits = validated_limits(config)
        kinematics = Kinematics(config["urdf"], ee_tool_offset=ee_tool_offset)
        if mock:
            backend = MockBackend()
        else:
            backend = OffsetBackend(connect_live(config), joint_offsets(config))
        return cls(backend, kinematics, limits, mock=mock, ee_tool_offset=ee_tool_offset)

    def __enter__(self) -> "SO101":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        # A jaw driven into its stop latches an overload fault. Goal = measured
        # position drops that fault so torque-off can finish and the port can close.
        self._unload_motors()
        try:
            self._backend.disconnect()
        except Exception as exc:
            bus = getattr(self._backend, "bus", None)
            if bus is not None and getattr(bus, "is_connected", False):
                try:
                    bus.disconnect(False)
                except Exception:
                    port = getattr(bus, "port_handler", None)
                    if port is not None:
                        try:
                            port.closePort()
                        except Exception:
                            pass
            print(f"Port closed. Torque-off stopped early: {exc}")

    def _unload_motors(self) -> None:
        if self.mock:
            return
        bus = getattr(self._backend, "bus", None)
        if bus is None or not getattr(bus, "is_connected", False):
            return
        try:
            release_overload(bus)
        except Exception:
            pass

    def track(self, callback) -> None:
        """Call `callback()` on connect and after every motion step."""
        self._on_motion = callback
        callback()

    def on_target(self, callback) -> None:
        """Call `callback(pose)` with the 4x4 goal pose solved by each `move_to`."""
        self._on_target = callback

    def set_speed(self, deg_per_s: float) -> None:
        """Joint speed for later moves, in degrees per second (1 to 120)."""
        speed = float(deg_per_s)
        if not math.isfinite(speed) or not 1.0 <= speed <= MAX_SPEED_DEG_S:
            raise ValueError(f"Speed must be from 1 to {MAX_SPEED_DEG_S:.0f} deg/s")
        self._speed = speed

    def get_joints(self) -> dict[str, float]:
        """Arm joints in degrees. Gripper is percent: 0 closed, 100 fully open."""
        return joint_state(self._backend)

    def gripper_link_pose(self):
        """4x4 pose of gripper_link, where the AprilTag is mounted."""
        return self._kin.fk_gripper_link(self.get_joints())

    def get_pose(self) -> Pose:
        """Tool pose: metres and XYZ Euler degrees.

        Position is the point `ee_tool_offset` along gripper_frame_link +Z, the
        same point `move_to` drives to the goal. Roll, pitch, and yaw are
        degrees about the base axes: X forward, Y left, Z up. Pitch +90 points
        the fingers straight down, and a wrist twist in that pose is a yaw.
        The frame drawn in Rerun stays on gripper_frame_link.
        """
        return self.pose_at(self.get_joints())

    def pose_at(self, joints: dict[str, float]) -> Pose:
        """Tool pose for a joint sample already read from the arm."""
        matrix = self._kin.fk(joints)
        roll, pitch, yaw = tool_rpy(matrix[:3, :3])
        x, y, z = self._kin.tool_point(matrix)
        return Pose(x, y, z, roll, pitch, yaw)

    def move_to(self, x: float, y: float, z: float,
                roll: float | None = None, pitch: float | None = None, yaw: float | None = None, *,
                wait: bool = True, cancel=None, downward: bool = False) -> bool:
        """Move the tool to a pose in the base frame.

        x, y, z are metres of the tool point: `ee_tool_offset` along
        gripper_frame_link +Z. roll, pitch, and yaw are degrees about the base
        axes, the angles `get_pose` reports. Pitch +90 points the fingers
        straight down. Pass all three or none.
        None leaves orientation free, unless `downward` points the fingers
        (+X) straight down. The twist about that axis stays free.
        Returns False when `cancel` stops the move before it finishes.
        """
        point = (float(x), float(y), float(z))
        given = (roll, pitch, yaw)
        if any(value is not None for value in given) and any(value is None for value in given):
            raise ValueError("roll, pitch, and yaw are one pose")
        rpy = None if roll is None else (float(roll), float(pitch), float(yaw))
        # Seed IK from the last command so a small step does not pick a new elbow
        # while the encoders are still catching up.
        state = self.get_joints()
        if self._sent and set(JOINTS) <= set(self._sent):
            state = {name: float(self._sent[name]) for name in JOINTS}
        targets = self._kin.position_ik(
            state, point, self._limits, downward=downward, rpy=rpy)
        if self._on_target is not None:
            self._on_target(self._kin.fk(targets))
        return self._move(targets, wait=wait, cancel=cancel)

    def move_joint(self, name: str, delta_deg: float) -> None:
        """Step one arm joint. The step must be non-zero and at most 2 degrees."""
        delta = float(delta_deg)
        if name not in JOINTS or not math.isfinite(delta) or delta == 0 or abs(delta) > 2:
            raise ValueError("Use an arm joint and a step of at most 2 degrees")
        base = self.get_joints()[name]
        if self._sent and name in self._sent:
            base = float(self._sent[name])
        self._move({name: base + delta})

    def home(self, *, wait: bool = True, cancel=None) -> bool:
        """Move to the calibrated midpoint: every arm joint at 0 degrees."""
        return self._move({name: 0.0 for name in JOINTS}, wait=wait, cancel=cancel)

    def rest(self, *, wait: bool = True, cancel=None) -> bool:
        """Fold the arm and close the gripper.

        shoulder_lift -90°, elbow_flex 90°, wrist_flex 45°, pan and roll at 0°.
        Returns False when `cancel` stops the arm move before the fold finishes.
        """
        arrived = self._move(dict(REST_JOINTS), wait=wait, cancel=cancel)
        if arrived:
            self.grip()
        return arrived

    def grip(self) -> None:
        """Close the gripper."""
        self._gripper(0.0)

    def release(self) -> None:
        """Open the gripper to the default stop, halfway."""
        self._gripper(OPEN_PERCENT)

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

    def _move(self, targets: dict[str, float], *, wait: bool = True, cancel=None) -> bool:
        self._goal = targets
        start = None
        if self._sent and set(targets) <= set(self._sent):
            start = {name: float(self._sent[name]) for name in targets}
        sent: dict[str, float] = {}
        done = move_joints(
            self._backend, targets, self._limits,
            # Mock waits a little longer between steps so Rerun can draw each pose.
            speed_deg_s=self._speed, period_s=0.03 if self.mock else 0.02,
            timeout_s=60.0, wait=wait, on_step=self._on_motion, cancel=cancel,
            start=start, commanded=sent,
        )
        if sent:
            if self._sent is None:
                self._sent = {}
            self._sent.update(sent)
        return done is not None

    def _gripper(self, percent: float) -> None:
        """Command the gripper (0 closed, 100 fully open).

        A jaw that stops short of the goal is on the stop or an object. The goal
        is rewritten to that position so the servo is not left in an overload fault.
        """
        self._backend.send_action({"gripper.pos": percent})
        if self._on_motion is not None:
            self._on_motion()
        if self.mock:
            return
        deadline = time.monotonic() + 2.0
        started = time.monotonic()
        previous = None
        stable_since = started
        while time.monotonic() < deadline:
            measured = self.get_joints()["gripper"]
            if abs(measured - percent) <= 8.0:
                return
            now = time.monotonic()
            if previous is None or abs(measured - previous) >= 0.8:
                stable_since = now
            previous = measured
            if now - started >= 0.25 and now - stable_since >= 0.2:
                self._hold_gripper(measured)
                return
            time.sleep(0.05)
        self._hold_gripper(self.get_joints()["gripper"])
        raise MotionError("Gripper did not settle")

    def _hold_gripper(self, percent: float) -> None:
        self._backend.send_action({"gripper.pos": percent})
        if self._on_motion is not None:
            self._on_motion()
        print(f"Gripper holding at {percent:.0f}%.")
