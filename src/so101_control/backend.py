"""Live Feetech follower and an in-memory stand-in. Both speak joint degrees."""

from .kinematics import JOINTS


class MockBackend:
    """Joint angles in the same frame as a calibrated arm. No serial port."""

    def __init__(self):
        self.positions = {f"{joint}.pos": 0.0 for joint in (*JOINTS, "gripper")}
        self.connected = True

    def get_observation(self) -> dict:
        if not self.connected:
            raise RuntimeError("Mock arm is closed")
        return self.positions.copy()

    def send_action(self, action: dict) -> dict:
        if not self.connected:
            raise RuntimeError("Mock arm is closed")
        if not set(action) <= set(self.positions):
            raise ValueError("Unknown joint")
        self.positions.update(action)
        return action.copy()

    def disconnect(self) -> None:
        self.connected = False


class OffsetBackend:
    """Add station joint offsets on read, and subtract them before a motor goal."""

    def __init__(self, robot, offsets: dict[str, float]):
        self.robot = robot
        self.offsets = offsets

    def _correct(self, values: dict, sign: float) -> dict:
        corrected = {}
        for key, value in values.items():
            name = key[:-4] if key.endswith(".pos") else None
            corrected[key] = value + sign * self.offsets[name] if name in self.offsets else value
        return corrected

    def get_observation(self) -> dict:
        return self._correct(self.robot.get_observation(), 1.0)

    def send_action(self, action: dict) -> dict:
        return self._correct(self.robot.send_action(self._correct(action, -1.0)), 1.0)

    def disconnect(self) -> None:
        self.robot.disconnect()

    def __getattr__(self, name):
        return getattr(self.robot, name)


def connect_live(config: dict):
    port = config.get("port")
    if not port:
        raise ValueError("Set port in config/robot.yaml")
    arm_id = config.get("arm_id", "workshop_follower")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # No relative cap here. Joint steps are limited in move_joints. The gripper
    # is 0–100, so a cap of 10 (meant for degrees) clips a normal open/close.
    follower = SO101Follower(SO101FollowerConfig(
        port=port, id=arm_id, use_degrees=True, max_relative_target=None,
    ))
    try:
        follower.connect(calibrate=False)
        if not follower.is_calibrated:
            raise RuntimeError(
                f"No motor calibration for arm_id={arm_id!r}. Run lerobot-calibrate first "
                f"(see the README) for port {port}."
            )
    except BaseException:
        if follower.bus.is_connected:
            try:
                follower.disconnect()
            except Exception:
                follower.bus.disconnect(True)
        raise
    return follower
