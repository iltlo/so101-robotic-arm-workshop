"""Small adapter over LeRobot; no motor commands in offline mode."""

from typing import Protocol
from .config import ALL_JOINTS


class Robot(Protocol):
    def get_observation(self) -> dict: ...
    def send_action(self, action: dict) -> dict: ...
    def disconnect(self) -> None: ...


class OfflineRobot:
    def __init__(self, positions: dict[str, float] | None = None):
        self.positions = {f"{joint}.pos": 0.0 for joint in ALL_JOINTS}
        self.positions.update(positions or {})
        self.connected = True

    def get_observation(self) -> dict:
        if not self.connected:
            raise RuntimeError("Offline robot disconnected")
        return self.positions.copy()

    def send_action(self, action: dict) -> dict:
        if not self.connected:
            raise RuntimeError("Offline robot disconnected")
        if not set(action) <= set(self.positions):
            raise ValueError("Unknown joint")
        self.positions.update(action)
        return action.copy()

    def disconnect(self) -> None:
        self.connected = False


def connect_live(config: dict, *, enable_live: bool) -> Robot:
    if not enable_live:
        raise PermissionError("Live arm connection requires --enable-live")
    port = config.get("port")
    if not port:
        raise ValueError("Set port in station.yaml")
    arm_id = config.get("arm_id", "workshop_follower")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    follower = SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=arm_id,
            use_degrees=True,
            max_relative_target=5.0,
        )
    )
    # Setup/calibration is performed separately through the official LeRobot tools.
    try:
        follower.connect(calibrate=False)
        if not follower.is_calibrated:
            raise RuntimeError(
                "Motor calibration missing for arm_id={!r}. "
                "Run LeRobot calibration first, for example: "
                "uv run --locked --all-extras lerobot-calibrate "
                "--robot.type=so101_follower --robot.port={} --robot.id={}"
                .format(arm_id, port, arm_id)
            )
    except BaseException:
        # connect() can fail before LeRobot considers the whole robot connected.
        if follower.bus.is_connected:
            try:
                follower.disconnect()
            except Exception:
                follower.bus.disconnect(True)
        raise
    return follower