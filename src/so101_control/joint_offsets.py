"""Station-side joint zero corrections; LeRobot motor calibration stays untouched."""

import math

from .config import ALL_JOINTS, JOINTS


def station_offsets(config):
    """Validate corrections in degrees (arm) and percent (gripper)."""
    values = config.get("joint_zero_offsets_deg", {})
    if not isinstance(values, dict) or set(values) - set(JOINTS):
        raise ValueError("joint_zero_offsets_deg must contain only SO-101 arm joints")
    offsets = {}
    for name in JOINTS:
        value = values.get(name, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 30:
            raise ValueError(f"Invalid zero offset for {name} (max ±30°)")
        offsets[name] = float(value)
    gripper = config.get("gripper_zero_offset_percent", 0.0)
    if not isinstance(gripper, (int, float)) or not math.isfinite(gripper) or abs(gripper) > 10:
        raise ValueError("Invalid gripper zero offset (max ±10%)")
    offsets["gripper"] = float(gripper)
    return offsets


class StationRobot:
    """Apply station corrections to feedback and invert them on motion goals."""

    def __init__(self, robot, config):
        self.robot = robot
        self.offsets = station_offsets(config)

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def get_observation(self):
        observed = self.robot.get_observation()
        return {key: value + self.offsets[key[:-4]] if key.endswith(".pos") and key[:-4] in ALL_JOINTS
                else value for key, value in observed.items()}

    def send_action(self, action):
        raw = {key: value - self.offsets[key[:-4]] if key.endswith(".pos") and key[:-4] in ALL_JOINTS
               else value for key, value in action.items()}
        returned = self.robot.send_action(raw)
        return {key: value + self.offsets[key[:-4]] if key.endswith(".pos") and key[:-4] in ALL_JOINTS
                else value for key, value in returned.items()}

    def disconnect(self):
        self.robot.disconnect()