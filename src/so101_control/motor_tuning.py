"""Preview and save SO-101 station-side zero corrections, never motor registers."""

from pathlib import Path
import math
import shutil

import yaml

from .config import ALL_JOINTS
from .joint_offsets import station_offsets


STEP_DEG = 0.5
MAX_ADJUST_DEG = 30.0
GRIPPER_STEP_PERCENT = 1.0
MAX_GRIPPER_ADJUST_PERCENT = 10.0


class MotorZeroTuner:
    """Software calibration: raw LeRobot feedback + station corrections only."""

    def __init__(self, robot, config):
        if not robot.is_calibrated:
            raise ValueError("Complete matching LeRobot motor calibration required")
        self.robot = robot
        self.original = station_offsets(config)
        self.proposed = self.original.copy()
        self.selected = None

    @property
    def has_changes(self):
        return self.proposed != self.original

    def select(self, digit):
        if digit == "0":
            self.selected = None
        elif digit in "123456":
            self.selected = ALL_JOINTS[int(digit) - 1]
        else:
            raise ValueError("Select 0–6")
        return self.selected

    def adjust(self, direction):
        if self.selected is None or direction not in (-1, 1):
            raise ValueError("Select a joint (1–6), then use Up/Down")
        name = self.selected
        step = GRIPPER_STEP_PERCENT if name == "gripper" else STEP_DEG
        limit = MAX_GRIPPER_ADJUST_PERCENT if name == "gripper" else MAX_ADJUST_DEG
        proposed = round(self.proposed[name] + direction * step, 6)
        if abs(proposed) > limit:
            raise ValueError(f"Maximum station correction for {name} is ±{limit:g} "
                             f"{'%' if name == 'gripper' else 'degrees'}")
        self.proposed[name] = proposed
        return proposed

    def preview(self):
        """LeRobot already normalizes feedback; show station-corrected degrees."""
        observed = self.robot.get_observation()
        state = {}
        for name in ALL_JOINTS:
            value = float(observed[f"{name}.pos"]) + self.proposed[name]
            if not math.isfinite(value):
                raise ValueError(f"Invalid feedback for {name}")
            state[name] = value
        return state

    def commit(self, config_path):
        """Save only station.yaml; invalidate old FK/camera calibration."""
        if not self.has_changes:
            raise ValueError("No changes to save")
        self.preview()
        station = Path(config_path)
        config = yaml.safe_load(station.read_text())
        if not isinstance(config, dict):
            raise ValueError("Invalid station configuration")
        station_offsets(config)
        camera_file = station.parent / "calibration.yaml"
        camera_data = None
        if camera_file.is_file():
            camera_data = yaml.safe_load(camera_file.read_text())
            if not isinstance(camera_data, dict):
                raise ValueError("Cannot invalidate camera calibration")
        backup = station.with_name(station.name + ".before-zero-tuning.bak")
        shutil.copy2(station, backup)
        if camera_data is not None:
            camera_backup = camera_file.with_name(camera_file.name + ".before-zero-tuning.bak")
            shutil.copy2(camera_file, camera_backup)
            camera_data["accepted"] = False
            camera_file.write_text(yaml.safe_dump(camera_data, sort_keys=False))
        config["joint_zero_offsets_deg"] = {name: self.proposed[name] for name in ALL_JOINTS[:-1]}
        config["gripper_zero_offset_percent"] = self.proposed["gripper"]
        config["fk_tcp_verified"] = False
        config["base_plane_verified"] = False
        station.write_text(yaml.safe_dump(config, sort_keys=False))
        self.original = self.proposed.copy()
        return backup