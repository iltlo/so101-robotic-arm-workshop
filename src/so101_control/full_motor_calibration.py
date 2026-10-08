"""Workshop-owned SO-101 midpoint/range capture using LeRobot motor I/O.

All capture and Rerun preview are read-only. No registers change before an
explicit save, and no motion goals are ever sent.
"""

from dataclasses import replace
from pathlib import Path
import shutil

import yaml

from .config import ALL_JOINTS


RESOLUTION = 4095
HALF_TURN = RESOLUTION // 2
MIN_TRAVEL_TICKS = 200


class FullMotorCalibration:
    """Capture a supported midpoint and full safe range, then commit once."""

    def __init__(self, robot):
        from lerobot.motors import MotorCalibration

        self.robot = robot
        self.bus = robot.bus
        self.old = self.bus.read_calibration()
        if set(self.old) != set(ALL_JOINTS) or set(self.bus.motors) != set(ALL_JOINTS):
            raise ValueError("Expected the six SO-101 motors")
        self.calibration_type = MotorCalibration
        self.midpoint = None
        self.minimum = None
        self.maximum = None
        self.candidate = None

    def raw(self):
        return self.bus.sync_read("Present_Position", normalize=False, num_retry=2)

    def _actual(self, positions):
        # Feetech reports Actual_Position - Homing_Offset. Recover physical
        # encoder readings before calculating a new offset, without writing it.
        return {name: int(positions[name]) + self.old[name].homing_offset for name in ALL_JOINTS}

    def capture_midpoint(self):
        self.midpoint = self._actual(self.raw())
        self.minimum = self.maximum = self.candidate = None
        return self.midpoint

    def begin_ranges(self):
        if self.midpoint is None:
            raise ValueError("Capture the physical midpoint first")
        current = self._actual(self.raw())
        self.minimum = current.copy()
        self.maximum = current.copy()

    def sample(self):
        actual = self._actual(self.raw())
        if self.minimum is not None:
            for name in ALL_JOINTS:
                if name != "wrist_roll":
                    self.minimum[name] = min(self.minimum[name], actual[name])
                    self.maximum[name] = max(self.maximum[name], actual[name])
        return actual

    def observe(self):
        """Read physical positions without extending the recorded range."""
        return self._actual(self.raw())

    def finish_ranges(self):
        if self.minimum is None:
            raise ValueError("Begin recording full safe motion ranges first")
        self.sample()
        candidate = {}
        for name in ALL_JOINTS:
            offset = self.midpoint[name] - HALF_TURN
            if name == "wrist_roll":
                low, high = 0, RESOLUTION
            else:
                low = self.minimum[name] - offset
                high = self.maximum[name] - offset
                midpoint = self.midpoint[name] - offset
                if high - low < MIN_TRAVEL_TICKS or not low < midpoint < high:
                    raise ValueError(f"{name}: insufficient range or midpoint outside recorded travel; continue recording")
                if low < 0 or high > RESOLUTION:
                    raise ValueError(f"{name}: recorded range crosses encoder limits; inspect motor")
                if name != "gripper" and abs((low + high) / 2 - HALF_TURN) > 250:
                    raise ValueError(f"{name}: physical midpoint far from recorded travel center; recapture midpoint")
            candidate[name] = self.calibration_type(
                id=self.bus.motors[name].id, drive_mode=0, homing_offset=offset,
                range_min=low, range_max=high)
        self.candidate = candidate
        return candidate

    def preview(self, actual=None):
        """Display rough joint pose until ranges are captured; never command goals."""
        if actual is None:
            actual = self.observe()
        state = {}
        for name in ALL_JOINTS:
            cal = self.candidate[name] if self.candidate else self.old[name]
            position = actual[name] - cal.homing_offset
            if name == "gripper":
                if self.candidate is not None and cal.range_max > cal.range_min:
                    state[name] = max(0.0, min(100.0, (position - cal.range_min) * 100 /
                                                      (cal.range_max - cal.range_min)))
                elif self.old[name].range_max > self.old[name].range_min:
                    state[name] = max(0.0, min(100.0, (position - cal.range_min) * 100 /
                                                      (cal.range_max - cal.range_min)))
                else:
                    state[name] = 50.0
            else:
                if self.candidate is None and self.midpoint is not None:
                    state[name] = (actual[name] - self.midpoint[name]) * 360 / RESOLUTION
                elif self.candidate is None and cal.range_max <= cal.range_min:
                    state[name] = 0.0
                else:
                    state[name] = (position - (cal.range_min + cal.range_max) / 2) * 360 / RESOLUTION
        return state

    def commit(self, config_path):
        if self.candidate is None:
            raise ValueError("Finish and review complete ranges before saving")
        if any(self.bus.read("Torque_Enable", name, normalize=False) for name in ALL_JOINTS):
            raise RuntimeError("Torque must be OFF for all motors")
        station = Path(config_path)
        config = yaml.safe_load(station.read_text())
        if not isinstance(config, dict):
            raise ValueError("Invalid station configuration")
        path = Path(self.robot.calibration_fpath)
        if path.exists():
            backup = path.with_name(path.name + ".before-full-calibration.bak")
            if backup.exists():
                raise FileExistsError(f"Preserve previous backup before retrying: {backup}")
        else:
            backup = None
        camera_file = station.parent / "calibration.yaml"
        camera_data = None
        if camera_file.exists():
            camera_backup = camera_file.with_name(camera_file.name + ".before-full-calibration.bak")
            if camera_backup.exists():
                raise FileExistsError(f"Preserve previous camera backup: {camera_backup}")
            camera_data = yaml.safe_load(camera_file.read_text())
            if not isinstance(camera_data, dict):
                raise ValueError("Cannot invalidate camera calibration")
        if backup:
            shutil.copy2(path, backup)
        if camera_data is not None:
            shutil.copy2(camera_file, camera_backup)
            camera_data["accepted"] = False
            camera_file.write_text(yaml.safe_dump(camera_data, sort_keys=False))
        config["fk_tcp_verified"] = False
        config["base_plane_verified"] = False
        config["joint_zero_offsets_deg"] = {}
        config["gripper_zero_offset_percent"] = 0.0
        station.write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            self.bus.write_calibration(self.candidate, cache=False)
            readback = self.bus.read_calibration()
            if any(readback[name] != self.candidate[name] for name in ALL_JOINTS):
                raise RuntimeError("Motor calibration readback differs")
        except Exception as exc:
            try:
                self.bus.write_calibration(self.old, cache=False)
            except Exception as rollback_exc:
                raise RuntimeError("Calibration AND rollback failed; do not use arm") from rollback_exc
            raise RuntimeError("Calibration failed; old motor settings restored; verify before use") from exc
        self.robot.calibration = {name: replace(cal) for name, cal in self.candidate.items()}
        self.bus.calibration = self.robot.calibration
        try:
            self.robot._save_calibration()
        except Exception as exc:
            raise RuntimeError("Motor registers updated but LeRobot file save failed; "
                               "do not reconnect until repaired") from exc
        config["motor_calibration_file"] = str(path)
        config["motor_homing_offsets_ticks"] = {
            name: cal.homing_offset for name, cal in self.candidate.items()}
        station.write_text(yaml.safe_dump(config, sort_keys=False))
        return backup