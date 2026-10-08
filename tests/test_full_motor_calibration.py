"""Full calibration feedback and persistence tests do not access a serial port."""

from dataclasses import dataclass
import json

import numpy as np
import pytest
import yaml

from so101_control.config import ALL_JOINTS
from so101_control.full_motor_calibration import FullMotorCalibration


@dataclass
class Calibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class Bus:
    def __init__(self):
        self.motors = {name: type("Motor", (), {"id": number})() for number, name in enumerate(ALL_JOINTS, 1)}
        self.stored = {name: Calibration(i, 0, 100, 500, 3500)
                       for i, name in enumerate(ALL_JOINTS, 1)}
        self.samples = []
        self.writes = []
        self.torque = 0

    def sync_read(self, register, *, normalize, num_retry):
        assert register == "Present_Position" and not normalize
        return self.samples.pop(0)

    def read_calibration(self):
        return {n: Calibration(**vars(c)) for n, c in self.stored.items()}

    def read(self, register, name, *, normalize):
        assert register == "Torque_Enable" and not normalize
        return self.torque

    def write_calibration(self, candidate, *, cache):
        assert cache is False
        self.writes.append(candidate)
        self.stored = {n: Calibration(**vars(c)) for n, c in candidate.items()}


class Robot:
    def __init__(self, path):
        self.bus = Bus()
        self.calibration_fpath = path
        self.calibration = {}

    def _save_calibration(self):
        self.calibration_fpath.write_text(json.dumps({n: vars(c) for n, c in self.calibration.items()}))


def observe(robot, **overrides):
    data = dict.fromkeys(ALL_JOINTS, 1947)  # +100 ticks = 2047 physical midpoint
    data.update(overrides)
    robot.bus.samples.append(data)


def test_full_midpoint_range_and_commit(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.motors.MotorCalibration", Calibration)
    robot = Robot(tmp_path / "follower.json")
    robot.calibration_fpath.write_text("previous")
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump({"fk_tcp_verified": True, "base_plane_verified": True}))
    camera = tmp_path / "calibration.yaml"
    camera.write_text(yaml.safe_dump({"accepted": True, "base_T_camera": np.eye(4).tolist()}))
    capture = FullMotorCalibration(robot)
    observe(robot)
    capture.capture_midpoint()
    observe(robot)
    capture.begin_ranges()
    observe(robot, elbow_flex=1100)
    capture.sample()
    observe(robot, elbow_flex=2700)
    capture.sample()
    observe(robot)
    # Give every non-roll motor enough travel around the captured middle.
    capture.minimum.update({n: 1100 for n in ALL_JOINTS if n != "wrist_roll"})
    capture.maximum.update({n: 2700 for n in ALL_JOINTS if n != "wrist_roll"})
    candidate = capture.finish_ranges()
    assert candidate["elbow_flex"].homing_offset == 0
    assert candidate["elbow_flex"].range_min == 1100
    assert candidate["elbow_flex"].range_max == 2700
    assert (candidate["wrist_roll"].range_min, candidate["wrist_roll"].range_max) == (0, 4095)
    assert not robot.bus.writes
    robot.bus.torque = 1
    with pytest.raises(RuntimeError, match="Torque"):
        capture.commit(station)
    assert not robot.bus.writes
    robot.bus.torque = 0
    backup = capture.commit(station)
    assert backup.read_text() == "previous"
    assert robot.bus.writes
    assert json.loads(robot.calibration_fpath.read_text())["elbow_flex"]["range_max"] == 2700
    assert yaml.safe_load(camera.read_text())["accepted"] is False
    assert yaml.safe_load(station.read_text())["fk_tcp_verified"] is False


def test_incomplete_ranges_never_write(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.motors.MotorCalibration", Calibration)
    robot = Robot(tmp_path / "follower.json")
    capture = FullMotorCalibration(robot)
    observe(robot)
    capture.capture_midpoint()
    observe(robot)
    capture.begin_ranges()
    observe(robot, elbow_flex=1949)
    with pytest.raises(ValueError, match="insufficient range"):
        capture.finish_ranges()
    assert capture.candidate is None and not robot.bus.writes


def test_midpoint_far_from_recorded_center_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.motors.MotorCalibration", Calibration)
    robot = Robot(tmp_path / "follower.json")
    capture = FullMotorCalibration(robot)
    observe(robot)
    capture.capture_midpoint()
    observe(robot)
    capture.begin_ranges()
    capture.minimum.update({n: 1100 for n in ALL_JOINTS if n != "wrist_roll"})
    capture.maximum.update({n: 2700 for n in ALL_JOINTS if n != "wrist_roll"})
    capture.minimum["elbow_flex"] = 500
    capture.maximum["elbow_flex"] = 2200
    observe(robot)
    with pytest.raises(ValueError, match="midpoint far"):
        capture.finish_ranges()
    assert not robot.bus.writes