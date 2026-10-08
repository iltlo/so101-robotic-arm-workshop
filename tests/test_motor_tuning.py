"""Station-side tuning never changes motor registers or LeRobot's file."""

import numpy as np
import pytest
import yaml

from so101_control.config import ALL_JOINTS
from so101_control.joint_offsets import StationRobot, station_offsets
from so101_control.motor_tuning import MotorZeroTuner


class Robot:
    is_calibrated = True

    def __init__(self):
        self.observed = {f"{name}.pos": 0.0 for name in ALL_JOINTS}
        self.observed["elbow_flex.pos"] = 97.0  # outside a recorded range, still visible
        self.actions = []

    def get_observation(self):
        return self.observed.copy()

    def send_action(self, action):
        self.actions.append(action.copy())
        return action.copy()

    def disconnect(self):
        pass


def test_preview_uses_normalized_feedback_even_if_motor_is_outside_recorded_range():
    robot = Robot()
    tuner = MotorZeroTuner(robot, {"joint_zero_offsets_deg": {"elbow_flex": -3.0}})
    assert tuner.preview()["elbow_flex"] == 94.0
    tuner.select("3")
    assert tuner.adjust(1) == -2.5
    assert tuner.preview()["elbow_flex"] == 94.5
    assert robot.actions == []
    assert tuner.select("0") is None
    with pytest.raises(ValueError):
        tuner.adjust(1)
    tuner.select("6")
    assert tuner.adjust(-1) == -1.0
    with pytest.raises(ValueError, match="Maximum"):
        for _ in range(11):
            tuner.adjust(-1)


def test_tuner_saves_only_station_yaml_and_invalidates_camera(tmp_path):
    robot = Robot()
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump({"fk_tcp_verified": True, "base_plane_verified": True,
                                       "joint_zero_offsets_deg": {"shoulder_pan": 1}}))
    motor = tmp_path / "lerobot.json"
    motor.write_text("must stay unchanged")
    camera = tmp_path / "calibration.yaml"
    camera.write_text(yaml.safe_dump({"accepted": True, "base_T_camera": np.eye(4).tolist()}))
    tuner = MotorZeroTuner(robot, yaml.safe_load(station.read_text()))
    tuner.select("3")
    tuner.adjust(1)
    backup = tuner.commit(station)
    saved = yaml.safe_load(station.read_text())
    assert saved["joint_zero_offsets_deg"]["elbow_flex"] == 0.5
    assert saved["joint_zero_offsets_deg"]["shoulder_pan"] == 1.0
    assert saved["fk_tcp_verified"] is False and saved["base_plane_verified"] is False
    assert yaml.safe_load(backup.read_text())["fk_tcp_verified"] is True
    assert yaml.safe_load(camera.read_text())["accepted"] is False
    assert motor.read_text() == "must stay unchanged"
    assert robot.actions == []


def test_repeated_saves_replace_fixed_station_camera_backups(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump({"joint_zero_offsets_deg": {"elbow_flex": -7.0},
                                       "fk_tcp_verified": True}))
    camera = tmp_path / "calibration.yaml"
    camera.write_text(yaml.safe_dump({"accepted": True, "base_T_camera": np.eye(4).tolist()}))
    old_station = tmp_path / "station.yaml.before-zero-tuning.bak"
    old_camera = tmp_path / "calibration.yaml.before-zero-tuning.bak"
    old_station.write_text("historic station")
    old_camera.write_text("historic camera")
    robot = Robot()
    first = MotorZeroTuner(robot, yaml.safe_load(station.read_text()))
    first.select("2")
    first.adjust(-1)
    first_backup = first.commit(station)
    assert first_backup == old_station
    assert yaml.safe_load(first_backup.read_text())["joint_zero_offsets_deg"]["elbow_flex"] == -7.0
    camera.write_text(yaml.safe_dump({"accepted": True, "base_T_camera": np.eye(4).tolist()}))

    second = MotorZeroTuner(robot, yaml.safe_load(station.read_text()))
    second.select("3")
    second.adjust(1)
    second_backup = second.commit(station)
    assert second_backup == first_backup == old_station
    assert yaml.safe_load(second_backup.read_text())["joint_zero_offsets_deg"]["shoulder_lift"] == -0.5
    assert yaml.safe_load(old_camera.read_text())["accepted"] is True
    assert list(tmp_path.glob("*.before-zero-tuning.*.bak")) == []
    assert yaml.safe_load(station.read_text())["joint_zero_offsets_deg"]["elbow_flex"] == -6.5
    assert yaml.safe_load(camera.read_text())["accepted"] is False


def test_station_robot_feedback_and_command_are_inverse():
    raw = Robot()
    config = {"joint_zero_offsets_deg": {"elbow_flex": 4.0}, "gripper_zero_offset_percent": 2.0}
    adapted = StationRobot(raw, config)
    assert adapted.get_observation()["elbow_flex.pos"] == 101.0
    assert adapted.get_observation()["gripper.pos"] == 2.0
    returned = adapted.send_action({"elbow_flex.pos": 5.0, "gripper.pos": 10.0})
    assert raw.actions == [{"elbow_flex.pos": 1.0, "gripper.pos": 8.0}]
    assert returned == {"elbow_flex.pos": 5.0, "gripper.pos": 10.0}
    with pytest.raises(ValueError):
        station_offsets({"joint_zero_offsets_deg": {"elbow_flex": 31}})
