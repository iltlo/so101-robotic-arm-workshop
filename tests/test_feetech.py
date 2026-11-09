"""Tick conversion matches LeRobot use_degrees and the 0–100 gripper."""

import threading
import time

from so101_control.estop import trip
from so101_control.feetech import (
    MAX_TICK,
    MOTORS,
    FeetechBus,
    MemoryLink,
    decode_sign_magnitude,
    degrees_to_ticks,
    percent_to_ticks,
    ticks_to_degrees,
    ticks_to_percent,
)


def test_degrees_are_zero_at_the_range_midpoint():
    assert ticks_to_degrees(2000, 1000, 3000) == 0
    assert degrees_to_ticks(0, 1000, 3000) == 2000
    ticks = degrees_to_ticks(90, 1000, 3000)
    assert ticks_to_degrees(ticks, 1000, 3000) == pytest_approx_degrees(90)


def pytest_approx_degrees(value):
    import pytest
    return pytest.approx(value, abs=0.2)


def test_gripper_percent_flips_when_drive_mode_is_set():
    assert ticks_to_percent(1000, 1000, 3000, 0) == 0
    assert ticks_to_percent(3000, 1000, 3000, 0) == 100
    assert percent_to_ticks(0, 1000, 3000, 0) == 1000
    assert percent_to_ticks(100, 1000, 3000, 0) == 3000
    assert ticks_to_percent(1000, 1000, 3000, 1) == 100
    assert percent_to_ticks(0, 1000, 3000, 1) == 3000


def _bus():
    link = MemoryLink()
    bus = FeetechBus("mem", link=link)
    link.open_port()
    bus._open = True
    bus.calibration = {
        name: {"id": motor_id, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
        for name, motor_id, _mode in MOTORS
    }
    for _name, motor_id, _mode in MOTORS:
        link.values[(motor_id, 56)] = 2000
    return bus, link


def test_goal_round_trip_uses_degrees_and_percent():
    bus, _link = _bus()
    bus.sync_write("Goal_Position", {"shoulder_pan": 0.0, "gripper": 100.0})
    present = bus.sync_read("Present_Position", ["shoulder_pan", "gripper"])
    # The memory link stores the goal in the goal register, not present.
    # Read the goal back through the same conversion by copying it to present.
    goal_address = 42
    present_address = 56
    for _name, motor_id, _mode in MOTORS:
        if (motor_id, goal_address) in _link.values:
            _link.values[(motor_id, present_address)] = _link.values[(motor_id, goal_address)]
    present = bus.sync_read("Present_Position")
    assert present["shoulder_pan"] == 0
    assert present["gripper"] == 100


def test_midpoint_writes_homing_from_the_current_count():
    bus, link = _bus()
    for _name, motor_id, _mode in MOTORS:
        link.values[(motor_id, 56)] = 3000
    offsets = bus.set_midpoint()
    assert offsets["shoulder_pan"] == 3000 - (MAX_TICK // 2)
    encoded = link.values[(1, 31)]
    assert decode_sign_magnitude(encoded, 11) == offsets["shoulder_pan"]
    assert bus.is_calibrated is False


def test_saved_calibration_loads_a_full_turn_wrist(tmp_path):
    import json

    bus, link = _bus()
    bus.calibration_path = tmp_path / "motors.json"
    stored = {
        name: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 10 * motor_id,
            "range_min": 0 if name == "wrist_roll" else 800,
            "range_max": MAX_TICK if name == "wrist_roll" else 3200,
        }
        for name, motor_id, _mode in MOTORS
    }
    bus.calibration_path.write_text(json.dumps(stored))
    bus._load_calibration(apply=True)
    assert bus.is_calibrated is True
    assert bus.calibration["shoulder_pan"]["range_min"] == 800
    assert bus.calibration["wrist_roll"]["range_max"] == MAX_TICK
    assert link.values[(1, 9)] == 800
    assert link.values[(1, 11)] == 3200
    encoded = link.values[(1, 31)]
    assert decode_sign_magnitude(encoded, 11) == 10


def test_range_capture_writes_the_swept_limits(tmp_path):
    bus, link = _bus()
    bus.calibration_path = tmp_path / "motors.json"
    for _name, motor_id, _mode in MOTORS:
        link.values[(motor_id, 56)] = 1500
    bus.start_range_capture()
    time.sleep(0.08)
    for _name, motor_id, _mode in MOTORS:
        link.values[(motor_id, 56)] = 2500
    time.sleep(0.12)
    saved = bus.stop_range_capture()
    assert saved["elbow_flex"]["range_min"] == 1500
    assert saved["elbow_flex"]["range_max"] == 2500
    assert saved["wrist_roll"]["range_min"] == 0
    assert saved["wrist_roll"]["range_max"] == MAX_TICK
    assert bus.is_calibrated is True
    assert bus.calibration_path.is_file()


def test_estop_is_visible_while_torque_off_waits_for_the_lock():
    event = threading.Event()
    lock = threading.Lock()
    lock.acquire()
    entered = threading.Event()

    def torque_off():
        entered.set()
        lock.acquire()
        lock.release()

    worker = threading.Thread(target=lambda: trip(event, torque_off))
    worker.start()
    assert entered.wait(1)
    time.sleep(0.05)
    assert event.is_set()
    assert worker.is_alive()
    lock.release()
    worker.join(1)
    assert not worker.is_alive()
