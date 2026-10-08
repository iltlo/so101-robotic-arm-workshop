"""Calibrate SO-101 midpoint and full joint ranges with a live Rerun twin.

Support the arm: torque is disabled; no position goals are sent. Use Enter to
capture the physical midpoint, then Enter to start/stop range recording, S/Y
to commit, or Q to abort. This works without an existing calibration file.
"""

import argparse
import time

from so101_control.config import load_station
from so101_control.full_motor_calibration import FullMotorCalibration
from so101_control.kinematics import Kinematics
from so101_control.torque import keyboard_requests
from so101_control.twin import Twin


def run_calibration(robot, capture, kin, twin, config_path, keys):
    robot.bus.disable_torque()
    print("Torque OFF: support the arm; this program sends no position goals.")
    print("Move arm and gripper to PHYSICAL midpoints, then Enter. "
          "Enter again starts recording; sweep all except wrist_roll through full SAFE ranges.")
    stage = "middle"
    while True:
        actual = capture.sample() if stage == "ranges" else capture.observe()
        state = capture.preview(actual)
        twin.update(state, ee_pose=kin.fk(state))
        event = keys.consume()
        if event == "q":
            print("Aborted. No new calibration saved; support arm on disconnect.")
            return
        if stage == "middle" and event == "enter":
            print("Captured midpoint (physical ticks):", capture.capture_midpoint())
            print("Enter to start range recording. Q aborts.")
            stage = "ready"
        elif stage == "ready" and event == "enter":
            capture.begin_ranges()
            print("Recording: slowly sweep all joints except wrist_roll through full safe range. "
                  "Enter stops; Q aborts.")
            stage = "ranges"
        elif stage == "ranges" and event == "enter":
            try:
                candidate = capture.finish_ranges()
            except ValueError as exc:
                print(f"Cannot finish: {exc}")
            else:
                for name, cal in candidate.items():
                    print(f"{name}: offset={cal.homing_offset}, min={cal.range_min}, max={cal.range_max}")
                print("Review physical alignment in Rerun. Press S then Y to write, or Q to abort.")
                stage = "review"
        elif stage == "review" and event == "s":
            print("FINAL CONFIRMATION: press Y to overwrite motor registers and LeRobot JSON; "
                  "any other key cancels save.")
            stage = "confirm"
        elif stage == "confirm" and event is not None:
            if event == "y":
                backup = capture.commit(config_path)
                print(f"Calibration saved: {robot.calibration_fpath}; previous backup: {backup}")
                print("Recheck FK and re-run hand-eye before using physical motion.")
                return
            print("Save cancelled; S to try again or Q to abort.")
            stage = "review"
        time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true", help="connect motors and disable torque")
    args = parser.parse_args(argv)
    if not args.enable_live:
        parser.error("Full motor calibration requires --enable-live; support the arm")
    config = load_station(args.config)
    if not config.get("port") or not config.get("arm_id"):
        raise ValueError("Set port and arm_id in station.yaml")

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=config["port"], id=config["arm_id"], use_degrees=True))
    try:
        robot.bus.connect()  # no calibrated robot.connect() requirement
        robot.bus.disable_torque()
        capture = FullMotorCalibration(robot)
        kin = Kinematics(config["urdf"])
        twin = Twin(config["urdf"])
        with keyboard_requests(keys=("enter", "s", "y", "q")) as keys:
            try:
                run_calibration(robot, capture, kin, twin, args.config, keys)
            except KeyboardInterrupt:
                print("Interrupted. Pending calibration not saved; support arm while disconnecting.")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()