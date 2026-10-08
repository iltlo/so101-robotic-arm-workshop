"""Preview station-side joint-zero corrections in Rerun; save with S then Y.

The tool never sends joint goals or writes motor calibration. It disables torque
to allow hand-positioning; support the arm throughout. Run full motor calibration first.
"""

import argparse
import time

from so101_control.config import load_station
from so101_control.kinematics import Kinematics
from so101_control.motor_tuning import MotorZeroTuner
from so101_control.session import station_robot
from so101_control.torque import keyboard_requests
from so101_control.twin import Twin


def run_tuner(robot, tuner, kin, twin, config_path, keys):
    robot.bus.disable_torque()
    print("Torque OFF: support the arm. No motion goals will be sent.")
    print("1–5 arm joints, 6 gripper, 0 deselect; Up/Down ±0.5° (max ±30°), gripper ±1% (max ±10%).")
    print("S then Y saves station.yaml ONLY; LeRobot calibration and motor registers stay unchanged. Q discards.")
    confirming = False
    while True:
        state = tuner.preview()
        twin.update(state, ee_pose=kin.fk(state))
        event = keys.consume()
        if event is None:
            time.sleep(0.1)
            continue
        if event == "q":
            print("Stopped. Pending adjustments not saved." if tuner.has_changes else "Stopped.")
            return
        if confirming:
            confirming = False
            if event == "y":
                backup = tuner.commit(config_path)
                print(f"Saved station-side offsets; station backup: {backup}")
                print("FK/TCP and camera extrinsics must be checked again before live use.")
            else:
                print("Save cancelled.")
            continue
        if event == "s":
            confirming = True
            print("Confirm station.yaml update ONLY: press Y; any other key cancels.")
        elif event in ("0", "1", "2", "3", "4", "5", "6"):
            name = tuner.select(event)
            print(f"Selected: {name or 'none'}")
        elif event in ("up", "down"):
            try:
                offset = tuner.adjust(1 if event == "up" else -1)
            except ValueError as exc:
                print(exc)
            else:
                unit = "%" if tuner.selected == "gripper" else "deg"
                print(f"{tuner.selected}: station correction {offset:+g} {unit} (preview only)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true", help="connect follower; disables motor torque")
    args = parser.parse_args(argv)
    if not args.enable_live:
        parser.error("Motor zero tuning requires --enable-live; support the arm")
    config = load_station(args.config)
    with station_robot(config, True) as robot:
        tuner = MotorZeroTuner(robot.robot, config)
        kin = Kinematics(config["urdf"])
        twin = Twin(config["urdf"])
        with keyboard_requests(keys=("0", "1", "2", "3", "4", "5", "6", "up", "down", "s", "y", "q")) as keys:
            try:
                run_tuner(robot, tuner, kin, twin, args.config, keys)
            except KeyboardInterrupt:
                print("Interrupted. Pending offsets not saved; support arm while disconnecting.")


if __name__ == "__main__":
    main()