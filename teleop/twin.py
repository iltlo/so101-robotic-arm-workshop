"""Read-only Rerun twin using the same SO-101 URDF as FK and IK.

Run without --enable-live for an offline demonstration or save a multi-pose
recording with --recording. Torque toggling is separately gated and may allow
the arm to drop; it is not an emergency stop.
"""

import argparse
import time

from so101_control.config import load_station
from so101_control.kinematics import Kinematics
from so101_control.motion import joint_state
from so101_control.session import station_robot
from so101_control.twin import Twin


def _live_loop(robot, kin, twin, keys, *, toggle_torque, interval_s=0.1):
    """Never stop reading feedback when torque is disabled."""
    from so101_control.torque import ArmTorque

    torque = ArmTorque(robot)
    while True:
        if toggle_torque and keys.consume():
            torque.toggle()
        state = joint_state(robot)
        twin.update(state, ee_pose=kin.fk(state))
        time.sleep(interval_s)


def _offline_demo(twin, kin, robot):
    # Distinct synthetic poses make an offline recording useful for checking
    # moving meshes, the display-only jaw, and FK EE axes without serial IO.
    for joint, value in ((None, 0), ("shoulder_pan", 25), ("elbow_flex", -20),
                         ("gripper", 100), ("gripper", 0)):
        if joint is not None:
            robot.positions[f"{joint}.pos"] = float(value)
        state = joint_state(robot)
        twin.update(state, ee_pose=kin.fk(state))
    twin.rr.get_global_data_recording().flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true", help="read follower feedback (does not command motion)")
    parser.add_argument("--recording", help="save .rrd data and matching .rbl blueprint")
    parser.add_argument("--allow-torque-toggle", action="store_true",
                        help="enable T key; disabling torque may drop the arm")
    args = parser.parse_args(argv)
    if args.allow_torque_toggle and not args.enable_live:
        parser.error("--allow-torque-toggle requires --enable-live")

    config = load_station(args.config)
    kin = Kinematics(config["urdf"])
    twin = Twin(config["urdf"], recording=args.recording)
    with station_robot(config, args.enable_live) as robot:
        if not args.enable_live:
            _offline_demo(twin, kin, robot)
            if not args.recording:
                print("Offline three-pose demonstration logged; close the viewer to exit.")
            return

        from so101_control.torque import torque_toggle_listener

        print("Read-only live twin; Ctrl-C exits. T is disabled unless --allow-torque-toggle was supplied.")
        if args.allow_torque_toggle:
            print("T toggles torque immediately; torque OFF may drop the arm. Support it and keep physical power cutoff accessible.")
        with torque_toggle_listener(args.allow_torque_toggle) as keys:
            try:
                _live_loop(robot, kin, twin, keys, toggle_torque=args.allow_torque_toggle)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()