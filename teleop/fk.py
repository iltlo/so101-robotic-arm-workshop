"""Read the follower and show its gripper_frame_link pose in the base frame (metres)."""

import argparse

from so101_control.config import load_station
from so101_control.kinematics import Kinematics
from so101_control.motion import joint_state
from so101_control.session import station_robot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true")
    args = parser.parse_args(argv)
    config = load_station(args.config)
    kin = Kinematics(config["urdf"])
    with station_robot(config, args.enable_live) as robot:
        print(kin.fk(joint_state(robot)))


if __name__ == "__main__":
    main()