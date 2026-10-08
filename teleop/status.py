"""Read five arm joints and gripper, or show an offline zero-state sample."""

import argparse

from so101_control.config import load_station
from so101_control.motion import joint_state
from so101_control.session import station_robot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true")
    args = parser.parse_args(argv)
    with station_robot(load_station(args.config), args.enable_live) as robot:
        print(joint_state(robot))


if __name__ == "__main__":
    main()