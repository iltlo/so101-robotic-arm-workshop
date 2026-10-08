"""Manually jog calibrated arm joints in bounded 2-degree increments."""

import argparse

from so101_control.config import JOINTS, load_station, validated_limits
from so101_control.motion import jog_command
from so101_control.session import station_robot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true")
    args = parser.parse_args(argv)
    config = load_station(args.config)
    limits = validated_limits(config) if args.enable_live else {joint: (-180, 180) for joint in JOINTS}
    with station_robot(config, args.enable_live) as robot:
        print("Type <joint> <delta_degrees> (max 2), or q to exit; keyboard is not an emergency stop.")
        while True:
            try:
                command = input("jog> ").strip()
            except EOFError:
                break
            if command.lower() == "q":
                break
            try:
                print(jog_command(robot, limits, command))
            except (ValueError, RuntimeError) as exc:
                print("Rejected:", exc)


if __name__ == "__main__":
    main()