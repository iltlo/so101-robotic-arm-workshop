"""Find the follower USB port by disconnecting its adapter; never move the arm."""

import argparse

from so101_control.ports import find_port


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    args = parser.parse_args(argv)
    return find_port(args.config)


if __name__ == "__main__":
    main()