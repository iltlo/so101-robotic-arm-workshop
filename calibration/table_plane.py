"""Calibrate the table plane from a flat tag using an accepted RGB camera calibration."""

import argparse

from so101_control.calibration import table_plane
from so101_control.config import load_station
from so101_control.session import capture_tag, rgb_camera
from so101_control.station_calibration import load_calibration, matches_camera, save_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    args = parser.parse_args(argv)

    config = load_station(args.config)
    data = load_calibration(args.config)
    with rgb_camera(config) as cam:
        matches_camera(data, cam.intrinsics, config)
        if input("Move the same ID 0 tag FLAT to table; remove EE tag from view. Ready? Type YES to continue: ").strip() != "YES":
            return
        pose = capture_tag(cam)
        plane = table_plane(data["base_T_camera"], pose)
        print("Base frame plane point/normal:", plane)
        if input("Confirm table is flat and accept this plane? Type YES to continue: ").strip() == "YES":
            data["plane"] = {key: value.tolist() for key, value in plane.items()}
            save_calibration(args.config, data)


if __name__ == "__main__":
    main()