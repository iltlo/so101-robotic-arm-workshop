"""View camera-to-AprilTag optical pose in Rerun; camera only, no robot motion."""

import argparse

from so101_control.config import load_station
from so101_control.session import rgb_camera
from so101_control.tag_view import TagPoseView


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size-m", type=float, default=0.04,
                        help="measured black-square AprilTag edge in metres (default: 0.04)")
    args = parser.parse_args(argv)
    config = load_station(args.config)
    with rgb_camera(config) as cam:
        view = TagPoseView(cam.intrinsics, tag_id=args.tag_id, tag_size_m=args.tag_size_m)
        print("Camera-only AprilTag pose in Rerun. No robot connection or motion. Ctrl-C to stop.")
        try:
            while True:
                view.update(cam.read().rgb_bgr)
        except KeyboardInterrupt:
            print("Stopped camera-only viewer.")


if __name__ == "__main__":
    main()