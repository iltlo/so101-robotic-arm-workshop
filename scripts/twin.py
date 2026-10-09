"""Show the arm in Rerun.

Mock plays five poses and exits. --live follows the real joints and does not
move them, unless you pass --relax (T turns torque off; support the arm).

    uv run python scripts/twin.py --mock
    uv run python scripts/twin.py --live
"""

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.twin import Twin, follow, play_demo


if __name__ == "__main__":
    parser = arm_parser(__doc__)
    parser.add_argument("--relax", action="store_true", help="press T to toggle torque (live only)")
    parser.add_argument("--recording", help="save a .rrd instead of opening a window")
    args = parser.parse_args()
    if args.relax and args.mock:
        parser.error("--relax needs --live")
    twin = Twin(load_robot()["urdf"], recording=args.recording)
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            play_demo(arm, twin)
            print("Mock twin logged. Close the viewer when you are done.")
        else:
            follow(arm, twin, relax=args.relax)
