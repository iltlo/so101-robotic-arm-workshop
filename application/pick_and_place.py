"""Pick at one point and place at another. Edit the two constants first.

Coordinates are metres in the base frame. Ctrl-C is not an e-stop; keep the
area clear.

    uv run python application/pick_and_place.py --mock
    uv run python application/pick_and_place.py --live
"""

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.views import watch_arm

# Metres.
PICK = (0.22, 0.06, 0.02)
PLACE = (0.22, -0.06, 0.02)
CLEARANCE = 0.06


def jump(arm, x, y, z) -> None:
    arm.move_to(x, y, z + CLEARANCE, downward=True)
    arm.move_to(x, y, z, downward=True)


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Homing.")
        arm.home()
        arm.release()
        print(f"Pick {PICK}")
        jump(arm, *PICK)
        arm.grip()
        arm.move_to(PICK[0], PICK[1], PICK[2] + CLEARANCE, downward=True)
        print(f"Place {PLACE}")
        jump(arm, *PLACE)
        arm.release()
        arm.home()
        print("Done.", arm.get_pose())
