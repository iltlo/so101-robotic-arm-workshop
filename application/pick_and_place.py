"""Pick at one point and place at another. Edit the two constants first.

Coordinates are metres in the base frame. The arm lifts CLEARANCE metres
between the poses. Ctrl-C is not an e-stop; keep the area clear.

    uv run python application/pick_and_place.py --mock
    uv run python application/pick_and_place.py --live
"""

from so101_control import SO101
from so101_utils.config import arm_parser

# Metres. Raise z if the gripper would hit the table.
PICK = (0.22, 0.06, 0.02)
PLACE = (0.22, -0.06, 0.02)
CLEARANCE = 0.06


def jump(arm, x, y, z) -> None:
    arm.move_to(x, y, z + CLEARANCE)
    arm.move_to(x, y, z)


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        arm.set_speed(10)
        print("Homing.")
        arm.home()
        arm.release()
        print(f"Pick {PICK}")
        jump(arm, *PICK)
        arm.grip()
        arm.move_to(PICK[0], PICK[1], PICK[2] + CLEARANCE)
        print(f"Place {PLACE}")
        jump(arm, *PLACE)
        arm.release()
        arm.home()
        print("Done.", arm.get_pose())
