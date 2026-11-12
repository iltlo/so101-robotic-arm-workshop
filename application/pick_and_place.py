"""Pick at one point and place at another. Edit the settings first.

Coordinates are metres in the base frame. The arm folds to the rest pose before
the script ends. Ctrl-C is not an e-stop; keep the area clear.

    uv run python application/pick_and_place.py --mock
    uv run python application/pick_and_place.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.interrupts import ignore_sigint
from so101_utils.views import watch_arm

# ============== CONFIGURATION CONSTANTS ==============

# Metres to lift above the current and target points during a jump move.
JUMP_HEIGHT_M = 0.05

# Metres in the base frame.
PICK_LOCATION = (0.22, 0.06, 0.02)
PLACE_LOCATION = (0.22, -0.06, 0.02)

# Seconds.
MOVE_DELAY_S = 0.5
GRIP_DELAY_S = 1.0
RELEASE_DELAY_S = 1.0

# ============== END CONFIGURATION ==============


def jump_to(arm, x, y, z) -> None:
    """Lift from the current pose, travel above the target, then descend.

    The lift keeps the pose it starts from: at home the arm is reaching far
    enough that the same point with the fingers down is out of range.
    """
    pose = arm.get_pose()
    lift_z = max(pose.z, z) + JUMP_HEIGHT_M
    for target in ((pose.x, pose.y, lift_z), (x, y, lift_z)):
        print(f"  Moving to {target}")
        arm.move_to(*target)
        time.sleep(MOVE_DELAY_S)
    print(f"  Moving down to {(x, y, z)}")
    arm.move_to(x, y, z, downward=True)
    time.sleep(MOVE_DELAY_S)


def pick_and_place(arm) -> None:
    print("Opening gripper.")
    arm.release()
    time.sleep(GRIP_DELAY_S)
    print(f"Moving to pick location {PICK_LOCATION}")
    jump_to(arm, *PICK_LOCATION)
    print("Gripping.")
    arm.grip()
    time.sleep(GRIP_DELAY_S)
    print(f"Moving to place location {PLACE_LOCATION}")
    jump_to(arm, *PLACE_LOCATION)
    print("Releasing.")
    arm.release()
    time.sleep(RELEASE_DELAY_S)


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Homing.")
        arm.home()
        try:
            pick_and_place(arm)
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
        finally:
            ignore_sigint()
            print("Going to rest, gripper open.")
            arm.rest()
            arm.release()
            print("Done.", arm.get_pose())
