"""Toggle the gripper until Ctrl-C. 0% is closed; opening commands 50%.

    uv run python scripts/gripper.py --mock
    uv run python scripts/gripper.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.interrupts import ignore_sigint
from so101_utils.views import watch_arm

# Time each jaw position stays before the next one.
TOGGLE_DELAY_S = 1.0


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Toggling the gripper. Ctrl-C stops.")
        try:
            while True:
                print("Gripper closed (0%).")
                arm.grip()
                time.sleep(TOGGLE_DELAY_S)
                print("Gripper open (50%).")
                arm.release()
                time.sleep(TOGGLE_DELAY_S)
        except KeyboardInterrupt:
            print("\nStopping gripper toggle.")
        finally:
            ignore_sigint()
            arm.release()
