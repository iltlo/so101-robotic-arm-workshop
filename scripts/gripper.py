"""Open and close the gripper. 0% is closed. Opening commands 50%.

    uv run python scripts/gripper.py --mock
    uv run python scripts/gripper.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.views import watch_arm

# Time to see the open jaw before it closes.
VIEW_PAUSE_S = 0.5


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Opening.")
        arm.release()
        time.sleep(VIEW_PAUSE_S)
        print("Closing.")
        arm.grip()
        print(f"gripper = {arm.get_joints()['gripper']:.0f}%")
