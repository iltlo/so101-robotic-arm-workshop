"""Open and close the gripper. 0% is closed, 100% is open.

    uv run python scripts/gripper.py --mock
    uv run python scripts/gripper.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        print("Opening.")
        arm.release()
        time.sleep(0.5)
        print("Closing.")
        arm.grip()
        print(f"gripper = {arm.get_joints()['gripper']:.0f}%")
