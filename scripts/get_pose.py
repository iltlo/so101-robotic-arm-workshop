"""Print the TCP pose until Ctrl-C. Metres and degrees, robot base frame.

Torque is off, so the arm can be posed by hand. Support it; it can fall.
Joint angles are printed to 0.1 degree. The same two lines are rewritten.

    uv run python scripts/get_pose.py --mock
    uv run python scripts/get_pose.py --live
"""

import time

from so101_control import SO101
from so101_control.kinematics import JOINTS
from so101_utils.config import arm_parser


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        arm.relax()
        print("Ctrl-C stops.")
        print()
        print()
        try:
            while True:
                state = arm.get_joints()
                angles = "  ".join(f"{name}={state[name]:.1f}" for name in JOINTS)
                print(
                    f"\033[2A\r{arm.get_pose()}\033[K\n"
                    f"{angles} deg  gripper={state['gripper']:.1f}%\033[K",
                    flush=True,
                )
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
