"""Home, then visit three points. Coordinates are metres in the base frame.

The arm folds to the rest pose before the script ends.

    uv run python scripts/arm_move.py --mock
    uv run python scripts/arm_move.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.interrupts import ignore_sigint
from so101_utils.views import watch_arm

# A short triangle in front of the home pose.
POINTS = (
    (0.20, 0.00, 0.10),
    (0.22, 0.06, 0.10),
    (0.22, -0.06, 0.10),
)
# Pause at each point before printing the pose, then keep it in view.
POINT_PAUSE_S = 2.0
POSE_PAUSE_S = 1.0


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Homing (all arm joints to 0°).")
        arm.home()
        try:
            for point in POINTS:
                print(f"Moving to {point}")
                arm.move_to(*point)
                time.sleep(POINT_PAUSE_S)
                print("Current pose:", arm.get_pose())
                time.sleep(POSE_PAUSE_S)
        finally:
            ignore_sigint()
            print("Going to rest.")
            arm.rest()
