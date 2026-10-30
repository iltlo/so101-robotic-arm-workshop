"""Home, then visit three points. Coordinates are metres in the base frame.

    uv run python scripts/arm_move.py --mock
    uv run python scripts/arm_move.py --live
"""

import time

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.views import watch_arm

# A short triangle in front of the home pose.
POINTS = (
    (0.20, 0.00, 0.10),
    (0.22, 0.06, 0.10),
    (0.22, -0.06, 0.10),
)
VIEW_PAUSE_S = 0.5


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print("Homing (all arm joints to 0°).")
        arm.home()
        for point in POINTS:
            print(f"Moving to {point}")
            arm.move_to(*point)
            print(arm.get_pose())
            time.sleep(VIEW_PAUSE_S)
