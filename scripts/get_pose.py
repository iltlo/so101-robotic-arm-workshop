"""Print the TCP pose. Metres and degrees, robot base frame.

    uv run python scripts/get_pose.py --mock
    uv run python scripts/get_pose.py --live
"""

from so101_control import SO101
from so101_utils.config import arm_parser


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        print(arm.get_pose())
        print(arm.get_joints())
