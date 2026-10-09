"""Step one arm joint by at most 2 degrees.

Type `shoulder_pan 1` or `elbow_flex -1`. `q` quits.
Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
"""

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.twin import watch_arm


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        if args.mock:
            watch_arm(arm, load_robot()["urdf"])
            print("Mock arm is in the Rerun window.")
        print(arm.get_joints())
        while True:
            command = input("joint degrees> ").strip()
            if command.lower() in {"q", "quit"}:
                break
            try:
                name, amount = command.split()
                delta = float(amount)
                if name not in arm.get_joints() or name == "gripper" or not 0 < abs(delta) <= 2:
                    raise ValueError("Use <arm joint> <delta> with 0 < |delta| <= 2")
                arm.move_joint(name, delta)
            except (ValueError, RuntimeError) as exc:
                print(exc)
                continue
            print(arm.get_joints())
