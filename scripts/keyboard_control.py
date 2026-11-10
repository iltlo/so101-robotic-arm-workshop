"""Cartesian keyboard control. Steps are metres. Roll, pitch, and yaw are base-frame degrees.

    WASD or arrows : x / y
    Q / E          : z down / up
    Z / C          : rotate the gripper
    F / V          : pitch
    H              : home
    R              : rest
    P              : print pose
    Space          : grip / release
    + / -          : larger / smaller step
    Esc            : quit

    uv run python scripts/keyboard_control.py --mock
    uv run python scripts/keyboard_control.py --live
"""

from pynput import keyboard

from so101_control import SO101
from so101_control.teleop import TELEOP_SPEED_DEG_S, Teleop, run
from so101_utils.config import arm_parser, load_robot
from so101_utils.views import watch_arm

if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        arm.set_speed(TELEOP_SPEED_DEG_S)
        twin = watch_arm(arm, load_robot()["urdf"])
        print("Arm is in the Rerun window.")
        teleop = Teleop(arm)
        print(f"Step {teleop.step * 1000:.0f} mm, joint speed {TELEOP_SPEED_DEG_S:.0f} deg/s.")
        print(teleop.arm.get_pose())
        print("Keyboard control. Esc quits. Keep clear of the arm.")

        with keyboard.Listener(on_press=teleop.on_press, on_release=teleop.on_release) as listener:
            if not listener.is_alive():
                raise RuntimeError("Keyboard listener did not start. An X11 session is required.")
            try:
                run(teleop, listener.is_alive, arm.refresh)
            except KeyboardInterrupt:
                print("Stopped.")
