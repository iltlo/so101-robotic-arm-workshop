"""Cartesian keyboard control. Steps are metres; wrist roll is degrees.

    WASD or arrows : x / y
    Q / E          : z up / down
    Z / C          : wrist roll
    H              : home
    P              : print pose
    Space          : grip / release
    + / -          : larger / smaller step
    Esc            : quit

    uv run python scripts/keyboard_control.py --mock
    uv run python scripts/keyboard_control.py --live
"""

from pynput import keyboard

from so101_control import SO101
from so101_utils.config import arm_parser


class Teleop:
    def __init__(self, arm: SO101):
        self.arm = arm
        self.step = 0.01  # metres
        self.roll_step = 5.0  # degrees
        self.closed = False
        self.running = True

    def _nudge(self, dx=0.0, dy=0.0, dz=0.0, dr=0.0) -> None:
        pose = self.arm.get_pose()
        # r is wrist_roll, not the TCP euler angle.
        roll = self.arm.get_joints()["wrist_roll"] + dr
        try:
            self.arm.move_to(pose.x + dx, pose.y + dy, pose.z + dz, r=roll)
        except (ValueError, RuntimeError) as exc:
            print(exc)
            return
        print(self.arm.get_pose())

    def on_press(self, key):
        char = getattr(key, "char", None)
        if char:
            char = char.lower()
        if key == keyboard.Key.esc:
            self.running = False
            return False
        if key in (keyboard.Key.left, ) or char == "a":
            self._nudge(dy=self.step)
        elif key in (keyboard.Key.right,) or char == "d":
            self._nudge(dy=-self.step)
        elif key in (keyboard.Key.up,) or char == "w":
            self._nudge(dx=self.step)
        elif key in (keyboard.Key.down,) or char == "s":
            self._nudge(dx=-self.step)
        elif char == "q":
            self._nudge(dz=self.step)
        elif char == "e":
            self._nudge(dz=-self.step)
        elif char == "z":
            self._nudge(dr=self.roll_step)
        elif char == "c":
            self._nudge(dr=-self.roll_step)
        elif char == "h":
            self.arm.home()
            print("Home.", self.arm.get_pose())
        elif char == "p":
            print(self.arm.get_pose())
        elif key == keyboard.Key.space:
            if self.closed:
                self.arm.release()
            else:
                self.arm.grip()
            self.closed = not self.closed
        elif char in {"+", "="}:
            self.step = min(0.05, self.step + 0.005)
            print(f"Step {self.step:.3f} m")
        elif char == "-":
            self.step = max(0.005, self.step - 0.005)
            print(f"Step {self.step:.3f} m")


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        arm.set_speed(15)
        teleop = Teleop(arm)
        print(teleop.arm.get_pose())
        print("Keyboard control. Esc quits. Keep clear of the arm.")
        with keyboard.Listener(on_press=teleop.on_press) as listener:
            listener.join()
