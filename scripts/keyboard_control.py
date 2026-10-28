"""Cartesian keyboard control. Steps are metres; wrist roll is degrees.

    WASD or arrows : x / y
    Q / E          : z up / down
    Z / C          : wrist roll
    H              : home
    R              : rest
    P              : print pose
    Space          : grip / release
    + / -          : larger / smaller step
    Esc            : quit

Each press steps immediately. The next key does not wait for the arm to settle.
Hold a movement key to keep stepping.

    uv run python scripts/keyboard_control.py --mock
    uv run python scripts/keyboard_control.py --live

Rerun shows the arm. --mock moves that drawing and does not open the serial port. Esc quits.
"""

from collections import deque
from threading import Lock
import time

from pynput import keyboard

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.twin import watch_arm

_MOVES = {"w", "a", "s", "d", "q", "e", "z", "c"}


class Teleop:
    def __init__(self, arm: SO101):
        self.arm = arm
        self.step = 0.01  # metres
        self.roll_step = 5.0  # degrees
        self.closed = False
        self.running = True
        self._command = None  # xyz metres and wrist_roll degrees
        self._notice = None
        self._lock = Lock()
        self._held: set[str] = set()
        self._queue: deque[str] = deque()

    def _token(self, key):
        if key == keyboard.Key.esc:
            return "esc"
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.left:
            return "a"
        if key == keyboard.Key.right:
            return "d"
        if key == keyboard.Key.up:
            return "w"
        if key == keyboard.Key.down:
            return "s"
        char = getattr(key, "char", None)
        return char.lower() if isinstance(char, str) else None

    def on_press(self, key):
        # Listener thread: record the key and return. A motor call here blocks
        # every later key, and closing the port on Ctrl-C breaks that read.
        token = self._token(key)
        if token == "esc":
            self.running = False
            return False
        if token is None:
            return
        with self._lock:
            if token in self._held:
                return
            self._held.add(token)
            self._queue.append(token)

    def on_release(self, key):
        token = self._token(key)
        if token is None:
            return
        with self._lock:
            self._held.discard(token)

    def poll(self) -> None:
        """Apply one queued key, or one step for the keys still held."""
        if not self.running:
            return
        with self._lock:
            token = self._queue.popleft() if self._queue else None
            held = set(self._held)
        if token is not None:
            if not self._apply(token):
                with self._lock:
                    self._queue.appendleft(token)
            return
        dx, dy, dz, dr = self._delta(held)
        if dx == dy == dz == dr == 0.0:
            time.sleep(0.02)
            return
        self._nudge(dx, dy, dz, dr)

    def _delta(self, tokens):
        dx = dy = dz = dr = 0.0
        for token in tokens:
            if token == "w":
                dx += self.step
            elif token == "s":
                dx -= self.step
            elif token == "a":
                dy += self.step
            elif token == "d":
                dy -= self.step
            elif token == "q":
                dz += self.step
            elif token == "e":
                dz -= self.step
            elif token == "z":
                dr += self.roll_step
            elif token == "c":
                dr -= self.roll_step
        return dx, dy, dz, dr

    def _apply(self, token: str) -> bool:
        if token in _MOVES:
            return self._nudge(*self._delta([token]))
        if token == "h":
            self._command = None
            try:
                arrived = self.arm.home(wait=False, cancel=lambda: not self.running)
            except (ValueError, RuntimeError) as exc:
                self._say(str(exc))
                return True
            self._notice = None
            if arrived:
                print("Home.", self.arm.get_pose())
            return True
        if token == "r":
            self._command = None
            try:
                arrived = self.arm.rest(wait=False, cancel=lambda: not self.running)
            except (ValueError, RuntimeError) as exc:
                self._say(str(exc))
                return True
            self._notice = None
            if arrived:
                self.closed = True
                print("Rest.", self.arm.get_pose())
            return True
        if token == "p":
            print(self.arm.get_pose())
            return True
        if token == "space":
            try:
                if self.closed:
                    self.arm.release()
                else:
                    self.arm.grip()
            except (ValueError, RuntimeError) as exc:
                self._say(str(exc))
                return True
            self.closed = not self.closed
            return True
        if token in {"+", "="}:
            self.step = min(0.05, self.step + 0.005)
            print(f"Step {self.step:.3f} m")
            return True
        if token == "-":
            self.step = max(0.005, self.step - 0.005)
            print(f"Step {self.step:.3f} m")
            return True
        return True

    def _nudge(self, dx=0.0, dy=0.0, dz=0.0, dr=0.0) -> bool:
        measured = self.arm.get_pose()
        measured_roll = self.arm.get_joints()["wrist_roll"]
        base = self._command
        if base is None:
            base = (measured.x, measured.y, measured.z, measured_roll)
        x, y, z = base[0] + dx, base[1] + dy, base[2] + dz
        roll = base[3] + dr
        # The goal stays near the measured hand. Otherwise a held key walks
        # the target away while the arm is still catching up.
        ahead = max(abs(x - measured.x), abs(y - measured.y), abs(z - measured.z))
        if ahead > self.step * 5 + 1e-6 or abs(roll - measured_roll) > self.roll_step * 5 + 1e-6:
            self._say("Waiting for the arm to catch up.")
            time.sleep(0.05)
            return False
        try:
            # Do not wait for the 2° settle band. A live joint often stops
            # short of it, and that wait used to block every later key.
            self.arm.move_to(x, y, z, r=roll, wait=False)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc))
            time.sleep(0.05)
            return True
        self._command = (x, y, z, roll)
        self._notice = None
        print(f"Goal x={x:.4f} m, y={y:.4f} m, z={z:.4f} m, wrist_roll={roll:.1f} deg")
        return True

    def _say(self, message: str) -> None:
        if message != self._notice:
            print(message)
            self._notice = message


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    with SO101.connect(mock=args.mock) as arm:
        watch_arm(arm, load_robot()["urdf"])
        print("Arm is in the Rerun window.")
        teleop = Teleop(arm)
        print(teleop.arm.get_pose())
        print("Keyboard control. Esc quits. Keep clear of the arm.")
        with keyboard.Listener(on_press=teleop.on_press, on_release=teleop.on_release) as listener:
            if not listener.is_alive():
                raise RuntimeError("Keyboard listener did not start. An X11 session is required.")
            try:
                while teleop.running and listener.is_alive():
                    teleop.poll()
            except KeyboardInterrupt:
                print("Stopped.")
