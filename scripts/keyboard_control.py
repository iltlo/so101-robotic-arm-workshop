"""Cartesian keyboard control. Steps are metres. Roll, pitch, and yaw are base-frame degrees.

    WASD or arrows : x / y
    Q / E          : z down / up
    Z / C          : roll
    F / V          : pitch
    H              : home
    R              : rest
    P              : print pose
    Space          : grip / release
    + / -          : larger / smaller step
    Esc            : quit

Each press steps immediately. The next key does not wait for the arm to settle.
Hold a movement key to keep stepping. WASD and Q/E move the tool point and
leave orientation free. Z/C and F/V send the base-frame roll, pitch, and yaw.

    uv run python scripts/keyboard_control.py --mock
    uv run python scripts/keyboard_control.py --live

Rerun follows the measured joints, including while a move is still catching up.
--mock moves that drawing and does not open the serial port. Esc quits.
"""

from collections import deque
from threading import Lock
import time

from pynput import keyboard

from so101_control import SO101
from so101_utils.config import arm_parser, load_robot
from so101_utils.twin import watch_arm

_MOVES = {"w", "a", "s", "d", "q", "e", "z", "c", "f", "v"}
# Same as the arm default. An 8° step at this speed holds the next key for about 0.09 s.
TELEOP_SPEED_DEG_S = 90.0


class Teleop:
    def __init__(self, arm: SO101):
        self.arm = arm
        self.step = 0.005  # metres
        self.roll_step = 5.0  # degrees, base roll
        self.pitch_step = 5.0  # degrees, base pitch
        self.closed = False
        self.running = True
        self._command = None  # xyz metres; roll, pitch, yaw only while those keys are used
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
        dx, dy, dz, dr, dp = self._delta(held)
        if dx == dy == dz == dr == dp == 0.0:
            time.sleep(0.02)
            return
        self._nudge(dx, dy, dz, dr, dp)

    def _delta(self, tokens):
        dx = dy = dz = dr = dp = 0.0
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
                dz -= self.step
            elif token == "e":
                dz += self.step
            elif token == "z":
                dr += self.roll_step
            elif token == "c":
                dr -= self.roll_step
            elif token == "f":
                dp += self.pitch_step
            elif token == "v":
                dp -= self.pitch_step
        return dx, dy, dz, dr, dp

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

    def _nudge(self, dx=0.0, dy=0.0, dz=0.0, dr=0.0, dp=0.0) -> bool:
        if self._command is None:
            measured = self.arm.pose_at(self.arm.get_joints())
            base = (measured.x, measured.y, measured.z, None, None, None)
        else:
            base = self._command
        x, y, z = base[0] + dx, base[1] + dy, base[2] + dz
        # A pure translation must not pin orientation. This arm has five joints,
        # so holding roll, pitch, and yaw stops the tool point short of the goal.
        roll = pitch = yaw = None
        if dr != 0.0 or dp != 0.0:
            if base[3] is None:
                measured = self.arm.pose_at(self.arm.get_joints())
                base = (x, y, z, measured.roll, measured.pitch, measured.yaw)
            roll = base[3] + dr
            pitch = base[4] + dp
            yaw = base[5]
        try:
            # Do not wait for the 2° settle band. A live joint often stops
            # short of it, and that wait used to block every later key.
            self.arm.move_to(x, y, z, roll, pitch, yaw, wait=False)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc))
            time.sleep(0.05)
            return True
        self._command = (x, y, z, roll, pitch, yaw)
        self._notice = None
        if roll is None:
            print(f"Goal x={x:.4f} m, y={y:.4f} m, z={z:.4f} m")
        else:
            print(f"Goal x={x:.4f} m, y={y:.4f} m, z={z:.4f} m, "
                  f"roll={roll:.1f} deg, pitch={pitch:.1f} deg, yaw={yaw:.1f} deg")
        return True

    def _say(self, message: str) -> None:
        if message != self._notice:
            print(message)
            self._notice = message


def run(teleop: Teleop, listener_alive, refresh) -> None:
    """Apply keys, and keep drawing the joints the arm is actually at."""
    while teleop.running and listener_alive():
        teleop.poll()
        refresh()


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

        def refresh() -> None:
            twin.show(arm)
            twin.rr.get_global_data_recording().flush(timeout_sec=0)

        with keyboard.Listener(on_press=teleop.on_press, on_release=teleop.on_release) as listener:
            if not listener.is_alive():
                raise RuntimeError("Keyboard listener did not start. An X11 session is required.")
            try:
                run(teleop, listener.is_alive, refresh)
            except KeyboardInterrupt:
                print("Stopped.")
