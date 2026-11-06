"""Cartesian keyboard control. Steps are metres. Roll and pitch step about the
gripper's own axes; every goal is in the base frame.

The browser and the pynput script both feed tokens into this class. pynput is
imported only when a pynput key object arrives, so the workshop server does not
need it.
"""

from collections import deque
from threading import Lock
import time

from so101_control.arm import SO101
from so101_control.kinematics import IKError
from so101_utils.model import rotate_gripper

_MOVES = {"w", "a", "s", "d", "q", "e", "z", "c", "f", "v"}
# Same as the arm default.
TELEOP_SPEED_DEG_S = 90.0
# Metres per translation key press, and the range the +/- keys step through.
STEP_M = 0.005
MIN_STEP_M = 0.005
MAX_STEP_M = 0.05
# Degrees per roll and pitch key press.
ROLL_STEP_DEG = 5.0
PITCH_STEP_DEG = 5.0
# Loop pacing when nothing moved the arm, or the move was rejected.
IDLE_S = 0.02
ERROR_S = 0.05

_ARROWS = {"left": "a", "right": "d", "up": "w", "down": "s"}


class Teleop:
    def __init__(self, arm: SO101, cancel=None):
        self.arm = arm
        self.step = STEP_M
        self.roll_step = ROLL_STEP_DEG
        self.pitch_step = PITCH_STEP_DEG
        self.closed = False
        self.running = True
        self._cancel = cancel
        self._command = None  # xyz metres; roll, pitch, yaw only while those keys are used
        self._turn_point = None  # xyz a run of roll and pitch keys pivots about
        self._notice = None
        self._lock = Lock()
        self._held: set[str] = set()
        self._queue: deque[str] = deque()

    def _halted(self) -> bool:
        if not self.running:
            return True
        return self._cancel is not None and bool(self._cancel())

    def _token(self, key):
        if isinstance(key, str):
            return _ARROWS.get(key, key.lower())
        try:
            from pynput import keyboard
        except ImportError:
            return None
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

    def press_token(self, token: str) -> None:
        """Record one key from the browser. Motion happens later, in `poll`."""
        token = _ARROWS.get(token, token)
        if token == "esc":
            self.running = False
            return
        if not token:
            return
        with self._lock:
            if token in self._held:
                return
            self._held.add(token)
            self._queue.append(token)

    def release_token(self, token: str) -> None:
        token = _ARROWS.get(token, token)
        if not token:
            return
        with self._lock:
            self._held.discard(token)

    def reset(self) -> None:
        """Forget held keys and the target so the next nudge starts at the arm."""
        with self._lock:
            self._held.clear()
            self._queue.clear()
            self._command = None
            self._turn_point = None

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
        if self._halted():
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
            time.sleep(IDLE_S)
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
            self._turn_point = None
            try:
                arrived = self.arm.home(wait=False, cancel=self._halted)
            except (ValueError, RuntimeError) as exc:
                self._say(str(exc))
                return True
            self._notice = None
            if arrived:
                print("Home.", self.arm.get_pose())
            return True
        if token == "r":
            self._command = None
            self._turn_point = None
            try:
                arrived = self.arm.rest(wait=False, cancel=self._halted)
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
            self.step = min(MAX_STEP_M, self.step + STEP_M)
            print(f"Step {self.step:.3f} m")
            return True
        if token == "-":
            self.step = max(MIN_STEP_M, self.step - STEP_M)
            print(f"Step {self.step:.3f} m")
            return True
        return True

    def _nudge(self, dx=0.0, dy=0.0, dz=0.0, dr=0.0, dp=0.0) -> bool:
        if self._command is None:
            measured = self.arm.pose_at(self.arm.get_joints())
            base = (measured.x, measured.y, measured.z,
                    measured.roll, measured.pitch, measured.yaw)
        else:
            base = self._command
        turning = dr != 0.0 or dp != 0.0
        moving = dx != 0.0 or dy != 0.0 or dz != 0.0
        # A run of roll and pitch keys pivots about the point the run started at.
        # Re-basing on the solved pose each time would walk the tool point by the
        # solver error, which is what made a pitch press lurch.
        if turning and not moving and self._turn_point is not None:
            x, y, z = self._turn_point
        else:
            x, y, z = base[0] + dx, base[1] + dy, base[2] + dz
        try:
            # No settle wait: a live joint often stops short of the 2° band.
            # A nudge is one goal, not a ramp the user waits through.
            kwargs = {"wait": False, "single_goal": True}
            if self._cancel is not None:
                kwargs["cancel"] = self._halted
            if turning:
                result = self._turn((x, y, z), base[3:6], dr, dp, **kwargs)
                if result is None:
                    self._say("Turn blocked: the tool point would move")
                    return True
                (roll, pitch, yaw), arrived = result
            else:
                roll = pitch = yaw = None
                arrived = self.arm.move_to(x, y, z, **kwargs)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc))
            time.sleep(ERROR_S)
            return True
        if not arrived:
            self._command = None
            self._turn_point = None
            return True
        solved = self.arm.target_pose()
        if solved is None:
            self._command = None
            self._turn_point = None
            return True
        # Step from where the solver put the tool, not from the request. The
        # five-joint arm leaves a little solve error, and walking the request
        # ahead of it turns later nudges into branches and jumps.
        self._command = (solved.x, solved.y, solved.z,
                         solved.roll, solved.pitch, solved.yaw)
        self._turn_point = (x, y, z) if turning else None
        self._notice = None
        if roll is None:
            print(f"Goal x={x:.4f} m, y={y:.4f} m, z={z:.4f} m")
        else:
            print(f"Goal x={x:.4f} m, y={y:.4f} m, z={z:.4f} m, "
                  f"roll={roll:.1f} deg, pitch={pitch:.1f} deg, yaw={yaw:.1f} deg")
        return True

    def _turn(self, point, base_angles, dr, dp, **kwargs):
        """Turn about the gripper frame while holding the tool point.

        A step can need more wrist travel than the pose allows; halving keeps a
        blocked turn from snapping to a different arm branch. Returns the angles
        that moved and whether the move finished, or None when even the smallest
        step cannot hold the point.
        """
        for scale in (1.0, 0.5, 0.25):
            angles = rotate_gripper(base_angles, roll=dr * scale, pitch=dp * scale)
            try:
                arrived = self.arm.move_to(*point, *angles, hold_point=True, **kwargs)
                return angles, arrived
            except IKError:
                continue
        return None

    def _say(self, message: str) -> None:
        if message != self._notice:
            print(message)
            self._notice = message


def run(teleop: Teleop, listener_alive, refresh) -> None:
    """Apply keys, and keep drawing the joints the arm is actually at."""
    while teleop.running and listener_alive():
        teleop.poll()
        refresh()
