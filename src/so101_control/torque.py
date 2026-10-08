"""Explicitly opted-in keyboard torque control; never an emergency stop."""

from contextlib import contextmanager
from threading import Lock
from collections import deque


class ToggleRequests:
    def __init__(self):
        self._lock = Lock()
        self._pending = False

    def request(self):
        with self._lock:
            self._pending = True

    def consume(self):
        with self._lock:
            pending = self._pending
            self._pending = False
            return pending


@contextmanager
def torque_toggle_listener(enabled=False):
    keys = ToggleRequests()
    if not enabled:
        yield keys
        return
    from pynput import keyboard

    pressed = set()

    def on_press(key):
        char = getattr(key, "char", None)
        if char and char.lower() == "t" and "t" not in pressed:
            pressed.add("t")
            keys.request()

    def on_release(key):
        if getattr(key, "char", "") and key.char.lower() == "t":
            pressed.discard("t")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        yield keys
    finally:
        listener.stop()


def process_torque_toggle(robot, currently_enabled):
    """Called from the main loop, never inside a keyboard callback."""
    bus = getattr(robot, "bus", None)
    if bus is None:
        raise RuntimeError("Torque control requires a connected LeRobot follower")
    if currently_enabled:
        bus.disable_torque()
        print("Torque OFF (arm may move freely)")
        return False
    bus.enable_torque()
    print("Torque ON")
    return True


class KeyboardRequests:
    """Thread-safe, debounced key events; callbacks never perform motor IO."""

    def __init__(self):
        self._lock = Lock()
        self._events = deque()

    def request(self, event):
        with self._lock:
            self._events.append(event)

    def consume(self):
        with self._lock:
            return self._events.popleft() if self._events else None


@contextmanager
def keyboard_requests(keys=("t", "enter", "c", "q")):
    """Listen for controls globally (X11); fail rather than silently lose keys."""
    from pynput import keyboard

    requests = KeyboardRequests()
    held = set()

    def identify(key):
        if key == keyboard.Key.enter:
            return "enter"
        if key == keyboard.Key.up:
            return "up"
        if key == keyboard.Key.down:
            return "down"
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def on_press(key):
        name = identify(key)
        if name in keys and name not in held:
            held.add(name)
            requests.request(name)

    def on_release(key):
        held.discard(identify(key))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        if not listener.is_alive():
            raise RuntimeError("Keyboard listener failed to start; use an X11 desktop")
        yield requests
    finally:
        listener.stop()


class ArmTorque:
    """Shared torque state: re-enable only after latching present positions."""

    def __init__(self, robot):
        self.robot = robot
        self.enabled = True

    def toggle(self):
        bus = getattr(self.robot, "bus", None)
        if bus is None:
            raise RuntimeError("Torque control requires a connected LeRobot follower")
        if self.enabled:
            bus.disable_torque()
            self.enabled = False
            print("Torque OFF (support the arm; it may move freely)")
        else:
            positions = bus.sync_read("Present_Position")
            bus.sync_write("Goal_Position", positions)
            bus.enable_torque()
            self.enabled = True
            print("Torque ON at measured position")
        return self.enabled