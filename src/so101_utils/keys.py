"""Keyboard events. Callbacks never talk to the motors."""

from collections import deque
from contextlib import contextmanager
from threading import Lock


class _Keys:
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
def keyboard(keys=("t", "enter", "c", "q")):
    """Global key listener (X11). consume() returns one key name, or None."""
    from pynput import keyboard as kb

    requests = _Keys()
    held = set()

    def identify(key):
        if key == kb.Key.enter:
            return "enter"
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def on_press(key):
        name = identify(key)
        if name in keys and name not in held:
            held.add(name)
            requests.request(name)

    def on_release(key):
        held.discard(identify(key))

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        if not listener.is_alive():
            raise RuntimeError("Keyboard listener did not start. An X11 session is required.")
        yield requests
    finally:
        listener.stop()
