"""Local torque-off. The event is set before any serial call."""

import threading
from typing import Callable


def trip(event: threading.Event, torque_off: Callable[[], None] | None) -> None:
    """Latch e-stop, then turn torque off.

    `torque_off` may block on the serial lock. Callers observe `event` as set
    while that wait is still in progress.
    """
    event.set()
    if torque_off is not None:
        torque_off()
