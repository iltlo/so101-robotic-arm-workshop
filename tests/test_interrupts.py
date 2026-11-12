"""The fold-to-rest cleanup must survive the second Ctrl-C that uv forwards."""

import signal
import threading

from so101_utils.interrupts import ignore_sigint


def test_ignore_sigint_ignores_and_restores_the_handler():
    previous = signal.getsignal(signal.SIGINT)
    try:
        ignore_sigint()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_ignore_sigint_is_safe_off_the_main_thread():
    errors = []

    def worker():
        try:
            ignore_sigint()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []
