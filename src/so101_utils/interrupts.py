"""Let a fold-to-rest cleanup survive the second Ctrl-C that uv forwards."""

import signal


def ignore_sigint() -> None:
    """Ignore further SIGINTs so a cleanup move can finish.

    `uv run` both receives the terminal's SIGINT and forwards its own, so one
    Ctrl-C can arrive twice. The first stops the task; letting the second abort
    the fold to rest would leave the arm over the table.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
