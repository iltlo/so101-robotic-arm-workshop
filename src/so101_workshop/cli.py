"""`so101-workshop run` starts the loopback workshop server."""

import argparse
import os
import signal
import time
from pathlib import Path
from threading import Thread

from so101_utils.config import set_station
from so101_workshop.server import create_app
from so101_workshop.station import prepare_station


def _launcher_pid() -> int | None:
    """The uvx wrapper around this process, if uvx started us.

    uvx does not forward signals to its child. Killing the wrapper with SIGKILL
    would otherwise leave this station (and its Rerun viewer) running.
    """
    parent = os.getppid()
    if parent <= 1 or os.name == "nt":
        return None
    try:
        command = Path(f"/proc/{parent}/cmdline").read_bytes()
    except OSError:
        return None
    text = command.replace(b"\0", b" ").decode(errors="replace")
    return parent if "uv tool" in text or "uvx" in text else None


def _exit_with_launcher(parent: int) -> None:
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGTERM)
            return


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="so101-workshop")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start the local workshop and open it in a browser")
    run.add_argument("--port", type=int, default=8020)
    run.add_argument("--config-dir", default=None, help="station directory (default: this repo, or ~/.so101-workshop)")
    run.add_argument("--mock", action="store_true", help="connect the in-memory arm when the server starts")
    run.add_argument("--no-detectors", action="store_true", help="do not open the YOLO websocket or MQTT client")
    run.add_argument("--no-viewer", action="store_true", help="do not host the Rerun web viewer")
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)


def _run(args) -> None:
    import uvicorn

    station = prepare_station(args.config_dir)
    set_station(station)
    from so101_workshop.agent import Agent
    from so101_workshop.viewer import install_cleanup_handlers
    install_cleanup_handlers()
    agent = Agent()
    if args.mock:
        agent.connect_arm(mock=True)
    app = create_app(agent, listen=not args.no_detectors, viewer=not args.no_viewer)
    url = f"http://127.0.0.1:{args.port}"
    print(f"SO-101 workshop at {url}", flush=True)
    print(f"Station files: {station}", flush=True)
    launcher = _launcher_pid()
    if launcher is not None:
        Thread(target=_exit_with_launcher, args=(launcher,), daemon=True).start()
    uvicorn.run(
        app, host="127.0.0.1", port=args.port, log_level="info",
        # A browser tab keeps a websocket open, which otherwise makes Ctrl+C
        # hang on "Waiting for connections to close" and strands the viewer.
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
