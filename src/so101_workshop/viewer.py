"""Serve the existing Rerun arm blueprint in a browser panel.

The gRPC sink and the web viewer run in a separate ``rerun`` process. Hosting
them inside this process blocks the interpreter, so the arm could not move.

The preferred ports are 9881/9091. When another workshop or viewer already
owns them, the next free pair is used so two stations can run side by side.
"""

from urllib.parse import quote
from pathlib import Path
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from so101_utils.views import Twin


GRPC_PORT = 9881
WEB_PORT = 9091
PORT_SCAN = 20
# The stream is live-only: the viewer keeps a short rolling buffer for a
# late-joining page, then drops the oldest frames. Nothing is written to disk.
VIEWER_MEMORY_LIMIT = "256MB"
SERVER_MEMORY_LIMIT = "64MB"

# Every viewer spawned here, even ones whose handshake never finished.
# `stop_all` uses this at shutdown so a stalled start cannot leave a viewer
# holding the gRPC/web ports after the workshop exits.
_PROCESSES: set[subprocess.Popen] = set()


def rerun_binary() -> str | None:
    """The rerun CLI that belongs to this interpreter's environment.

    PATH often finds a snap or an unrelated install before the one that
    ``rerun-sdk`` installed next to this Python.
    """
    for name in ("rerun", "rerun.exe"):
        candidate = Path(sys.executable).with_name(name)
        if candidate.is_file():
            return str(candidate)
    return shutil.which("rerun")


def _stop_process(process: subprocess.Popen) -> None:
    """Stop the rerun wrapper and the real viewer child it may have spawned."""
    if process.poll() is not None:
        _PROCESSES.discard(process)
        return
    pid = getattr(process, "pid", None)
    if os.name == "nt":
        if pid is None:
            process.terminate()
        else:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
    elif pid is None:
        process.terminate()
    else:
        # start_new_session made it the group leader, so the child goes too.
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if pid is not None and os.name != "nt":
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                process.kill()
        else:
            process.kill()
    _PROCESSES.discard(process)


def stop_all() -> None:
    """Stop every viewer this process spawned, including a stalled start."""
    for process in list(_PROCESSES):
        _stop_process(process)


class WorkshopViewer:
    def __init__(self, twin, url: str, process: subprocess.Popen, grpc_port: int):
        self.twin = twin
        self.url = url
        self.process = process
        self.grpc_port = grpc_port
        self._closed = False

    def close(self) -> None:
        # Drop the gRPC sink first. Terminating the server while the SDK still
        # holds the sink makes every later flush print "connection severed".
        self._closed = True
        try:
            self.twin.close()
        except Exception:
            pass
        _stop_process(self.process)

    def running(self) -> bool:
        if self._closed or self.process.poll() is not None:
            return False
        return _port_open(self.grpc_port)


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.05)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _pick_ports() -> tuple[int, int]:
    for offset in range(PORT_SCAN):
        grpc, web = GRPC_PORT + offset, WEB_PORT + offset
        if _port_free(grpc) and _port_free(web):
            return grpc, web
    raise RuntimeError(
        f"No free Rerun ports between {GRPC_PORT} and {GRPC_PORT + PORT_SCAN - 1}"
    )


def _viewer_command(binary: str, grpc_port: int, web_port: int, web_origin: str) -> list[str]:
    return [
        binary, "--headless", "--serve-web",
        "--bind", "127.0.0.1",
        "--port", str(grpc_port),
        "--web-viewer-port", str(web_port),
        "--memory-limit", VIEWER_MEMORY_LIMIT,
        "--server-memory-limit", SERVER_MEMORY_LIMIT,
        "--hide-welcome-screen",
        "--cors-allow-origin", web_origin,
        "--cors-allow-origin", f"http://localhost:{web_port}",
    ]


def open_viewer(urdf, intrinsics) -> WorkshopViewer:
    """Start a headless Rerun server and log the arm blueprint into it."""
    binary = rerun_binary()
    if binary is None:
        raise RuntimeError("The rerun viewer is not installed. Install the twin extra.")
    grpc_port, web_port = _pick_ports()
    web_origin = f"http://127.0.0.1:{web_port}"
    session = {} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(
        _viewer_command(binary, grpc_port, web_port, web_origin),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **session,
    )
    _PROCESSES.add(process)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _stop_process(process)
            raise RuntimeError("Rerun viewer exited before it started listening")
        if _port_open(grpc_port) and _port_open(web_port):
            break
        time.sleep(0.05)
    else:
        _stop_process(process)
        raise RuntimeError("Rerun viewer did not start")
    if process.poll() is not None:
        _stop_process(process)
        raise RuntimeError("Rerun viewer exited before it started listening")
    uri = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
    try:
        # A restarted viewer must not keep the previous gRPC sink, or its
        # first flush raises "connection severed" against the dead port.
        import rerun as rr
        rr.disconnect()
        twin = Twin(urdf, spawn=False, intrinsics=intrinsics, grpc_url=uri)
    except Exception:
        try:
            import rerun as rr
            rr.disconnect()
        except Exception:
            pass
        _stop_process(process)
        raise
    # localhost often resolves to IPv6, and this server is bound to 127.0.0.1.
    # persist would restore the welcome screen instead of this recording.
    url = (
        f"{web_origin}/?url={quote(uri, safe='')}"
        "&persist=false&hide_welcome_screen"
    )
    return WorkshopViewer(twin, url, process, grpc_port)
