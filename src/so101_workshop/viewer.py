"""Serve the existing Rerun arm blueprint in a browser panel.

The gRPC sink and the web viewer run in a separate ``rerun`` process. Hosting
them inside this process blocks the interpreter, so the arm could not move.

The preferred ports are 9881/9091. When another workshop or viewer already
owns them, the next free pair is used so two stations can run side by side.
"""

from urllib.parse import quote
from pathlib import Path
import atexit
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

from so101_utils.views import Twin


GRPC_PORT = 9881
WEB_PORT = 9091
PORT_SCAN = 20
# Live only: old frames are dropped, nothing is written to disk. The cap has to
# hold the static arm meshes plus recent camera frames. A few dozen megabytes
# is not enough; the server then discards the meshes and the plot history, and
# the panel shows a speck and empty charts.
VIEWER_MEMORY_LIMIT = "1GB"
SERVER_MEMORY_LIMIT = "512MB"

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
        _clear_viewer_record(getattr(process, "pid", None))
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
    _clear_viewer_record(pid)


def stop_all() -> None:
    """Stop every viewer this process spawned, including a stalled start."""
    for process in list(_PROCESSES):
        _stop_process(process)


def _viewer_state_path() -> Path:
    """One record per station, so a crashed run cannot leave a hidden viewer."""
    from so101_utils.config import ROOT

    digest = hashlib.sha1(str(Path(ROOT).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "so101-workshop" / f"viewer-{digest}.json"


def _record_viewer(process: subprocess.Popen, grpc_port: int, web_port: int) -> None:
    pid = getattr(process, "pid", None)
    if pid is None:
        return
    path = _viewer_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "pid": pid, "grpc_port": grpc_port, "web_port": web_port,
        }))
    except OSError:
        pass


def _clear_viewer_record(pid) -> None:
    if pid is None:
        return
    path = _viewer_state_path()
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    if state.get("pid") == pid:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_rerun_process(pid: int) -> bool:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, errors="replace", timeout=5,
            )
            return result.returncode == 0 and "rerun" in result.stdout.lower()
        if sys.platform == "darwin":
            # `comm` is only the interpreter, so read the full command line.
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True, text=True, errors="replace", timeout=5,
            )
            return "rerun" in result.stdout.lower()
        return b"rerun" in Path(f"/proc/{pid}/cmdline").read_bytes().lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _wait_for_port(port: int, timeout: float = 3.0) -> None:
    if port <= 0:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_open(port):
            return
        time.sleep(0.05)


def stop_stale_viewer() -> None:
    """Stop a viewer recorded by an earlier run of this station.

    Windows does not take child processes down with the workshop, so a crash
    or a closed console can leave one running on the preferred ports.
    """
    path = _viewer_state_path()
    try:
        state = json.loads(path.read_text())
        pid = int(state["pid"])
        grpc_port = int(state.get("grpc_port") or 0)
    except (OSError, ValueError, KeyError, TypeError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    if _is_rerun_process(pid):
        _terminate_pid(pid)
        _wait_for_port(grpc_port)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


_HANDLERS = []
_HANDLERS_INSTALLED = False


def install_cleanup_handlers() -> None:
    """Take viewer children down when the server exits or its console closes."""
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return
    _HANDLERS_INSTALLED = True
    atexit.register(stop_all)
    if os.name != "nt":
        return
    # Detaching the viewer from the console keeps Ctrl+C from killing it while
    # the SDK still has data to flush, so close events need their own cleanup.
    import ctypes
    from ctypes import wintypes

    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def handler(event):
        if event in (2, 5, 6):  # close, logoff, shutdown
            stop_all()
        return False

    callback = handler_type(handler)
    _HANDLERS.append(callback)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, True)


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
    # 0.26 is the last Rerun with an Intel Mac wheel. It has no --headless and
    # no --cors-allow-origin; --serve-web is the browser panel, and that build
    # already allows the page to connect across ports. web_origin is the page
    # the workshop iframe is served from; 0.26 does not take it as a flag.
    del web_origin
    return [
        binary, "--serve-web",
        "--bind", "127.0.0.1",
        "--port", str(grpc_port),
        "--web-viewer-port", str(web_port),
        "--memory-limit", VIEWER_MEMORY_LIMIT,
        "--server-memory-limit", SERVER_MEMORY_LIMIT,
        "--hide-welcome-screen",
    ]


def _spawn_options() -> dict:
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def open_viewer(urdf, intrinsics) -> WorkshopViewer:
    """Start a headless Rerun server and log the arm blueprint into it."""
    binary = rerun_binary()
    if binary is None:
        raise RuntimeError("The rerun viewer is not installed. Install the twin extra.")
    stop_stale_viewer()
    grpc_port, web_port = _pick_ports()
    web_origin = f"http://127.0.0.1:{web_port}"
    process = subprocess.Popen(
        _viewer_command(binary, grpc_port, web_port, web_origin),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_spawn_options(),
    )
    _PROCESSES.add(process)
    _record_viewer(process, grpc_port, web_port)
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
    # WebGPU draws a black 3D view and blank plots on current macOS browsers.
    # The web viewer falls back to WebGL when this is set.
    url = (
        f"{web_origin}/?url={quote(uri, safe='')}"
        "&persist=false&hide_welcome_screen&renderer=webgl"
    )
    return WorkshopViewer(twin, url, process, grpc_port)
