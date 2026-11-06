"""Find the follower USB port without connecting to the motors."""

from pathlib import Path
import re
from typing import Callable

from so101_utils import config as config_mod


def get_ports() -> set[str]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("Install pyserial to list USB devices") from exc
    return {port.device for port in list_ports.comports()}


def port_that_vanished(before: set[str], *, read_ports: Callable[[], set[str]] = get_ports) -> str:
    """The one port present in `before` and absent now."""
    vanished = set(before) - set(read_ports())
    if len(vanished) != 1:
        raise ValueError(f"Expected one port to disappear, found {sorted(vanished)}. File unchanged.")
    return vanished.pop()


def write_port(port: str, config_path: str | Path | None = None) -> str:
    """Replace the single `port:` line in robot.yaml."""
    path = Path(config_path) if config_path else config_mod.CONFIG / "robot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text()
    updated, count = re.subn(r"(?m)^port:\s*.*$", f"port: {port}", text, count=1)
    if count != 1:
        raise ValueError(f"{path} needs a single port: line")
    path.write_text(updated if updated.endswith("\n") else updated + "\n")
    return port


def find_port(
    config_path: str | Path | None = None,
    *,
    read_ports: Callable[[], set[str]] = get_ports,
    prompt: Callable[[str], str] = input,
) -> str | None:
    """Ask the user to unplug the arm, then write `port` in robot.yaml."""
    path = Path(config_path) if config_path else config_mod.CONFIG / "robot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    prompt("Plug in the follower USB adapter (motors can stay off), then press Enter...")
    connected = read_ports()
    prompt("Unplug only that adapter, then press Enter...")
    port = port_that_vanished(connected, read_ports=read_ports)
    print(f"Follower adapter: {port}")
    if prompt(f"Write port: {port} in {path}? Type YES to confirm: ").strip() != "YES":
        print("No changes saved")
        return None
    write_port(port, path)
    print("Saved. Plug the adapter back in before --live.")
    return port
