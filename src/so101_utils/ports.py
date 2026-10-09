"""Find the follower USB port without connecting to the motors."""

from pathlib import Path
import re
from typing import Callable

from so101_utils.config import CONFIG


def get_ports() -> set[str]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("Install pyserial (the ports extra) to list USB devices") from exc
    return {port.device for port in list_ports.comports()}


def find_port(
    config_path: str | Path | None = None,
    *,
    read_ports: Callable[[], set[str]] = get_ports,
    prompt: Callable[[str], str] = input,
) -> str | None:
    """Ask the user to unplug the arm, then write `port` in robot.yaml."""
    path = Path(config_path) if config_path else CONFIG / "robot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    prompt("Plug in the follower USB adapter (motors can stay off), then press Enter...")
    connected = read_ports()
    prompt("Unplug only that adapter, then press Enter...")
    vanished = connected - read_ports()
    if len(vanished) != 1:
        raise ValueError(f"Expected one port to disappear, found {sorted(vanished)}. File unchanged.")
    port = vanished.pop()
    print(f"Follower adapter: {port}")
    if prompt(f"Write port: {port} in {path}? Type YES to confirm: ").strip() != "YES":
        print("No changes saved")
        return None
    text = path.read_text()
    updated, count = re.subn(r"(?m)^port:\s*.*$", f"port: {port}", text, count=1)
    if count != 1:
        raise ValueError(f"{path} needs a single port: line")
    path.write_text(updated if updated.endswith("\n") else updated + "\n")
    print(f"Saved. Plug the adapter back in before --live.")
    return port
