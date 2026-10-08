"""Identify the follower USB serial port without connecting to or moving the arm."""

from pathlib import Path
from typing import Callable

import yaml


def get_ports() -> set[str]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("Install the port-discovery dependency: python -m pip install -e '.[ports]'") from exc
    return {port.device for port in list_ports.comports()}


def find_port(
    config_path: str | Path,
    *,
    read_ports: Callable[[], set[str]] = get_ports,
    prompt: Callable[[str], str] = input,
) -> str | None:
    """Write only `port` after exactly one serial device disappears and user confirms."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Copy config/station.example.yaml to {path} first")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Station YAML must be a mapping; no port written")

    prompt("Connect the follower USB adapter (leave motors unpowered), then press Enter...")
    connected = read_ports()
    prompt("Disconnect JUST the follower USB adapter, then press Enter...")
    disconnected = read_ports()
    vanished = connected - disconnected
    if len(vanished) != 1:
        raise ValueError(f"Expected exactly one disappearing serial port, found {sorted(vanished)}; YAML unchanged")
    port = vanished.pop()
    print(f"Follower USB adapter detected at {port}")
    if prompt(f"Update only 'port' in {path}? Type YES to confirm: ").strip() != "YES":
        print("No changes saved")
        return None
    config["port"] = port
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Saved port in {path}; reconnect the follower USB adapter before using --enable-live")
    return port