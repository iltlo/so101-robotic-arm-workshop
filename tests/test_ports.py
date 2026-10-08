"""Port discovery is simulated; tests never enumerate or write to a real USB device."""

import yaml
import pytest

from so101_control.ports import find_port


def _setup(tmp_path):
    path = tmp_path / "station.yaml"
    original = {"port": "/dev/old", "camera_serial": "rgb-camera", "joint_limits_deg": {}}
    path.write_text(yaml.safe_dump(original))
    return path, original


def test_writes_only_port_on_unique_disappearance(tmp_path):
    path, original = _setup(tmp_path)
    observations = iter([{"/dev/ttyACM0", "/dev/ttyUSB0"}, {"/dev/ttyUSB0"}])
    assert find_port(path, read_ports=lambda: next(observations), prompt=lambda _: "YES") == "/dev/ttyACM0"
    assert yaml.safe_load(path.read_text()) == {**original, "port": "/dev/ttyACM0"}


@pytest.mark.parametrize("connected,disconnected", [
    ({"/dev/ttyACM0"}, {"/dev/ttyACM0"}),
    ({"/dev/ttyACM0", "/dev/ttyUSB0"}, set()),
])
def test_ambiguous_or_missing_does_not_write(tmp_path, connected, disconnected):
    path, _ = _setup(tmp_path)
    before = path.read_bytes()
    observations = iter([connected, disconnected])
    with pytest.raises(ValueError, match="exactly one"):
        find_port(path, read_ports=lambda: next(observations), prompt=lambda _: "")
    assert path.read_bytes() == before


def test_decline_keeps_configuration(tmp_path):
    path, _ = _setup(tmp_path)
    before = path.read_bytes()
    observations = iter([{"/dev/ttyACM1"}, set()])
    assert find_port(path, read_ports=lambda: next(observations), prompt=lambda _: "") is None
    assert path.read_bytes() == before


def test_missing_station_file_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_port(tmp_path / "station.yaml", prompt=lambda _: "")


def test_cli_routes_find_port_without_loading_station(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    import so101_control.ports as ports

    filename = tmp_path / "station.yaml"
    monkeypatch.setattr(ports, "find_port", lambda path: path)
    path = Path(__file__).parents[1] / "setup" / "find_port.py"
    spec = importlib.util.spec_from_file_location("setup_find_port", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--config", str(filename)]) == str(filename)


def test_real_pyserial_enumeration_returns_a_set():
    from so101_control.ports import get_ports

    ports = get_ports()  # read-only; a follower need not be connected
    assert isinstance(ports, set)
    assert all(isinstance(name, str) for name in ports)