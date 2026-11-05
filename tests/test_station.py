"""Station resolution stays on the checkout when launched from it."""

from so101_workshop.station import prepare_station, resolve_station


def test_checkout_is_the_station(tmp_path, monkeypatch):
    monkeypatch.delenv("SO101_WORKSHOP_HOME", raising=False)
    root = resolve_station(None)
    assert (root / "config" / "robot.yaml").is_file()
    assert (root / "pyproject.toml").is_file()


def test_explicit_directory_is_seeded(tmp_path):
    station = prepare_station(str(tmp_path / "station"))
    assert (station / "config" / "robot.yaml").is_file()
    assert (station / "config" / "camera.yaml").is_file()
    assert "backend: recamera" in (station / "config" / "camera.yaml").read_text()
    text = (station / "config" / "robot.yaml").read_text()
    assert "port:" in text
    prepare_station(str(station))
    assert text == (station / "config" / "robot.yaml").read_text()
