"""The workshop page stays wired, and camera profiles work without hardware."""

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so101_utils import config as station_config
from so101_workshop.agent import Agent
from so101_workshop.server import create_app

PAGE = Path(__file__).resolve().parents[1] / "src" / "so101_workshop" / "static" / "index.html"


@pytest.fixture
def station(tmp_path, monkeypatch):
    source = station_config.CONFIG
    config = tmp_path / "config"
    config.mkdir()
    for name in ("camera.yaml", "camera_demo.yaml", "camera_realsense.yaml"):
        shutil.copy(source / name, config / name)
    monkeypatch.setattr(station_config, "CONFIG", config)
    return config


def test_script_references_only_existing_elements():
    text = PAGE.read_text()
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    referenced = set(re.findall(r'\$\("([^"]+)"\)', text))
    assert referenced <= ids, sorted(referenced - ids)


def test_each_named_button_is_wired():
    text = PAGE.read_text()
    referenced = set(re.findall(r'\$\("([^"]+)"\)', text))
    for tag in re.findall(r"<button\b[^>]*>", text):
        found = re.search(r'\bid="([^"]+)"', tag)
        if found:
            assert found.group(1) in referenced, f"{found.group(1)} has no handler"


def test_script_parses_with_node(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    text = PAGE.read_text()
    script = text.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    path = tmp_path / "index.js"
    path.write_text(script)
    result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_control_panel_and_rerun_share_the_width():
    text = PAGE.read_text()
    block = text.split(".layout {", 1)[1].split("}", 1)[0]
    assert "repeat(2, minmax(0, 1fr))" in block


def test_camera_profiles_are_listed_and_selected(station):
    agent = Agent()
    with TestClient(create_app(agent, listen=False)) as client:
        data = client.get("/api/cameras").json()
        names = {profile["name"] for profile in data["profiles"]}
        assert {"camera.yaml", "camera_demo.yaml", "camera_realsense.yaml"} <= names
        demo = next(item for item in data["profiles"] if item["name"] == "camera_demo.yaml")
        assert demo["backend"] == "demo"
        assert demo["available"] is True
        assert (demo["width"], demo["height"]) == (1280, 720)

        chosen = client.post("/api/cameras/select", json={"profile": "camera_demo.yaml"})
        assert chosen.status_code == 200, chosen.text
        saved = (station / "workshop.yaml").read_text()
        assert "camera_profile: camera_demo.yaml" in saved

        status = client.get("/api/status").json()
        assert status["camera"]["profile"] == "camera_demo.yaml"
        assert status["camera"]["presets"]

        refused = client.post("/api/cameras/select", json={"profile": "../camera.yaml"})
        assert refused.status_code == 400


def test_demo_camera_opens_and_streams(station):
    agent = Agent()
    with TestClient(create_app(agent, listen=False)) as client:
        started = client.post("/api/camera/connect", json={"profile": "camera_demo.yaml"})
        assert started.status_code == 200, started.text
        deadline = time.monotonic() + 5
        connected = False
        while time.monotonic() < deadline:
            status = client.get("/api/status").json()
            if status["camera"]["connected"]:
                connected = True
                break
            time.sleep(0.05)
        assert connected, client.get("/api/status").json()["camera"]
        deadline = time.monotonic() + 3
        while agent.latest_jpeg() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert agent.latest_jpeg() is not None, agent.camera_error
        chunk = next(agent.mjpeg())
        assert chunk.startswith(b"--frame")
        assert b"image/jpeg" in chunk

        assert client.post("/api/camera/disconnect").status_code == 200
        assert agent.cam is None
