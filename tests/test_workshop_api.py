"""The loopback API drives a mock arm and does not open the camera."""

from types import SimpleNamespace
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from so101_utils.config import load_robot
from so101_utils.transforms import Intrinsics
from so101_workshop import agent as agent_module
from so101_workshop.agent import Agent
from so101_workshop.cli import main
from so101_workshop.server import create_app


@pytest.fixture
def client():
    agent = Agent()
    app = create_app(agent, listen=False)
    with TestClient(app) as test_client:
        test_client.agent = agent
        yield test_client


def test_run_help_exits_cleanly():
    with pytest.raises(SystemExit) as caught:
        main(["run", "--help"])
    assert caught.value.code == 0


def test_pose_poll_is_position_and_rpy_only(client):
    assert client.post("/api/arm/connect", json={"mock": True}).status_code == 200
    started = time.monotonic()
    for _ in range(30):
        body = client.get("/api/pose").json()
    elapsed = time.monotonic() - started
    pose = body["pose"]
    assert set(pose) == {"x", "y", "z", "roll", "pitch", "yaw"}
    assert "tool_pitch" not in body["pose"]
    assert "quaternion" not in body["pose"]
    assert elapsed < 1.0


def test_status_starts_disconnected(client):
    from so101_utils.moves import APPROACH_M, HOVER_M, PICK_HEIGHT_M
    body = client.get("/api/status").json()
    assert body["arm"]["connected"] is False
    assert body["estop"] is False
    assert body["instructor"]["enabled"] is False
    assert client.get("/api/instructor").json()["enabled"] is False
    assert body["motion"] == {
        "hover_m": HOVER_M,
        "pick_height_m": PICK_HEIGHT_M,
        "approach_m": APPROACH_M,
    }


def test_status_rereads_robot_yaml_only_when_it_changes(client, monkeypatch):
    assert client.post("/api/arm/connect", json={"mock": True}).status_code == 200
    loads = []
    real = agent_module.load_robot

    def counting(*args, **kwargs):
        loads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_module, "load_robot", counting)
    client.get("/api/status")
    client.get("/api/status")
    assert loads == [1]


def test_a_recamera_jpeg_is_kept_for_the_page(client):
    import cv2

    from so101_utils.camera import Frame

    image = np.zeros((12, 18, 3), np.uint8)
    image[1, 1] = (3, 4, 5)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    client.agent._store_camera_frame(Frame(None, 0.0, jpeg=jpeg, width=18, height=12))
    assert client.agent.latest_jpeg() == jpeg
    assert client.agent.latest_bgr().shape == (12, 18, 3)
    assert client.agent._frame_size == (18, 12)


def test_status_reuses_the_camera_catalog(client, monkeypatch):
    agent = client.agent
    calls = []
    original = agent.camera_profiles
    monkeypatch.setattr(agent, "camera_profiles", lambda: (calls.append(1), original())[1])
    client.get("/api/status")
    client.get("/api/status")
    assert len(calls) == 1
    client.get("/api/cameras")
    assert len(calls) == 2


def test_click_with_missing_pixels_is_a_bad_request(client):
    assert client.post("/api/arm/connect", json={"mock": True}).status_code == 200
    assert client.post("/api/click", json={"x": None, "y": None}).status_code == 400
    assert client.post("/api/click-pick", json={"x": None, "y": None}).status_code == 400


def test_estop_latches_on_a_mock_arm(client):
    connected = client.post("/api/arm/connect", json={"mock": True})
    assert connected.status_code == 200
    stopped = client.post("/api/estop")
    assert stopped.status_code == 200
    assert stopped.json()["estop"] is True
    assert client.agent.arm.torque_on is False
    assert client.agent.estop_event.is_set()
    held = client.post("/api/hold")
    assert held.status_code == 200
    assert client.agent.estop_event.is_set() is False
    assert client.agent.arm.torque_on is True


def test_estop_from_both_transports_turns_torque_off_once(client, monkeypatch):
    client.post("/api/arm/connect", json={"mock": True})
    arm = client.agent.arm
    calls = []
    original = arm.relax
    monkeypatch.setattr(arm, "relax", lambda: (calls.append(1), original()))
    assert client.post("/api/estop").status_code == 200
    assert client.post("/api/estop").status_code == 200
    assert len(calls) == 1
    assert arm.torque_on is False

    assert client.post("/api/hold").status_code == 200
    assert client.post("/api/estop").status_code == 200
    assert len(calls) == 2


def test_teleop_token_moves_the_mock_arm(client):
    assert client.post("/api/arm/connect", json={"mock": True}).status_code == 200
    start = client.agent.arm.get_pose().x
    pressed = client.post("/api/teleop", json={"action": "press", "token": "w"})
    assert pressed.status_code == 200
    client.post("/api/teleop", json={"action": "release", "token": "w"})
    tapped = client.post("/api/teleop", json={"action": "tap", "token": "w"})
    assert tapped.status_code == 200
    assert client.agent.arm.get_pose().x > start


def test_click_without_a_camera_is_rejected(client):
    client.post("/api/arm/connect", json={"mock": True})
    response = client.post("/api/click", json={"x": 10, "y": 10})
    assert response.status_code == 400
    assert "Camera" in response.json()["detail"]


def test_detections_panel_is_empty_until_yolo_reports(client):
    body = client.get("/api/detections").json()
    assert body["enabled"] is True
    assert body["boxes"] == []
    assert body["detector"] == "recamera"


def test_a_parsed_recamera_payload_stores_its_boxes(client):
    client.agent._on_camera_payload({
        "boxes": [[10, 12, 6, 4, 80, 1]],
        "labels": ["cup"],
        "resolution": [100, 80],
    })
    body = client.get("/api/detections").json()
    assert body["boxes"][0]["label"] == "cup"
    assert body["boxes"][0]["x1"] == 7


def test_a_repeated_camera_frame_is_encoded_once(client):
    from threading import Thread

    from so101_utils.camera import Frame

    image = np.zeros((8, 8, 3), np.uint8)
    frame = Frame(image, 0.0)

    class Cam:
        def poll(self):
            return frame

    agent = client.agent
    agent.cam = Cam()
    agent._cam_stop.clear()
    thread = Thread(target=agent._camera_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1.0
    jpeg = None
    while time.monotonic() < deadline:
        jpeg = agent.latest_jpeg()
        if jpeg:
            break
        time.sleep(0.01)
    assert jpeg
    time.sleep(0.08)
    assert agent.latest_jpeg() is jpeg
    agent._cam_stop.set()
    thread.join(1)


def test_hand_eye_yaml_is_read_once_until_it_changes(client, monkeypatch):
    loads = []
    real = agent_module.load_hand_eye

    def counting(*args, **kwargs):
        loads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(agent_module, "load_hand_eye", counting)
    assert client.agent._hand_eye_config() is not None
    assert client.agent._hand_eye_config() is client.agent._hand_eye_config()
    assert loads == [1]


def test_a_recamera_frame_stores_its_boxes(client):
    client.agent._on_camera_payload(
        '{"data": {"boxes": [[10, 12, 6, 4, 80, 1]], "labels": ["cup"], "resolution": [100, 80]}}'
    )
    body = client.get("/api/detections").json()
    assert body["source"] == "recamera"
    assert body["boxes"][0]["label"] == "cup"
    assert body["boxes"][0]["x1"] == 7


def test_estop_blocks_a_later_jog(client):
    client.post("/api/arm/connect", json={"mock": True})
    client.post("/api/estop")
    response = client.post("/api/jog", json={"name": "shoulder_pan", "delta_deg": 1})
    assert response.status_code == 400
    assert "E-stop" in response.json()["detail"]


def _prepare_auto(client, monkeypatch, dwell=0.01):
    """Mock arm plus a camera whose tag sits at a fixed base-frame pose."""
    client.post("/api/arm/connect", json={"mock": True})
    agent = client.agent
    agent.cam = SimpleNamespace(
        intrinsics=Intrinsics(64, 48, 50.0, 50.0, 32.0, 24.0),
    )
    agent._bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    tag = load_robot()["tag"]
    fixed = np.eye(4)
    fixed[:3, 3] = (0.0, 0.0, 0.5)

    def detect(_frame):
        current = agent.arm.gripper_link_pose()
        return np.linalg.inv(fixed) @ current @ tag.gripper_T_tag, None

    monkeypatch.setattr(agent, "_detect_tag", detect)
    monkeypatch.setattr(agent_module, "DWELL_S", dwell)
    saved = []
    monkeypatch.setattr(
        agent_module.station_config, "save_hand_eye",
        lambda pose, *, residual_m, residual_deg: saved.append(pose),
    )
    return agent, saved


def _wait_auto(client, timeout=60.0):
    deadline = time.monotonic() + timeout
    auto = {}
    while time.monotonic() < deadline:
        auto = client.get("/api/status").json()["auto_hand_eye"]
        if not auto["running"]:
            return auto
        time.sleep(0.02)
    return auto


def test_auto_hand_eye_requires_a_camera(client):
    client.post("/api/arm/connect", json={"mock": True})
    response = client.post("/api/calibrate/hand-eye/auto")
    assert response.status_code == 400
    assert "Camera" in response.json()["detail"]


def test_auto_hand_eye_tour_saves_ten_poses(client, monkeypatch):
    _agent, saved = _prepare_auto(client, monkeypatch)
    started = client.post("/api/calibrate/hand-eye/auto")
    assert started.status_code == 200, started.text
    assert started.json()["running"] is True

    auto = _wait_auto(client)
    assert auto["error"] is None, auto
    assert auto["cancelled"] is False
    assert auto["captured"] == len(agent_module.calibration_poses()) == 10
    assert auto["results"] == ["ok"] * 10
    assert auto["residual_m"] < 1e-6
    assert len(saved) == 1
    status = client.get("/api/status").json()
    assert len(status["hand_eye_samples"]) == 10


def test_a_missed_tag_marks_that_pose_failed(client, monkeypatch):
    agent, saved = _prepare_auto(client, monkeypatch)
    seen = {"n": 0}
    detect = agent._detect_tag

    def miss_the_third(frame):
        seen["n"] += 1
        if seen["n"] == 3:
            raise ValueError("no tag")
        return detect(frame)

    monkeypatch.setattr(agent, "_detect_tag", miss_the_third)
    assert client.post("/api/calibrate/hand-eye/auto").status_code == 200

    auto = _wait_auto(client)
    assert auto["error"] is None, auto
    assert auto["results"][2] == "fail"
    assert auto["results"].count("ok") == 9
    assert auto["captured"] == 9
    assert len(saved) == 1


def test_estop_stops_auto_hand_eye(client, monkeypatch):
    _prepare_auto(client, monkeypatch, dwell=1.0)
    assert client.post("/api/calibrate/hand-eye/auto").status_code == 200

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        auto = client.get("/api/status").json()["auto_hand_eye"]
        if auto["step"] >= 1:
            break
        time.sleep(0.02)
    assert client.post("/api/estop").status_code == 200

    auto = _wait_auto(client, timeout=15)
    assert auto["running"] is False
    assert auto["cancelled"] is True
    assert "E-stop" in auto["message"]


def test_place_jump_height_persists(client):
    client.post("/api/arm/connect", json={"mock": True})
    saved = client.post("/api/place/jump", json={"jump_cm": 8})
    assert saved.status_code == 200
    status = client.get("/api/status").json()
    assert status["motion"]["place_jump_m"] == pytest.approx(0.08)
    assert client.post("/api/place/jump", json={"jump_cm": 500}).status_code == 400


def test_relax_turns_torque_off_without_estop(client):
    client.post("/api/arm/connect", json={"mock": True})
    arm = client.agent.arm
    assert arm.torque_on is True
    relaxed = client.post("/api/relax")
    assert relaxed.status_code == 200
    assert arm.torque_on is False
    assert client.agent.estop_event.is_set() is False


def test_place_pose_can_be_saved_and_reset(client, monkeypatch):
    from so101_control.arm import REST_JOINTS

    client.post("/api/arm/connect", json={"mock": True})
    arm = client.agent.arm
    custom = {name: float(angle) + (5.0 if name == "shoulder_pan" else 0.0)
              for name, angle in REST_JOINTS.items()}

    def fake_joints():
        joints = arm.get_joints()
        joints.update(custom)
        return joints

    monkeypatch.setattr(arm, "recent_joints", fake_joints)
    saved = client.post("/api/place/set")
    assert saved.status_code == 200
    assert saved.json()["default"] is False
    status = client.get("/api/status").json()
    assert status["motion"]["place_default"] is False
    assert status["motion"]["place_joints"]["shoulder_pan"] == pytest.approx(custom["shoulder_pan"])

    reset = client.post("/api/place/reset")
    assert reset.status_code == 200
    assert reset.json()["default"] is True
    status = client.get("/api/status").json()
    assert status["motion"]["place_default"] is True
