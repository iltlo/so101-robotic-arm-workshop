"""Every workshop feature that can run without the USB arm or the reCamera."""

from importlib.metadata import entry_points
from threading import Thread
import asyncio
import json
import socket
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from so101_utils.click_app import click_target
from so101_utils.config import load_camera, load_hand_eye
from so101_utils.moves import HOVER_M, PICK_HEIGHT_M, PICK_TOOL_OFFSET_M
from so101_workshop.agent import Agent, load_settings
from so101_workshop.detections import DetectionBridge, DetectionStore
from so101_workshop.hooks import NullHook
from so101_workshop.server import create_app
from so101_utils import config as station_config


class _Frame:
    def __init__(self, image):
        self.rgb_bgr = image


class _Cam:
    def __init__(self, image, intrinsics):
        self.intrinsics = intrinsics
        self.image = image

    def poll(self):
        return _Frame(self.image)


class _Hook(NullHook):
    def __init__(self):
        self.statuses = []
        self.frames = []
        self.boxes = []

    def on_status(self, status):
        self.statuses.append(status["message"])

    def on_frame(self, jpeg):
        self.frames.append(jpeg)

    def on_detections(self, boxes):
        self.boxes.append(list(boxes))


@pytest.fixture
def client():
    hook = _Hook()
    agent = Agent(hook)
    app = create_app(agent, listen=False)
    with TestClient(app) as test_client:
        test_client.agent = agent
        test_client.hook = hook
        yield test_client


def _connect(client):
    response = client.post("/api/arm/connect", json={"mock": True})
    assert response.status_code == 200, response.text
    return client.agent


def _attach_camera(agent, image=None):
    intrinsics = load_camera()["intrinsics"]
    if image is None:
        image = np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
        image[10, 10] = (8, 16, 32)
    agent.cam = _Cam(image, intrinsics)
    agent._frame_size = (intrinsics.width, intrinsics.height)
    agent._bgr = image
    agent._jpeg = b"stored-jpeg"
    return intrinsics


def _reachable_pixel(agent, height_m):
    """A pixel whose table ray the mock arm can reach with the fingers down."""
    intrinsics = agent.cam.intrinsics
    hand = load_hand_eye()["base_T_camera"]
    state = agent.arm.get_joints()
    width, height = intrinsics.width, intrinsics.height
    for y in range(height // 10, height, height // 6):
        for x in range(width // 10, width, width // 8):
            try:
                target = click_target((x, y), intrinsics, hand, height_m)
                agent.arm._kin.position_ik(
                    state, (float(target[0]), float(target[1]), float(target[2])),
                    agent.arm._limits, downward=True,
                )
            except Exception:
                continue
            return x, y, target
    raise AssertionError("No table pixel was reachable from the mock rest joints")


def test_console_script_is_so101_workshop():
    names = {item.name for item in entry_points(group="console_scripts")}
    assert "so101-workshop" in names


def test_page_lists_the_controls(client):
    page = client.get("/")
    assert page.status_code == 200
    for label in ("E-stop", "Torque on", "Mock arm", "Connect camera", "Hardware",
                  "Motor calibration", "Hand-eye", "Click and go", "Click and pick", "Rerun",
                  "Camera profile", "Intrinsics", "Start Rerun",
                  "Z down", "Roll +", "Pitch −", "H · Home", "R · Rest", "Space · Gripper",
                  "Home", "Rest", "Gripper", "Pose", "Step +", "Step −"):
        assert label in page.text
    assert "no stream" in page.text
    viewer = client.get("/api/viewer").json()
    assert viewer["enabled"] is False
    assert client.post("/api/viewer/mode", json={"mode": "hand-eye"}).status_code == 200
    assert client.post("/api/viewer/start").status_code == 400
    ports = client.get("/api/ports")
    assert ports.status_code == 200
    assert isinstance(ports.json()["ports"], list)
    probe = client.post("/api/probe", json={"host": "127.0.0.1"})
    assert probe.status_code == 200
    assert set(probe.json()["ports"]) == {"80", "8090"}
    assert probe.json()["ports"]["8090"] is False
    status = client.get("/api/status").json()
    assert status["camera"]["connected"] is False
    assert client.hook.statuses == [status["message"]]


def test_settings_default_instructor_off(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "workshop.yaml").write_text("features:\n  teleop: false\n  estop: false\n")
    monkeypatch.setattr(station_config, "CONFIG", config)
    features = load_settings()["features"]
    assert features["teleop"] is False
    assert features["estop"] is False
    assert features["click_and_go"] is True
    assert features["instructor"] is False


def test_teleop_tokens_jog_and_gripper(client):
    agent = _connect(client)
    start = agent.arm.get_pose()

    def tap(token):
        response = client.post("/api/teleop", json={"action": "tap", "token": token})
        assert response.status_code == 200, response.text
        return agent.arm.get_pose()

    assert tap("w").x > start.x + 0.002
    after_w = agent.arm.get_pose().x
    assert tap("s").x < after_w - 0.002
    # This arm has five joints, so a sideways step can undershoot the 5 mm goal.
    y_before = agent.arm.get_pose().y
    assert tap("left").y > y_before
    assert agent.teleop._command[1] == pytest.approx(y_before + 0.005, abs=1e-4)
    z_before = agent.arm.get_pose().z
    assert tap("e").z > z_before
    assert agent.teleop._command[2] == pytest.approx(z_before + 0.005, abs=1e-4)
    roll_before = agent.arm.get_pose().roll
    assert abs(tap("z").roll - roll_before) > 1.0
    pitch_before = agent.arm.get_pose().pitch
    assert abs(tap("f").pitch - pitch_before) > 1.0
    for token in ("a", "d", "q", "c", "v", "right", "up", "down"):
        assert client.post("/api/teleop", json={"action": "tap", "token": token}).status_code == 200

    assert agent.teleop.closed is False
    client.post("/api/teleop", json={"action": "tap", "token": "space"})
    assert agent.teleop.closed is True
    client.post("/api/teleop", json={"action": "tap", "token": "space"})
    assert agent.teleop.closed is False
    assert agent.arm.get_joints()["gripper"] == pytest.approx(50.0, abs=0.2)
    client.post("/api/teleop", json={"action": "tap", "token": "h"})
    homed = agent.arm.get_joints()
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        assert homed[name] == pytest.approx(0.0, abs=0.2)
    client.post("/api/teleop", json={"action": "tap", "token": "r"})
    assert agent.arm.get_joints()["shoulder_lift"] == pytest.approx(-90.0, abs=0.2)

    jog = client.post("/api/jog", json={"name": "shoulder_pan", "delta_deg": 1})
    assert jog.status_code == 200, jog.text
    assert agent.arm.get_joints()["shoulder_pan"] == pytest.approx(1.0, abs=0.2)
    too_far = client.post("/api/jog", json={"name": "shoulder_pan", "delta_deg": 3})
    assert too_far.status_code == 400
    unknown = client.post("/api/teleop", json={"action": "fly", "token": "w"})
    assert unknown.status_code == 400

    tuned = client.post("/api/teleop/config", json={"step_m": 0.02, "speed_deg_s": 45})
    assert tuned.status_code == 200, tuned.text
    assert tuned.json() == pytest.approx({"step_m": 0.02, "speed_deg_s": 45})
    status = client.get("/api/status").json()
    assert status["teleop"]["step_m"] == pytest.approx(0.02)
    assert status["arm"]["speed_deg_s"] == pytest.approx(45)
    refused = client.post("/api/teleop/config", json={"step_m": 0.5})
    assert refused.status_code == 400

    client.post("/api/teleop", json={"action": "press", "token": "w"})
    assert "w" in agent.teleop._held
    client.post("/api/estop")
    assert agent.estop_event.is_set()
    assert not agent.teleop._held
    assert client.post("/api/teleop", json={"action": "tap", "token": "w"}).status_code == 400


def test_estop_forgets_the_teleop_target_and_last_command(client):
    agent = _connect(client)
    assert client.post("/api/teleop", json={"action": "tap", "token": "w"}).status_code == 200
    assert agent.teleop._command is not None
    assert agent.arm._sent is not None

    assert client.post("/api/estop").status_code == 200
    assert agent.teleop._command is None
    assert agent.arm._sent is None
    assert agent.arm._goal is None

    assert client.post("/api/hold").status_code == 200
    assert agent.teleop._command is None
    assert agent.arm._sent is None


def test_websocket_estop_and_camera_disconnect(client):
    agent = _connect(client)
    with client.websocket_connect("/api/ws") as socket:
        socket.send_json({"type": "press", "token": "d"})
        assert socket.receive_json()["type"] == "ok"
        socket.send_json({"type": "release", "token": "d"})
        socket.receive_json()
        socket.send_json({"type": "estop"})
        body = socket.receive_json()
        assert body["estop"] is True
        socket.send_json({"type": "status"})
        assert socket.receive_json()["estop"] is True
    assert agent.arm.torque_on is False

    held = client.post("/api/hold")
    assert held.status_code == 200
    assert agent.arm.torque_on is True

    finished = Thread(target=lambda: None)
    finished.start()
    finished.join()
    agent._cam_thread = finished
    agent.disconnect_camera()
    assert agent.cam is None
    client.post("/api/arm/disconnect")
    assert agent.arm is None
    status = client.get("/api/status").json()
    assert status["arm"]["connected"] is False


def test_click_cancels_and_pick_uses_the_box_center(client, monkeypatch):
    agent = _connect(client)
    _attach_camera(agent)
    client.post("/api/teleop", json={"action": "tap", "token": "h"})
    pixel_x, pixel_y, target = _reachable_pixel(agent, HOVER_M)
    agent.arm.set_speed(20)

    first = {}
    second = {}

    def click(store):
        store["response"] = client.post("/api/click", json={"x": pixel_x, "y": pixel_y})

    left = Thread(target=click, args=(first,))
    left.start()
    time.sleep(0.15)
    right = Thread(target=click, args=(second,))
    right.start()
    left.join(30)
    right.join(30)
    assert first["response"].status_code == 200, first["response"].text
    assert second["response"].status_code == 200, second["response"].text
    assert first["response"].json()["status"] == "replaced"
    assert second["response"].json()["status"] == "arrived"
    arrived = second["response"].json()["target"]
    assert arrived[0] == pytest.approx(float(target[0]), abs=1e-6)
    pose = agent.arm.get_pose()
    assert pose.x == pytest.approx(arrived[0], abs=0.02)
    assert pose.y == pytest.approx(arrived[1], abs=0.02)

    outside = client.post("/api/click", json={"x": -1, "y": 0})
    assert outside.status_code == 400

    missing = client.post("/api/pick", json={"index": 0})
    assert missing.status_code == 400

    intrinsics = agent.cam.intrinsics
    agent.detections.update({
        "boxes": [{
            "x1": pixel_x - 4, "y1": pixel_y - 4, "x2": pixel_x + 4, "y2": pixel_y + 4,
            "score": 88, "class_id": 3, "label": "cup",
        }],
        "resolution": [intrinsics.width, intrinsics.height],
    })
    panel = client.get("/api/detections").json()
    assert panel["boxes"][0]["label"] == "cup"
    assert b"cup" not in agent.latest_jpeg()

    monkeypatch.setattr("so101_utils.moves.hold", lambda arm, seconds: None)
    agent.arm.set_speed(90)
    picked = client.post("/api/pick", json={"index": 0})
    assert picked.status_code == 200, picked.text
    assert picked.json()["status"] == "picked"
    pick_target = click_target((pixel_x, pixel_y), intrinsics, load_hand_eye()["base_T_camera"], PICK_HEIGHT_M)
    assert picked.json()["target"][2] == pytest.approx(float(pick_target[2]), abs=1e-6)
    assert agent.arm.ee_tool_offset == pytest.approx(PICK_TOOL_OFFSET_M)


def test_click_pick_runs_the_pick_at_the_pixel(client, monkeypatch):
    agent = _connect(client)
    _attach_camera(agent)
    client.post("/api/teleop", json={"action": "tap", "token": "h"})
    pixel_x, pixel_y, target = _reachable_pixel(agent, PICK_HEIGHT_M)
    agent.arm.set_speed(90)
    monkeypatch.setattr("so101_utils.moves.hold", lambda arm, seconds: None)
    picked = client.post("/api/click-pick", json={"x": pixel_x, "y": pixel_y})
    assert picked.status_code == 200, picked.text
    assert picked.json()["status"] == "picked"
    assert picked.json()["target"][0] == pytest.approx(float(target[0]), abs=1e-6)
    assert agent.arm.ee_tool_offset == pytest.approx(PICK_TOOL_OFFSET_M)
    assert client.post("/api/click-pick", json={"x": -1, "y": 0}).status_code == 400


def test_detect_pick_picks_the_box_centre_not_the_click(client, monkeypatch):
    agent = _connect(client)
    intrinsics = _attach_camera(agent)
    client.post("/api/teleop", json={"action": "tap", "token": "h"})
    pixel_x, pixel_y, _ = _reachable_pixel(agent, PICK_HEIGHT_M)
    agent.arm.set_speed(90)
    monkeypatch.setattr("so101_utils.moves.hold", lambda arm, seconds: None)
    agent.detections.update({
        "boxes": [{
            "x1": pixel_x - 12, "y1": pixel_y - 12,
            "x2": pixel_x + 12, "y2": pixel_y + 12,
            "score": 88, "class_id": 3, "label": "cup",
        }],
        "resolution": [intrinsics.width, intrinsics.height],
    })
    clicked = client.post(
        "/api/detect-pick", json={"x": pixel_x + 6, "y": pixel_y - 6},
    )
    assert clicked.status_code == 200, clicked.text
    body = clicked.json()
    assert body["status"] == "picked"
    assert body["box"] == 0
    assert body["label"] == "cup"
    assert body["center"] == [pixel_x, pixel_y]
    expected = click_target(
        (pixel_x, pixel_y), intrinsics, load_hand_eye()["base_T_camera"], PICK_HEIGHT_M,
    )
    assert body["target"][0] == pytest.approx(float(expected[0]), abs=1e-6)
    assert body["target"][1] == pytest.approx(float(expected[1]), abs=1e-6)
    assert agent.arm.ee_tool_offset == pytest.approx(PICK_TOOL_OFFSET_M)

    outside = client.post("/api/detect-pick", json={"x": 0, "y": 0})
    assert outside.status_code == 400
    assert "inside a detection box" in outside.json()["detail"]
    assert client.post("/api/detect-pick", json={}).status_code == 400


def test_boxes_that_ride_the_camera_socket_never_move_the_arm(client):
    agent = _connect(client)
    pose = agent.arm.get_pose()
    payload = json.dumps({
        "data": {
            "boxes": [[1, 2, 3, 4, 92, 0]],
            "labels": ["cup"],
            "resolution": [320, 240],
        },
    }).encode()
    agent._on_camera_payload(payload)
    panel = client.get("/api/detections").json()
    assert panel["source"] == "recamera"
    assert panel["boxes"][0]["label"] == "cup"
    assert panel["resolution"] == [320, 240]
    assert agent.arm.get_pose().x == pose.x


def test_jpeg_stream_ignores_boxes(client):
    agent = _connect(client)
    image = np.full((48, 64, 3), 40, np.uint8)
    image[4:20, 8:28] = (0, 0, 220)
    _attach_camera(agent, image)
    agent._jpeg = None
    agent._cam_stop.clear()
    thread = Thread(target=agent._camera_loop)
    thread.start()
    deadline = time.monotonic() + 2
    while agent.latest_jpeg() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    before = agent.latest_jpeg()
    assert before is not None
    agent.detections.update({
        "boxes": [{
            "x1": 1, "y1": 1, "x2": 30, "y2": 20,
            "score": 99, "class_id": 1, "label": "box",
        }],
        "resolution": [64, 48],
    })
    time.sleep(0.12)
    after = agent.latest_jpeg()
    agent._cam_stop.set()
    thread.join(2)
    assert before == after
    assert client.hook.frames
    chunk = next(agent.mjpeg())
    assert chunk.startswith(b"--frame")
    assert b"image/jpeg" in chunk
    assert after in chunk


def test_calibration_routes_do_not_touch_station_files(client, monkeypatch, tmp_path):
    agent = _connect(client)
    written = {}
    monkeypatch.setattr(
        "so101_workshop.agent.write_port",
        lambda port: written.setdefault("port", port),
    )

    class _Port:
        def __init__(self, device):
            self.device = device

    devices = ["/dev/ttyACM0", "/dev/ttyUSB0"]

    def comports():
        return [_Port(device) for device in devices]

    monkeypatch.setattr("serial.tools.list_ports.comports", comports)
    marked = client.post("/api/calibrate/ports/mark")
    assert marked.status_code == 200
    assert "/dev/ttyACM0" in marked.json()["ports"]
    devices = ["/dev/ttyUSB0"]
    found = client.post("/api/calibrate/ports/vanished")
    assert found.status_code == 200
    assert found.json()["port"] == "/dev/ttyACM0"
    refused = client.post("/api/calibrate/ports/save", json={"port": ""})
    assert refused.status_code == 400
    saved = client.post("/api/calibrate/ports/save", json={"port": "/dev/ttyACM0"})
    assert saved.status_code == 200
    assert written["port"] == "/dev/ttyACM0"

    midpoint = client.post("/api/calibrate/motors/midpoint")
    assert midpoint.status_code == 400
    assert "real arm" in midpoint.json()["detail"]

    blank = client.post("/api/calibrate/intrinsics/capture")
    assert blank.status_code == 400
    agent._bgr = np.zeros((240, 320, 3), np.uint8)
    empty = client.get("/api/calibrate/intrinsics/preview")
    assert empty.status_code == 200, empty.text
    assert empty.json()["found"] is False
    board = _chessboard()
    agent._bgr = board
    preview = client.get("/api/calibrate/intrinsics/preview")
    assert preview.status_code == 200, preview.text
    assert preview.json()["found"] is True
    assert len(preview.json()["corners"]) == 88
    captured = client.post("/api/calibrate/intrinsics/capture")
    assert captured.status_code == 200, captured.text
    assert captured.json()["views"] == 1
    early = client.post("/api/calibrate/intrinsics/solve")
    assert early.status_code == 400
    assert "10" in early.json()["detail"]

    saved_intrinsics = {}
    monkeypatch.setattr(
        "so101_utils.config.save_camera_intrinsics",
        lambda intr, serial=None, path=None: saved_intrinsics.setdefault("intr", intr),
    )
    while len(agent._intrinsics_samples) < 10:
        agent._intrinsics_samples.append(agent._intrinsics_samples[0])
    solved = client.post("/api/calibrate/intrinsics/solve")
    assert solved.status_code == 200, solved.text
    assert "intr" in saved_intrinsics
    assert solved.json()["fx"] > 0
    assert client.post("/api/calibrate/intrinsics/reset").json()["views"] == 0

    untagged = client.post("/api/calibrate/hand-eye/sample")
    assert untagged.status_code == 400
    _attach_camera(agent, board)
    eye = np.eye(4)
    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def _tag(*_args, **_kwargs):
        return eye, corners

    monkeypatch.setattr("so101_workshop.agent.find_tag", _tag)
    monkeypatch.setattr("so101_utils.hand_eye_flow.time.sleep", lambda _seconds: None)
    sample = client.post("/api/calibrate/hand-eye/sample")
    assert sample.status_code == 200, sample.text
    assert sample.json()["views"] == 1
    preview = client.get("/api/calibrate/hand-eye/preview")
    assert preview.status_code == 200, preview.text
    assert preview.json()["visible"] is True
    assert preview.json()["views"] == 1
    assert preview.json()["rpy"] == [0.0, 0.0, 0.0]
    assert len(preview.json()["corners"]) == 4
    assert client.post("/api/calibrate/hand-eye/reset").json()["views"] == 0
    short = client.post("/api/calibrate/hand-eye/solve")
    assert short.status_code == 400
    assert "4" in short.json()["detail"]


def test_click_pick_follows_click_and_go_and_detect_also_needs_yolo(client):
    client.post("/api/arm/connect", json={"mock": True})
    agent = client.agent
    agent.features["yolo"] = False
    refused_detect = client.post("/api/detect-pick", json={"x": 1, "y": 1})
    assert refused_detect.status_code == 404
    assert "yolo" in refused_detect.json()["detail"]
    assert client.post("/api/pick", json={"index": 0}).status_code == 404
    # Click and pick does not read boxes, so turning YOLO off leaves it available.
    missing_camera = client.post("/api/click-pick", json={"x": 1, "y": 1})
    assert missing_camera.status_code == 400
    assert "Camera" in missing_camera.json()["detail"]

    agent.features["yolo"] = True
    agent.features["click_and_go"] = False
    for path in ("/api/click", "/api/click-pick", "/api/detect-pick"):
        blocked = client.post(path, json={"x": 1, "y": 1})
        assert blocked.status_code == 404, path
        assert "click_and_go" in blocked.json()["detail"]


def test_flags_keep_estop_and_block_the_rest(monkeypatch):
    agent = Agent()
    agent.features["teleop"] = False
    agent.features["yolo"] = False
    agent.features["click_and_go"] = False
    agent.features["estop"] = False
    app = create_app(agent, listen=False)
    with TestClient(app) as client:
        client.post("/api/arm/connect", json={"mock": True})
        assert client.post("/api/estop").status_code == 200
        assert client.post("/api/jog", json={"name": "shoulder_pan", "delta_deg": 1}).status_code == 404
        assert client.post("/api/click", json={"x": 1, "y": 1}).status_code == 404
        assert client.get("/api/detections").json()["enabled"] is False
        assert client.post("/api/pick", json={"index": 0}).status_code == 404
        instructor = client.get("/api/instructor").json()
        assert instructor["enabled"] is False
        assert instructor["detail"]


def test_detection_socket_accepts_boxes_and_cannot_move_the_arm(client):
    agent = _connect(client)
    pose = agent.arm.get_pose()
    store = agent.detections
    port = _free_port()
    bridge = DetectionBridge(store, {
        "detections": {"ws_host": "127.0.0.1", "ws_port": port, "mqtt_url": ""},
    }, client.hook)
    bridge.start()
    try:
        deadline = time.monotonic() + 2
        while not _port_open(port) and time.monotonic() < deadline:
            time.sleep(0.02)
        payload = {
            "data": {
                "boxes": [[30, 10, 10, 40, 77, 4]],
                "labels": ["block"],
                "resolution": [320, 240],
                "image": "a" * 5000,
            },
        }
        asyncio.run(_send_json(port, payload))
        deadline = time.monotonic() + 2
        while not store.snapshot()["boxes"] and time.monotonic() < deadline:
            time.sleep(0.02)
        box = store.snapshot()["boxes"][0]
        assert box["label"] == "block"
        assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (25, 0, 35, 30)
        assert client.hook.boxes
        assert agent.arm.get_pose().x == pose.x
        assert not hasattr(bridge, "arm")
    finally:
        bridge.close()
        bridge._thread.join(2)

    empty = DetectionBridge(DetectionStore(), {"detections": {"mqtt_url": "mqtt://"}}, NullHook())
    empty._start_mqtt()
    assert "no host" in empty.store.snapshot()["error"]


def test_sdk_methods_the_bus_calls_exist():
    import scservo_sdk
    packet = scservo_sdk.PacketHandler(0)
    for name in ("read2ByteTxRx", "write1ByteTxRx", "write2ByteTxRx"):
        assert hasattr(packet, name)
    assert hasattr(scservo_sdk, "GroupSyncRead")
    assert hasattr(scservo_sdk, "GroupSyncWrite")
    assert hasattr(scservo_sdk.PortHandler("/dev/null"), "setPacketTimeout")


def _chessboard():
    import cv2
    square = 28
    cols, rows = 9, 12
    board = np.full(((rows + 2) * square, (cols + 2) * square), 255, np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                y, x = (row + 1) * square, (col + 1) * square
                board[y:y + square, x:x + square] = 0
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.05)
        return sock.connect_ex(("127.0.0.1", port)) == 0


async def _send_json(port, payload):
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
        await websocket.send(json.dumps(payload))
