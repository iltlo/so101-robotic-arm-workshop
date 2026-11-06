"""Loopback HTTP and WebSocket API for the workshop page."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from so101_control.motion import MotionError
from so101_workshop.agent import Agent

STATIC = Path(__file__).resolve().parent / "static"
_ERRORS = (MotionError, ValueError, RuntimeError, TimeoutError, OSError, TypeError)


def create_app(agent: Agent | None = None, *, listen: bool = False, viewer: bool = False) -> FastAPI:
    agent = agent or Agent()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        agent.viewer_allowed = viewer
        agent.start_worker()
        if viewer:
            agent.start_viewer()
        if listen:
            agent.start_detectors()
        yield
        agent.close()

    app = FastAPI(title="SO-101 workshop", lifespan=lifespan)
    app.state.agent = agent

    def call(fn):
        try:
            return fn()
        except _ERRORS as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def require(feature: str) -> None:
        if feature != "estop" and not agent.features.get(feature, True):
            raise HTTPException(status_code=404, detail=f"{feature} is disabled")

    def pixels(body: dict) -> tuple[int, int]:
        if body.get("x") is None or body.get("y") is None:
            raise HTTPException(status_code=400, detail="Click needs x and y pixels")
        try:
            return int(body["x"]), int(body["y"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Click needs x and y pixels") from exc

    def pixel_move(body: dict, features: tuple[str, ...], method):
        for feature in features:
            require(feature)
        x, y = pixels(body)
        return call(lambda: method(x, y))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/video.mjpg")
    def video():
        return StreamingResponse(
            agent.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def status():
        return agent.status()

    @app.get("/api/ports")
    def ports():
        require("connect")
        from so101_utils.ports import get_ports
        return {"ports": sorted(get_ports())}

    @app.post("/api/probe")
    def probe(body: dict = Body(default_factory=dict)):
        return agent.probe(body.get("host"))

    @app.post("/api/arm/connect")
    def connect_arm(body: dict = Body(default_factory=dict)):
        require("connect")
        return call(lambda: {
            "message": agent.connect_arm(
                mock=bool(body.get("mock")),
                port=body.get("port") or None,
                require_calibration=bool(body.get("require_calibration", True)),
            ),
        })

    @app.post("/api/arm/disconnect")
    def disconnect_arm():
        agent.disconnect_arm()
        return {"message": agent.message}

    @app.get("/api/cameras")
    def cameras():
        require("connect")
        profiles, devices = agent.camera_catalog(fresh=True)
        return {
            "profiles": profiles,
            "active": agent.camera_path().name,
            "devices": list(devices),
        }

    @app.post("/api/cameras/select")
    def select_camera(body: dict = Body(default_factory=dict)):
        require("connect")
        return call(lambda: {
            "profile": agent.select_camera_profile(body.get("profile")),
            "message": agent.message,
        })

    @app.post("/api/camera/connect")
    def connect_camera(body: dict = Body(default_factory=dict)):
        require("connect")
        overrides = {
            key: body[key] for key in ("device", "url", "serial", "width", "height", "fps")
            if body.get(key) is not None and body.get(key) != ""
        }
        return call(lambda: {
            "message": agent.connect_camera(
                profile=body.get("profile") or None,
                overrides=overrides or None,
            ),
        })

    @app.post("/api/camera/disconnect")
    def disconnect_camera():
        agent.disconnect_camera()
        return {"message": "Camera disconnected"}

    @app.post("/api/estop")
    def estop():
        agent.estop()
        return {"estop": True, "message": agent.message}

    @app.post("/api/hold")
    def hold():
        return call(lambda: {"message": agent.torque_on()})

    @app.post("/api/teleop")
    def teleop(body: dict):
        require("teleop")
        action = body.get("action")
        token = str(body.get("token") or "")

        def run():
            if action == "press":
                agent.press(token)
            elif action == "release":
                agent.release(token)
            elif action == "tap":
                agent.tap(token)
            else:
                raise ValueError("Unknown teleop action")
            return {"ok": True, "message": agent.message}

        return call(run)

    @app.post("/api/jog")
    def jog(body: dict):
        require("teleop")
        return call(lambda: {"message": agent.jog(str(body.get("name")), float(body.get("delta_deg")))})

    @app.post("/api/teleop/config")
    def teleop_config(body: dict = Body(default_factory=dict)):
        require("teleop")
        step = body.get("step_m")
        speed = body.get("speed_deg_s")
        return call(lambda: agent.set_teleop(
            step_m=None if step is None else float(step),
            speed_deg_s=None if speed is None else float(speed),
        ))

    @app.post("/api/click")
    def click(body: dict):
        return pixel_move(body, ("click_and_go",), agent.click)

    @app.post("/api/click-pick")
    def click_pick(body: dict):
        return pixel_move(body, ("click_and_go",), agent.click_pick)

    @app.get("/api/detections")
    def detections():
        if not agent.features.get("yolo", True):
            return {"enabled": False, "boxes": []}
        snapshot = agent.detections.snapshot()
        snapshot["enabled"] = True
        return snapshot

    @app.post("/api/pick")
    def pick(body: dict):
        require("yolo")
        return call(lambda: agent.pick_box(int(body["index"])))

    @app.post("/api/detect-pick")
    def detect_pick(body: dict):
        return pixel_move(body, ("click_and_go", "yolo"), agent.detect_pick)

    @app.post("/api/calibrate/ports/mark")
    def mark_ports():
        require("calibration")
        return call(lambda: {"ports": agent.mark_ports()})

    @app.post("/api/calibrate/ports/vanished")
    def vanished():
        require("calibration")
        return call(lambda: {"port": agent.vanished_port()})

    @app.post("/api/calibrate/ports/save")
    def save_port(body: dict):
        require("calibration")
        return call(lambda: {"port": agent.save_port(str(body.get("port")))})

    @app.post("/api/calibrate/motors/midpoint")
    def midpoint():
        require("calibration")
        return call(agent.motor_midpoint)

    @app.post("/api/calibrate/motors/range/start")
    def range_start():
        require("calibration")
        return call(lambda: {"message": agent.motor_range_start()})

    @app.post("/api/calibrate/motors/range/stop")
    def range_stop():
        require("calibration")
        return call(agent.motor_range_stop)

    @app.post("/api/calibrate/intrinsics/capture")
    def intrinsics_capture():
        require("calibration")
        return call(lambda: {"views": agent.capture_intrinsics()})

    @app.get("/api/calibrate/intrinsics/preview")
    def intrinsics_preview():
        require("calibration")
        return call(agent.intrinsics_preview)

    @app.post("/api/calibrate/intrinsics/solve")
    def intrinsics_solve():
        require("calibration")
        return call(agent.solve_intrinsics)

    @app.post("/api/calibrate/intrinsics/reset")
    def intrinsics_reset():
        require("calibration")
        return call(lambda: {"views": agent.reset_intrinsics()})

    @app.post("/api/calibrate/hand-eye/sample")
    def hand_sample():
        require("calibration")
        return call(lambda: {"views": agent.capture_hand_eye()})

    @app.get("/api/calibrate/hand-eye/preview")
    def hand_preview():
        require("calibration")
        return call(agent.hand_eye_preview)

    @app.post("/api/calibrate/hand-eye/reset")
    def hand_reset():
        require("calibration")
        return call(lambda: {"views": agent.reset_hand_eye()})

    @app.post("/api/calibrate/hand-eye/solve")
    def hand_solve():
        require("calibration")
        return call(agent.solve_hand_eye)

    @app.post("/api/calibrate/hand-eye/auto")
    def hand_auto():
        require("calibration")
        return call(agent.start_auto_hand_eye)

    @app.post("/api/calibrate/hand-eye/auto/cancel")
    def hand_auto_cancel():
        require("calibration")
        return call(agent.cancel_auto_hand_eye)

    @app.get("/api/viewer")
    def viewer_state():
        return agent.viewer_status()

    @app.post("/api/viewer/start")
    def viewer_start():
        return call(agent.start_viewer)

    @app.post("/api/viewer/mode")
    def viewer_mode(body: dict):
        return call(lambda: {"mode": agent.set_viewer_mode(str(body.get("mode") or ""))})

    @app.get("/api/instructor")
    def instructor():
        return agent.instructor_status()

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                data = await socket.receive_json()
                kind = data.get("type")
                try:
                    if kind == "estop":
                        # relax() waits on the serial lock. Run it off the event
                        # loop, or status and the page freeze until it returns.
                        await asyncio.to_thread(agent.estop)
                    elif kind == "press" and agent.features.get("teleop", True):
                        agent.press(str(data.get("token") or ""))
                    elif kind == "release" and agent.features.get("teleop", True):
                        agent.release(str(data.get("token") or ""))
                    elif kind == "status":
                        await socket.send_json(await asyncio.to_thread(agent.status))
                        continue
                    else:
                        continue
                except _ERRORS as exc:
                    await socket.send_json({"type": "error", "detail": str(exc)})
                    continue
                await socket.send_json({"type": "ok", "message": agent.message, "estop": agent.estop_event.is_set()})
        except WebSocketDisconnect:
            return

    return app
