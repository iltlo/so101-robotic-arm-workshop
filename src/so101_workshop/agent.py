"""Local workshop process. The arm, the one camera client, and e-stop live here."""

from importlib.util import find_spec
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread, current_thread
import json
import socket
import sys
import time

import yaml
from scipy.spatial.transform import Rotation

from so101_control.arm import SO101
from so101_control.teleop import MAX_STEP_M, MIN_STEP_M, TELEOP_SPEED_DEG_S, Teleop
from so101_utils.assets import assets_ready, ensure_assets
from so101_utils import config as station_config
from so101_utils.auto_hand_eye import DWELL_S, calibration_poses
from so101_utils.camera import camera_backend, list_video_devices
from so101_utils.click_app import box_centre, click_target, select_box
from so101_utils.config import (
    load_camera,
    load_hand_eye,
    load_robot,
    validated_limits,
)
from so101_utils.hand_eye_flow import record_sample
from so101_utils.intrinsics import WORKSHOP_MIN_VIEWS, capture_chessboard, find_chessboard, solve_captured
from so101_utils.moves import (
    APPROACH_M,
    CLICK_TOOL_OFFSET_M,
    HOVER_M,
    PICK_HEIGHT_M,
    PICK_TOOL_OFFSET_M,
    go_to,
    pick_and_return,
)
from so101_utils.ports import get_ports, port_that_vanished, write_port
from so101_utils.tags import find_tag
from so101_utils.transforms import MIN_POSES, Sample, solve_eye_to_hand
from so101_workshop.detections import (
    DetectionBridge,
    DetectionStore,
    parse_detections,
)
from so101_workshop.hooks import NullHook

_FEATURES = ("connect", "estop", "teleop", "click_and_go", "calibration", "yolo", "instructor")
_VIEWER_PERIOD_S = 0.15
_VIEWER_CAMERA_PERIOD_S = 0.5
# Status polls every 800 ms. Profiles and video devices only need to be read
# every few seconds; the hardware page refreshes them on demand.
CATALOG_TTL_S = 2.5


def _sample_xyz(sample) -> dict:
    pose = sample.base_T_gripper
    return {"x": float(pose[0, 3]), "y": float(pose[1, 3]), "z": float(pose[2, 3])}


def load_settings() -> dict:
    path = station_config.CONFIG / "workshop.yaml"
    if not path.is_file():
        data = {}
    else:
        data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("workshop.yaml must be a mapping")
    features = {name: True for name in _FEATURES}
    features["instructor"] = False
    raw = data.get("features") or {}
    if isinstance(raw, dict):
        for name in _FEATURES:
            if name in raw:
                features[name] = bool(raw[name])
    data["features"] = features
    data.setdefault("camera_host", "192.168.42.1")
    data.setdefault("detections", {})
    return data


class _Reporter:
    def __init__(self, agent, token: int):
        self.agent = agent
        self.token = token

    def report(self, message: str) -> None:
        self.agent.message = message

    def has_pending(self) -> bool:
        return self.agent.estop_event.is_set() or self.token != self.agent._move_token


class Agent:
    def __init__(self, hook=None):
        self.hook = hook or NullHook()
        self.settings = load_settings()
        self.features = self.settings["features"]
        self.arm = None
        self.teleop = None
        self.cam = None
        self.message = "Not connected"
        self.camera_error = None
        self.camera_profile = self.settings.get("camera_profile") or None
        self.camera_config_used = None
        self.last_probe = None
        self.estop_event = Event()
        self._estop_lock = Lock()
        self._estop_applied = False
        self.detections = DetectionStore()
        self._bridge = None
        self._queue: Queue = Queue()
        self._stop = Event()
        self._worker = None
        self._cam_stop = Event()
        self._cam_thread = None
        self._cam_cm = None
        self._camera_connecting = False
        self._jpeg = None
        self._bgr = None
        self._frame_size = None
        self._media = Lock()
        self._move_token = 0
        self._ports_before = None
        self._intrinsics_samples = []
        self._hand_samples = []
        self._auto_lock = Lock()
        self._auto_run = None
        self._auto_cancel = Event()
        self._auto_thread = None
        self._closed = False
        self.viewer = None
        self.viewer_allowed = True
        self.viewer_error = None
        self.viewer_mode = "arm"
        self._viewer_started = False
        self._viewer_starting = False
        self._viewer_lock = Lock()
        # Every Rerun draw takes this lock. Motion callbacks and the viewer
        # ticker run on different threads, and the SDK sink is not re-entrant.
        self._viewer_draw_lock = RLock()
        self._viewer_thread = None
        self._viewer_tick_thread = None
        self._viewer_tick = 0.0
        self._viewer_camera_tick = 0.0
        self._tag_detector = None
        self._tag_lock = Lock()
        self._catalog_lock = Lock()
        self._catalog_at = 0.0
        self._catalog: tuple[list[dict], list[int]] = ([], [])

    def start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = Thread(target=self._loop, name="workshop-motion", daemon=True)
            self._worker.start()
        # The viewer must keep drawing while a long job owns the motion worker,
        # such as the automatic hand-eye tour. Its own thread never moves the arm.
        if self._viewer_tick_thread is None or not self._viewer_tick_thread.is_alive():
            self._viewer_tick_thread = Thread(
                target=self._viewer_loop, name="workshop-viewer", daemon=True,
            )
            self._viewer_tick_thread.start()

    def start_detectors(self) -> None:
        if not self.features.get("yolo", True) or self._bridge is not None:
            return
        self._bridge = DetectionBridge(self.detections, self.settings, self.hook)
        self._bridge.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._cam_stop.set()
        self._auto_cancel.set()
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        # A viewer whose start is still blocked in the first gRPC flush has no
        # handle here yet. Stop every process this station started anyway.
        try:
            from so101_workshop.viewer import stop_all
            stop_all()
        except Exception:
            pass
        if self._bridge is not None:
            self._bridge.close()
        if self._worker is not None:
            self._worker.join(timeout=2)
        self.disconnect_camera()
        if self.arm is not None:
            try:
                self.arm.close()
            except Exception:
                pass
            self.arm = None

    def estop(self) -> None:
        """Latch e-stop before torque-off. Works with no camera and no network."""
        # Latch first, so motion stops even while the serial call is pending.
        self.estop_event.set()
        # The page sends e-stop on the socket and over HTTP at the same time.
        # Only the first caller may touch the serial port; the other returns
        # immediately instead of queueing a second blocking torque-off.
        if self._estop_applied:
            self.message = "E-stop. Torque off."
            return
        with self._estop_lock:
            if self._estop_applied:
                self.message = "E-stop. Torque off."
                return
            self._estop_applied = True
            try:
                if self.arm is not None:
                    self.arm.relax()
            finally:
                self._clear_teleop()
        self.message = "E-stop. Torque off."

    def _clear_teleop(self) -> None:
        if self.teleop is not None:
            self.teleop.reset()

    def torque_on(self):
        if self.arm is None:
            raise RuntimeError("Arm is not connected")

        def job():
            self.estop_event.clear()
            self._estop_applied = False
            self.arm.hold()
            self._clear_teleop()
            self.message = "Torque on"

        return self.submit(job)

    def connect_arm(self, *, mock: bool = False, port: str | None = None,
                    require_calibration: bool = True) -> str:
        if self._auto_running():
            raise RuntimeError("Stop auto calibration before reconnecting the arm")
        if port and not mock:
            write_port(port)
        self._ensure_urdf()
        if self.arm is not None:
            self.arm.close()
            self.arm = None
        self.arm = SO101.connect(
            mock=mock, ee_tool_offset=CLICK_TOOL_OFFSET_M,
            require_calibration=require_calibration,
        )
        self.arm.set_speed(TELEOP_SPEED_DEG_S)
        self.teleop = Teleop(self.arm, cancel=self.estop_event.is_set)
        self.estop_event.clear()
        self._estop_applied = False
        self._attach_viewer()
        self.message = "Mock arm connected" if mock else f"Arm connected on {load_robot()['port']}"
        return self.message

    def disconnect_arm(self) -> None:
        if self._auto_running():
            raise RuntimeError("Stop auto calibration before disconnecting the arm")
        if self.arm is not None:
            self.arm.close()
        self.arm = None
        self.teleop = None
        self.message = "Arm disconnected"

    def camera_path(self) -> Path:
        """The yaml the camera uses. camera.yaml is the editable default."""
        name = self.camera_profile
        if name:
            candidate = Path(station_config.CONFIG) / name
            if (Path(name).name == name and name.startswith("camera")
                    and name.endswith(".yaml") and candidate.is_file()):
                return candidate
        return Path(station_config.CONFIG) / "camera.yaml"

    def camera_cfg(self) -> dict:
        return load_camera(self.camera_path())

    def camera_profiles(self) -> list[dict]:
        """Every camera*.yaml in the station, with availability for this machine."""
        active = self.camera_path()
        profiles = []
        for path in sorted(Path(station_config.CONFIG).glob("camera*.yaml")):
            try:
                cfg = load_camera(path)
                backend = camera_backend(cfg)
                available, reason = self._camera_available(backend, cfg)
                profiles.append({
                    "name": path.name,
                    "label": path.stem.removeprefix("camera_").replace("_", " ").title()
                             or "Active camera",
                    "backend": backend,
                    "device": cfg.get("device"),
                    "url": cfg.get("url"),
                    "serial": cfg.get("serial"),
                    "width": cfg.get("width"),
                    "height": cfg.get("height"),
                    "fps": cfg.get("fps"),
                    "available": available,
                    "reason": reason,
                    "active": path == active,
                })
            except Exception as exc:
                profiles.append({
                    "name": path.name, "label": path.stem, "available": False,
                    "reason": str(exc), "active": False,
                })
        return profiles

    def camera_catalog(self, *, fresh: bool = False) -> tuple[list[dict], list[int]]:
        """Profiles and video devices, read at most once per TTL.

        The status route runs this on every poll. Re-reading six YAML files and
        scanning /dev every 800 ms competes with arm and camera I/O.
        """
        now = time.monotonic()
        with self._catalog_lock:
            if not fresh and now - self._catalog_at < CATALOG_TTL_S:
                return self._catalog
        catalog = (self.camera_profiles(), list_video_devices())
        with self._catalog_lock:
            self._catalog_at = time.monotonic()
            self._catalog = catalog
        return catalog

    def _camera_available(self, backend: str, cfg: dict) -> tuple[bool, str | None]:
        if backend == "realsense" and find_spec("pyrealsense2") is None:
            return False, "pyrealsense2 is not installed (vision extra)"
        if backend == "webcamera" and sys.platform.startswith("linux"):
            device = cfg.get("device")
            if isinstance(device, int) and not Path(f"/dev/video{device}").exists():
                return False, f"/dev/video{device} is not present"
        return True, None

    def select_camera_profile(self, name: str | None) -> str | None:
        """Persist the active profile. `None` or camera.yaml means the default file."""
        text = str(name or "").strip()
        profile = None
        if text and text != "camera.yaml":
            candidate = Path(text)
            if (candidate.name != text or not text.startswith("camera")
                    or not text.endswith(".yaml")):
                raise ValueError("Unknown camera profile")
            if not (Path(station_config.CONFIG) / text).is_file():
                raise ValueError(f"No camera profile {text}")
            profile = text
        self.camera_profile = profile
        with self._catalog_lock:
            self._catalog_at = 0.0
        try:
            station_config.update_workshop(camera_profile=profile)
        except OSError as exc:
            self.message = f"Camera selected, but the profile was not saved: {exc}"
        self.message = f"Camera profile {text or 'camera.yaml'}"
        return profile

    def connect_camera(self, profile: str | None = None, overrides: dict | None = None) -> str:
        if profile is not None:
            self.select_camera_profile(profile)
        if self.cam is not None or self._camera_connecting:
            return self.message
        cfg = self.camera_cfg()
        for key in ("device", "url", "serial", "width", "height", "fps"):
            value = (overrides or {}).get(key)
            if value is not None and value != "":
                cfg[key] = value
        try:
            cfg["width"] = int(cfg["width"])
            cfg["height"] = int(cfg["height"])
            cfg["fps"] = int(cfg["fps"])
            if cfg.get("device") is not None and cfg.get("device") != "":
                cfg["device"] = int(cfg["device"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid camera width, height, fps, or device: {exc}") from exc
        self.camera_config_used = {
            key: cfg.get(key) for key in
            ("backend", "device", "url", "serial", "width", "height", "fps")
        }
        self._camera_connecting = True
        self.camera_error = None
        self.message = "Opening the camera."
        Thread(target=self._open_camera, args=(cfg,), name="camera-open", daemon=True).start()
        return self.message

    def disconnect_camera(self) -> None:
        if not self._closed and self._auto_running():
            raise RuntimeError("Stop auto calibration before disconnecting the camera")
        self._cam_stop.set()
        thread = self._cam_thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2)
        cm = self._cam_cm
        self._cam_cm = None
        self.cam = None
        self._cam_thread = None
        self._camera_connecting = False
        if cm is not None:
            cm.__exit__(None, None, None)

    def probe(self, host: str | None = None) -> dict:
        host = host or str(self.settings.get("camera_host") or "192.168.42.1")
        ports = {}
        for port in (80, 8090):
            sock = socket.socket()
            sock.settimeout(0.4)
            try:
                sock.connect((host, port))
                ports[str(port)] = True
            except OSError:
                ports[str(port)] = False
            finally:
                sock.close()
        self.last_probe = {"host": host, "ports": ports}
        return self.last_probe

    def press(self, token: str) -> None:
        if self.teleop is None:
            raise RuntimeError("Arm is not connected")
        if self._auto_running():
            raise RuntimeError("Auto calibration is running")
        self.teleop.press_token(str(token))

    def release(self, token: str) -> None:
        if self.teleop is None:
            raise RuntimeError("Arm is not connected")
        self.teleop.release_token(str(token))

    def tap(self, token: str):
        self._require_motion()
        token = str(token)

        def job():
            self._require_motion()
            self.teleop.press_token(token)
            self.teleop.release_token(token)
            self.teleop.poll()
            if token == "h":
                self.message = "Home queued"
            elif token == "r":
                self.message = "Rest queued"
            elif token == "space":
                self.message = "Gripper closed" if self.teleop.closed else "Gripper open"
            elif token in {"+", "="}:
                self.message = f"Step {self.teleop.step * 1000:.0f} mm"
            elif token == "-":
                self.message = f"Step {self.teleop.step * 1000:.0f} mm"
            elif token == "p":
                self.message = repr(self.arm.get_pose())
            return self.message

        return self.submit(job)

    def set_teleop(self, *, step_m: float | None = None,
                   speed_deg_s: float | None = None) -> dict:
        if self.teleop is None:
            raise RuntimeError("Arm is not connected")
        if step_m is not None:
            step = float(step_m)
            if not MIN_STEP_M <= step <= MAX_STEP_M:
                raise ValueError(
                    f"Step must be {MIN_STEP_M * 1000:.0f} to {MAX_STEP_M * 1000:.0f} mm"
                )
            self.teleop.step = step
        if speed_deg_s is not None:
            self.arm.set_speed(float(speed_deg_s))
        self.message = (
            f"Teleop step {self.teleop.step * 1000:.0f} mm, "
            f"joint speed {self.arm.speed:.0f} deg/s"
        )
        return {"step_m": self.teleop.step, "speed_deg_s": self.arm.speed}

    def jog(self, name: str, delta_deg: float):
        self._require_motion()

        def job():
            self._require_motion()
            self.arm.move_joint(name, delta_deg)
            self.message = f"Jogged {name} by {float(delta_deg):.1f} deg"
            return self.message

        return self.submit(job)

    def click(self, x: int, y: int):
        return self._move_to_pixel(
            x, y, tool_offset=CLICK_TOOL_OFFSET_M, height_m=HOVER_M,
            move=go_to, done="arrived",
        )

    def pick_box(self, index: int):
        self._require_motion()
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        snapshot = self.detections.snapshot()
        boxes = snapshot["boxes"]
        if index < 0 or index >= len(boxes):
            raise ValueError("That detection is no longer in the list")
        centre = box_centre(boxes[index], snapshot["resolution"], self.cam.intrinsics)
        return self._pick_pixel(*centre)

    def click_pick(self, x: int, y: int) -> dict:
        """Pick at the clicked table point, then return to rest and open."""
        self._require_motion()
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        return self._pick_pixel(int(x), int(y))

    def detect_pick(self, x: int, y: int) -> dict:
        """Pick the centre of the detection box under the click."""
        self._require_motion()
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        snapshot = self.detections.snapshot()
        boxes = snapshot["boxes"]
        if not boxes:
            raise ValueError("No detection boxes yet")
        index, box, centre = select_box(
            (int(x), int(y)), boxes, snapshot["resolution"], self.cam.intrinsics,
        )
        result = dict(self._pick_pixel(*centre))
        result.update({"box": index, "label": box["label"], "center": list(centre)})
        return result

    def mark_ports(self) -> list[str]:
        self._ports_before = get_ports()
        return sorted(self._ports_before)

    def vanished_port(self) -> str:
        if self._ports_before is None:
            raise RuntimeError("Mark the plugged-in ports first")
        return port_that_vanished(self._ports_before)

    def save_port(self, port: str) -> str:
        if not port.strip():
            raise ValueError("Find the removed port first")
        write_port(port)
        self.message = f"Saved port {port}"
        return port

    def motor_midpoint(self) -> dict:
        bus = self._bus()
        return self.submit(bus.set_midpoint)

    def motor_range_start(self) -> str:
        bus = self._bus()
        self.submit(bus.start_range_capture)
        self.message = "Move every joint except wrist roll through its range, including the gripper"
        return self.message

    def motor_range_stop(self) -> dict:
        bus = self._bus()
        saved = self.submit(bus.stop_range_capture)
        self.message = "Motor ranges saved"
        return saved

    def capture_intrinsics(self) -> int:
        frame = self.latest_bgr()
        if frame is None:
            raise RuntimeError("Camera image is not ready")
        self._intrinsics_samples.append(capture_chessboard(frame))
        self.message = f"Captured {len(self._intrinsics_samples)} chessboard views"
        return len(self._intrinsics_samples)

    def intrinsics_preview(self) -> dict:
        """Chessboard corners in the newest frame, without capturing it."""
        frame = self.latest_bgr()
        if frame is None:
            raise RuntimeError("Camera image is not ready")
        height, width = frame.shape[:2]
        found = find_chessboard(frame)
        if found is None:
            return {"found": False, "size": [int(width), int(height)]}
        corners, _pattern = found
        return {
            "found": True,
            "size": [int(width), int(height)],
            "corners": [[float(a), float(b)] for a, b in corners.reshape(-1, 2)],
        }

    def solve_intrinsics(self) -> dict:
        frame = self.latest_bgr()
        if frame is None:
            raise RuntimeError("Camera image is not ready")
        height, width = frame.shape[:2]
        intr, rms = solve_captured(
            self._intrinsics_samples, (width, height), minimum=WORKSHOP_MIN_VIEWS,
        )
        station_config.save_camera_intrinsics(intr, serial=None, path=self.camera_path())
        self.message = f"Saved intrinsics, RMS {rms:.3f} px"
        return {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "rms": rms}

    def reset_intrinsics(self) -> int:
        self._intrinsics_samples = []
        self.message = "Chessboard captures cleared"
        return 0

    def capture_hand_eye(self):
        if self.arm is None:
            raise RuntimeError("Arm is not connected")
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        if self._auto_running():
            raise RuntimeError("Auto calibration is running")

        def job():
            seen = self._see_tag()

            def read_tag():
                return self._see_tag()

            sample = record_sample(self.arm, seen, read_tag)
            self._hand_samples.append(sample)
            self.message = f"Captured {len(self._hand_samples)} hand-eye poses"
            return len(self._hand_samples)

        return self.submit(job)

    def hand_eye_preview(self) -> dict:
        """Whether the configured tag is visible right now, without moving."""
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        frame = self.latest_bgr()
        if frame is None:
            raise RuntimeError("Camera image is not ready")
        tag = load_robot()["tag"]
        try:
            pose, corners = self._detect_tag(frame)
        except ValueError as exc:
            return {
                "visible": False,
                "tag_id": tag.id,
                "reason": str(exc),
                "views": len(self._hand_samples),
            }
        xyz = pose[:3, 3]
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return {
            "visible": True,
            "tag_id": tag.id,
            "translation": [float(value) for value in xyz],
            "rpy": [float(value) for value in rpy],
            "distance_m": float((xyz ** 2).sum() ** 0.5),
            "corners": [[float(a), float(b)] for a, b in corners],
            "views": len(self._hand_samples),
        }

    def reset_hand_eye(self) -> int:
        if self._auto_running():
            raise RuntimeError("Stop auto calibration before clearing poses")
        self._hand_samples = []
        self.message = "Hand-eye poses cleared"
        return 0

    def solve_hand_eye(self) -> dict:
        if self._auto_running():
            raise RuntimeError("Stop auto calibration before saving")
        tag = load_robot()["tag"]
        solved = solve_eye_to_hand(self._hand_samples, tag.gripper_T_tag)
        station_config.save_hand_eye(
            solved["base_T_camera"],
            residual_m=solved["max_residual_m"],
            residual_deg=solved["max_residual_deg"],
        )
        self.message = (
            f"Saved hand-eye. Residual {solved['max_residual_m'] * 1000:.1f} mm, "
            f"{solved['max_residual_deg']:.1f} deg."
        )
        return {
            "max_residual_m": solved["max_residual_m"],
            "max_residual_deg": solved["max_residual_deg"],
        }

    def auto_hand_eye_status(self) -> dict:
        with self._auto_lock:
            if self._auto_run is None:
                return {
                    "running": False, "step": 0, "total": len(calibration_poses()),
                    "captured": 0, "message": "", "error": None, "cancelled": False,
                    "residual_m": None, "residual_deg": None,
                }
            return dict(self._auto_run)

    def start_auto_hand_eye(self) -> dict:
        """Queue the fixed-pose tour and return before the arm starts moving."""
        if self.arm is None:
            raise RuntimeError("Arm is not connected")
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        if not self.arm.torque_on:
            raise RuntimeError("Turn torque on before auto calibration")
        if self.estop_event.is_set():
            raise RuntimeError("E-stop is on. Turn torque on before moving.")
        if not self._calibrated():
            raise RuntimeError("Calibrate the motors before moving.")
        if self.latest_bgr() is None:
            raise RuntimeError("Camera image is not ready")
        with self._auto_lock:
            if self._auto_run is not None and self._auto_run["running"]:
                raise RuntimeError("Auto calibration is already running")
            self._auto_cancel.clear()
            self._auto_run = {
                "running": True, "step": 0, "total": len(calibration_poses()),
                "captured": 0, "message": "Starting the auto hand-eye tour",
                "error": None, "cancelled": False,
                "residual_m": None, "residual_deg": None,
            }
        self._clear_teleop()
        self._auto_thread = Thread(
            target=self._auto_hand_eye_thread, name="auto-hand-eye", daemon=True,
        )
        self._auto_thread.start()
        self.message = "Auto hand-eye calibration started"
        return self.auto_hand_eye_status()

    def cancel_auto_hand_eye(self) -> dict:
        if self._auto_running():
            self._auto_cancel.set()
            self._auto_update(message="Stopping auto calibration")
            self.message = "Stopping auto calibration"
        return self.auto_hand_eye_status()

    def _auto_running(self) -> bool:
        with self._auto_lock:
            return bool(self._auto_run is not None and self._auto_run["running"])

    def _auto_update(self, **fields) -> None:
        with self._auto_lock:
            if self._auto_run is not None:
                self._auto_run.update(fields)

    def _finish_auto_hand_eye(self, **fields) -> None:
        with self._auto_lock:
            if self._auto_run is not None:
                self._auto_run.update(fields, running=False)

    def _auto_hand_eye_thread(self) -> None:
        try:
            # The tour owns the motion worker for its whole duration. The HTTP
            # route returns first, and status polling carries the progress.
            self.submit(self._auto_hand_eye_job, timeout=900)
        except Exception as exc:
            self._finish_auto_hand_eye(
                error=str(exc), message=f"Auto calibration failed: {exc}",
            )
            self.message = f"Auto calibration failed: {exc}"

    def _auto_hand_eye_job(self) -> None:
        samples = []
        try:
            if self._auto_tour(samples):
                solved = solve_eye_to_hand(samples, load_robot()["tag"].gripper_T_tag)
                station_config.save_hand_eye(
                    solved["base_T_camera"],
                    residual_m=solved["max_residual_m"],
                    residual_deg=solved["max_residual_deg"],
                )
                self._hand_samples = samples
                self.message = (
                    f"Auto hand-eye saved. Residual "
                    f"{solved['max_residual_m'] * 1000:.1f} mm, "
                    f"{solved['max_residual_deg']:.1f} deg."
                )
                self._finish_auto_hand_eye(
                    captured=len(samples),
                    residual_m=solved["max_residual_m"],
                    residual_deg=solved["max_residual_deg"],
                    message=self.message,
                )
            else:
                reason = ("E-stop stopped auto calibration" if self.estop_event.is_set()
                          else "Auto calibration stopped")
                self._finish_auto_hand_eye(cancelled=True, message=reason)
                self.message = reason
        except Exception as exc:
            self._finish_auto_hand_eye(
                error=str(exc), message=f"Auto calibration failed: {exc}",
            )
            self.message = f"Auto calibration failed: {exc}"
        finally:
            if self.arm is not None and not self.estop_event.is_set():
                try:
                    self.arm.rest(wait=False)
                except Exception:
                    pass

    def _auto_tour(self, samples: list[Sample]) -> bool:
        """Move through every fixed pose. False means cancelled or e-stopped."""
        poses = calibration_poses()
        if self._auto_stopped():
            return False
        self._auto_update(message="Folding to rest before the tour")
        if not self.arm.rest(wait=True, cancel=self._auto_stopped):
            return False
        for index, pose in enumerate(poses, start=1):
            if self._auto_stopped():
                return False
            self._auto_update(
                step=index, message=f"Pose {index} of {len(poses)}: moving",
            )
            self.arm.move_to(*pose, wait=False)
            self._auto_update(
                message=f"Pose {index} of {len(poses)}: settling",
            )
            if not self._auto_dwell(DWELL_S):
                return False
            sample = self._auto_sample()
            if sample is None:
                self._auto_update(
                    captured=len(samples),
                    message=f"Pose {index} of {len(poses)}: no tag in view, skipped",
                )
                continue
            samples.append(sample)
            self._auto_update(
                captured=len(samples),
                message=f"Pose {index} of {len(poses)}: {len(samples)} samples",
            )
        return True

    def _auto_sample(self) -> Sample | None:
        if self.arm is None:
            raise RuntimeError("Arm is disconnected")
        frame = self.latest_bgr()
        if frame is None:
            raise RuntimeError("Camera image is not ready")
        try:
            camera_T_tag, _corners = self._detect_tag(frame)
        except ValueError:
            return None
        joints = self.arm.get_joints()
        return Sample(self.arm.gripper_link_pose(joints), camera_T_tag)

    def _auto_stopped(self) -> bool:
        return self._auto_cancel.is_set() or self.estop_event.is_set()

    def _auto_dwell(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._auto_stopped():
                return False
            time.sleep(0.05)
        return True

    def _detect_tag(self, frame):
        """camera_T_tag and corners. One detector call at a time."""
        cam = self.cam
        if cam is None:
            raise RuntimeError("Camera is not connected")
        tag = load_robot()["tag"]
        with self._tag_lock:
            if self._tag_detector is None:
                from so101_utils.tags import tag_detector
                self._tag_detector = tag_detector(tag)
            return find_tag(frame, cam.intrinsics, tag, detector=self._tag_detector)

    def status(self) -> dict:
        pose = None
        joints = None
        limits = None
        arm = {
            "connected": self.arm is not None,
            "mock": bool(self.arm and self.arm.mock),
            "torque_on": bool(self.arm and self.arm.torque_on),
            "calibrated": self._calibrated(),
            "port": None,
            "pose": None,
            "joints": None,
            "limits": None,
            "speed_deg_s": None,
        }
        if self.arm is not None:
            try:
                arm["port"] = None if self.arm.mock else load_robot().get("port")
                # One serial read feeds both the pose and the joint table.
                joints = self.arm.get_joints()
                current = self.arm.pose_at(joints)
                pose = {
                    "x": current.x, "y": current.y, "z": current.z,
                    "roll": current.roll, "pitch": current.pitch, "yaw": current.yaw,
                }
                limits = {
                    name: [float(low), float(high)]
                    for name, (low, high) in validated_limits(load_robot()).items()
                }
            except Exception as exc:
                arm["error"] = str(exc)
            arm["pose"] = pose
            arm["joints"] = joints
            arm["limits"] = limits
            try:
                arm["speed_deg_s"] = self.arm.speed
            except Exception:
                pass
        width = height = None
        intrinsics = None
        if self.cam is not None:
            width = self.cam.intrinsics.width
            height = self.cam.intrinsics.height
            intrinsics = {
                "fx": self.cam.intrinsics.fx, "fy": self.cam.intrinsics.fy,
                "cx": self.cam.intrinsics.cx, "cy": self.cam.intrinsics.cy,
                "distortion_model": self.cam.intrinsics.distortion_model,
            }
        else:
            try:
                configured = self.camera_cfg()
                width = configured["width"]
                height = configured["height"]
                intr = configured["intrinsics"]
                intrinsics = {
                    "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                    "distortion_model": intr.distortion_model,
                }
            except Exception:
                pass
        try:
            saved_hand_eye = load_hand_eye()
            hand_eye_file = {
                "saved": True,
                "max_residual_m": saved_hand_eye.get("max_residual_m"),
                "max_residual_deg": saved_hand_eye.get("max_residual_deg"),
            }
        except Exception:
            hand_eye_file = {"saved": False}
        profiles, devices = self.camera_catalog()
        body = {
            "arm": arm,
            "camera": {
                "connected": self.cam is not None,
                "connecting": self._camera_connecting,
                "error": self.camera_error,
                "width": width,
                "height": height,
                "intrinsics": intrinsics,
                "profile": self.camera_path().name,
                "config": self.camera_config_used,
                "presets": profiles,
                "devices": devices,
                "realsense_available": find_spec("pyrealsense2") is not None,
            },
            "teleop": {
                "step_m": None if self.teleop is None else self.teleop.step,
                "roll_step_deg": None if self.teleop is None else self.teleop.roll_step,
                "pitch_step_deg": None if self.teleop is None else self.teleop.pitch_step,
                "gripper_closed": None if self.teleop is None else self.teleop.closed,
            },
            "hand_eye": hand_eye_file,
            "auto_hand_eye": self.auto_hand_eye_status(),
            "probe": self.last_probe,
            "estop": self.estop_event.is_set(),
            "message": self.message,
            "features": dict(self.features),
            "instructor": self.instructor_status(),
            "intrinsics_views": len(self._intrinsics_samples),
            "hand_eye_views": len(self._hand_samples),
            "hand_eye_samples": [_sample_xyz(sample) for sample in self._hand_samples],
            "detections": self.detections.snapshot(),
            "viewer": self.viewer_status(),
            "motion": {
                "hover_m": HOVER_M,
                "pick_height_m": PICK_HEIGHT_M,
                "approach_m": APPROACH_M,
            },
        }
        try:
            self.hook.on_status(body)
        except Exception:
            pass
        return body

    def instructor_status(self) -> dict:
        return {
            "enabled": False,
            "configured": bool(self.features.get("instructor")),
            "detail": "No outbound publisher is registered",
        }

    def viewer_status(self) -> dict:
        from so101_workshop.viewer import rerun_binary

        starting = self._viewer_starting
        running = self.viewer is not None and self.viewer.running()
        url = self.viewer.url if running else None
        available = find_spec("rerun") is not None and rerun_binary() is not None
        if not self.viewer_allowed:
            detail = "Rerun was disabled with --no-viewer."
        elif url:
            detail = "Rerun is showing the arm blueprint."
        elif starting:
            detail = "Starting the Rerun viewer."
        elif self.viewer is not None:
            detail = "The Rerun viewer stopped. Click Start Rerun."
        elif self.viewer_error:
            detail = self.viewer_error
        elif not available:
            detail = "The rerun package is not installed in this environment."
        else:
            detail = "Rerun is not running yet."
        return {
            "enabled": url is not None,
            "available": available,
            "allowed": self.viewer_allowed,
            "starting": starting,
            "url": url,
            "mode": self.viewer_mode,
            "detail": detail,
        }

    def set_viewer_mode(self, mode: str) -> str:
        if mode not in ("arm", "hand-eye"):
            raise ValueError("Viewer mode must be arm or hand-eye")
        self.viewer_mode = mode
        return mode

    def start_viewer(self) -> dict:
        """Begin opening Rerun in the background and return immediately.

        The first blueprint flush can stall for many seconds while the viewer
        process starts. The workshop page must stay reachable during that time,
        so the blocking work never runs on a server thread.
        """
        if not self.viewer_allowed:
            raise RuntimeError("Rerun was disabled with --no-viewer")
        with self._viewer_lock:
            if self._viewer_starting:
                return self.viewer_status()
            if self.viewer is not None and self.viewer.running():
                return self.viewer_status()
            if self.viewer is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None
            self._viewer_starting = True
        self._viewer_thread = Thread(target=self._open_viewer, name="rerun-start", daemon=True)
        self._viewer_thread.start()
        return self.viewer_status()

    def _open_viewer(self) -> None:
        try:
            self._ensure_urdf()
            from so101_workshop.viewer import open_viewer
            urdf = Path(station_config.ROOT) / "assets" / "SO101" / "so101_new_calib.urdf"
            viewer = open_viewer(urdf, self.camera_cfg()["intrinsics"])
            if self._closed:
                viewer.close()
                return
            self.viewer = viewer
            self.viewer_error = None
            self._viewer_started = True
            self._attach_viewer()
        except Exception as exc:
            self.viewer = None
            self._viewer_started = False
            self.viewer_error = str(exc)
        finally:
            with self._viewer_lock:
                self._viewer_starting = False

    def _attach_viewer(self) -> None:
        with self._viewer_draw_lock:
            if self.viewer is None or self.arm is None:
                return
            twin = self.viewer.twin
            if self.arm.ee_tool_offset:
                twin.mark_ee_tool(self.arm.ee_tool_offset)
            self.arm.track(self._draw_viewer)
            self.arm.on_target(self._draw_goal)

    def _viewer_live(self) -> bool:
        viewer = self.viewer
        return viewer is not None and viewer.twin.alive and viewer.running()

    def _draw_viewer(self) -> None:
        with self._viewer_draw_lock:
            if not self._viewer_live():
                if self.viewer is not None:
                    self._detach_viewer()
                return
            self.viewer.twin.show(self.arm)

    def _draw_goal(self, pose) -> None:
        with self._viewer_draw_lock:
            if not self._viewer_live():
                if self.viewer is not None:
                    self._detach_viewer()
                return
            self.viewer.twin.show_goal_pose(pose)

    def _detach_viewer(self) -> None:
        with self._viewer_draw_lock:
            viewer = self.viewer
            if viewer is None:
                return
            self.viewer = None
            self.viewer_error = "The Rerun viewer stopped. Click Start Rerun."
            try:
                viewer.close()
            except Exception:
                pass

    def latest_jpeg(self):
        with self._media:
            return self._jpeg

    def latest_bgr(self):
        with self._media:
            if self._bgr is None:
                return None
            return self._bgr.copy()

    def mjpeg(self):
        while not self._stop.is_set():
            frame = self.latest_jpeg()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.05)

    def submit(self, fn, timeout: float = 120):
        if self._closed:
            raise RuntimeError("Workshop is closed")
        from concurrent.futures import Future
        future: Future = Future()
        self._queue.put((fn, future))
        return future.result(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                fn, future = self._queue.get(timeout=0.02)
            except Empty:
                if not self.estop_event.is_set():
                    self._idle_teleop()
                continue
            try:
                future.set_result(fn())
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)

    def _idle_teleop(self) -> None:
        if self.teleop is None:
            return
        try:
            self.teleop.poll()
        except Exception as exc:
            self.message = str(exc)

    def _viewer_loop(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self._refresh_viewer()
            except Exception:
                pass

    def _refresh_viewer(self) -> None:
        with self._viewer_draw_lock:
            if self.viewer is None:
                return
            if not self.viewer.running():
                # Drop the dead gRPC sink before another draw reaches it, or each
                # flush prints "gRPC connection severed" until the server stops.
                self._detach_viewer()
                return
            arm = self.arm
            if arm is None:
                return
            now = time.monotonic()
            if now - self._viewer_tick < _VIEWER_PERIOD_S:
                return
            self._viewer_tick = now
            try:
                twin = self.viewer.twin
                frame = self.latest_bgr()
                if frame is not None and self.viewer_mode == "hand-eye":
                    self._log_hand_eye(twin, frame, arm)
                else:
                    twin.show(arm)
                    if (frame is not None and twin.intrinsics is not None
                            and now - self._viewer_camera_tick >= _VIEWER_CAMERA_PERIOD_S):
                        self._viewer_camera_tick = now
                        twin.log_camera(frame)
                        twin.flush()
            except Exception:
                return

    def _log_hand_eye(self, twin, frame, arm) -> None:
        from so101_utils.tags import tag_detector
        from so101_utils.views import log_hand_eye
        tag = load_robot()["tag"]
        try:
            base_T_camera = load_hand_eye()["base_T_camera"]
        except Exception:
            base_T_camera = None
        # Preview requests share this detector, so detection is serialised.
        with self._tag_lock:
            if self._tag_detector is None:
                self._tag_detector = tag_detector(tag)
            log_hand_eye(
                twin, frame, arm, tag, base_T_camera,
                detector=self._tag_detector,
            )

    def _open_camera(self, cfg: dict) -> None:
        from so101_utils.camera import open_camera
        try:
            context = open_camera(cfg, on_payload=self._on_camera_payload)
            cam = context.__enter__()
        except Exception as exc:
            self.camera_error = str(exc)
            self.message = str(exc)
            self._camera_connecting = False
            return
        self._cam_cm = context
        self.cam = cam
        self._cam_stop = Event()
        self._cam_thread = Thread(target=self._camera_loop, name="workshop-camera", daemon=True)
        self._cam_thread.start()
        self.camera_error = None
        self.message = "Camera connected"
        self._camera_connecting = False

    def _on_camera_payload(self, raw) -> None:
        """YOLO boxes that arrive with the same reCamera JPEG frame."""
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            payload = json.loads(text)
        except (UnicodeDecodeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        data = payload.get("data", payload)
        if not isinstance(data, dict) or "boxes" not in data:
            return
        try:
            parsed = parse_detections(payload)
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        self.detections.update(parsed, source="recamera")
        try:
            self.hook.on_detections(parsed["boxes"])
        except Exception:
            pass

    def _camera_loop(self) -> None:
        import cv2
        while not self._cam_stop.is_set() and not self._stop.is_set():
            try:
                frame = self.cam.poll() if self.cam is not None else None
            except Exception as exc:
                self.camera_error = str(exc)
                time.sleep(0.2)
                continue
            if frame is None:
                time.sleep(0.03)
                continue
            image = frame.rgb_bgr
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not ok:
                continue
            jpeg = encoded.tobytes()
            with self._media:
                self._jpeg = jpeg
                self._bgr = image
                self._frame_size = (int(image.shape[1]), int(image.shape[0]))
            try:
                self.hook.on_frame(jpeg)
            except Exception:
                pass
            time.sleep(0.03)

    def _pick_pixel(self, x: int, y: int):
        return self._move_to_pixel(
            x, y, tool_offset=PICK_TOOL_OFFSET_M, height_m=PICK_HEIGHT_M,
            move=pick_and_return, done="picked",
        )

    def _move_to_pixel(self, x: int, y: int, *, tool_offset, height_m, move, done):
        """Send one click ray to the table. A newer click replaces this one."""
        self._require_motion()
        if self.cam is None:
            raise RuntimeError("Camera is not connected")
        token = self._bump()
        pixel = (int(x), int(y))

        def job():
            if token != self._move_token:
                return {"status": "replaced"}
            self._require_motion()
            self._use_tool(tool_offset)
            self._check_pixel(pixel)
            hand = load_hand_eye()
            target = click_target(
                pixel, self.cam.intrinsics, hand["base_T_camera"], height_m,
            )
            arrived = move(self.arm, _Reporter(self, token), target)
            return {
                "status": done if arrived else "replaced",
                "target": [float(value) for value in target],
            }

        return self.submit(job)

    def _bump(self) -> int:
        self._move_token += 1
        return self._move_token

    def _require_motion(self) -> None:
        if self.arm is None:
            raise RuntimeError("Arm is not connected")
        if self._auto_running():
            raise RuntimeError("Auto calibration is running")
        if self.estop_event.is_set():
            raise RuntimeError("E-stop is on. Turn torque on before moving.")
        if not self._calibrated():
            raise RuntimeError("Calibrate the motors before moving.")

    def _calibrated(self) -> bool:
        if self.arm is None:
            return False
        if self.arm.mock:
            return True
        follower = getattr(self.arm._backend, "robot", None)
        if follower is None:
            return False
        return bool(getattr(follower, "is_calibrated", False))

    def _bus(self):
        if self.arm is None or self.arm.mock:
            raise RuntimeError("Connect the real arm to calibrate motors")
        return self.arm._backend.bus

    def _use_tool(self, metres: float) -> None:
        self.arm.ee_tool_offset = float(metres)
        if self.arm._kin is not None:
            self.arm._kin.ee_tool_offset = float(metres)

    def _check_pixel(self, pixel) -> None:
        size = self._frame_size
        if size is None:
            raise RuntimeError("Camera image is not ready")
        x, y = pixel
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            raise ValueError("Pixel outside the image")

    def _see_tag(self):
        frame = self.latest_bgr()
        if frame is None or self.cam is None:
            return None
        try:
            pose, _corners = self._detect_tag(frame)
        except (ValueError, RuntimeError):
            return None
        return pose

    def _ensure_urdf(self) -> None:
        dest = Path(station_config.ROOT) / "assets" / "SO101"
        if not assets_ready(dest):
            ensure_assets(dest)
