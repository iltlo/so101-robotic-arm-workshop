"""reCamera YOLO boxes. The JPEG preview never receives these rectangles."""

from threading import Event, Lock, Thread
import asyncio
import json
import time
from urllib.parse import urlparse


def parse_detections(payload) -> dict:
    """Normalize a reCamera model payload.

    Boxes are [centre_x, centre_y, width, height, score 0-100, class_id] in the
    frame named by `resolution`. The base64 image is ignored.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("Detection payload must be a JSON object")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("Detection payload has no data object")
    raw_boxes = data.get("boxes") or []
    labels = data.get("labels") or []
    if not isinstance(raw_boxes, list):
        raise ValueError("boxes must be a list")
    resolution = data.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        resolution = [int(resolution[0]), int(resolution[1])]
    else:
        resolution = None
    width, height = resolution if resolution else (None, None)
    boxes = []
    for index, raw in enumerate(raw_boxes):
        if not isinstance(raw, (list, tuple)) or len(raw) < 6:
            continue
        cx, cy, w, h = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        half_w, half_h = max(0.0, w) / 2.0, max(0.0, h) / 2.0
        x1, x2 = cx - half_w, cx + half_w
        y1, y2 = cy - half_h, cy + half_h
        # The model may predict past the edge for an object that is cut off.
        # Clamp so a pick target is always a pixel inside the frame.
        x1, x2 = max(0.0, x1), max(0.0, x2)
        y1, y2 = max(0.0, y1), max(0.0, y2)
        if resolution is not None:
            x1, x2 = min(x1, width), min(x2, width)
            y1, y2 = min(y1, height), min(y2, height)
        label = labels[index] if index < len(labels) else str(int(raw[5]))
        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(raw[4]),
            "class_id": int(raw[5]),
            "label": str(label),
        })
    return {"boxes": boxes, "resolution": resolution}


class DetectionStore:
    def __init__(self):
        self._lock = Lock()
        self.boxes: list[dict] = []
        self.resolution = None
        self.updated = None
        self.error = None
        self.listening = False
        self.source = None

    def update(self, parsed: dict, *, source: str | None = None) -> None:
        with self._lock:
            self.boxes = list(parsed["boxes"])
            self.resolution = parsed.get("resolution")
            self.updated = time.monotonic()
            self.error = None
            if source:
                self.source = source

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message

    def snapshot(self) -> dict:
        with self._lock:
            age = None if self.updated is None else time.monotonic() - self.updated
            return {
                "boxes": [dict(box) for box in self.boxes],
                "resolution": self.resolution,
                "age_s": age,
                "error": self.error,
                "listening": self.listening,
                "source": self.source,
            }


class DetectionBridge:
    """Listen for camera clients. This object never touches the arm."""

    def __init__(self, store: DetectionStore, settings: dict, hook):
        self.store = store
        self.hook = hook
        detections = settings.get("detections") or {}
        self.ws_host = str(detections.get("ws_host") or "0.0.0.0")
        self.ws_port = int(detections.get("ws_port") or 8765)
        self.mqtt_url = str(detections.get("mqtt_url") or "")
        self.mqtt_topic = str(detections.get("mqtt_topic") or "yolo11n_result")
        self._stop = None
        self._thread = None
        self._client = None

    def start(self) -> None:
        self._stop = Event()
        self._thread = Thread(target=self._run_ws, name="yolo-ws", daemon=True)
        self._thread.start()
        if self.mqtt_url:
            self._start_mqtt()
        self.store.listening = True

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:
                pass
        self.store.listening = False

    def _accept(self, payload, source: str = "ws") -> None:
        try:
            parsed = parse_detections(payload)
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            self.store.fail(str(exc))
            return
        self.store.update(parsed, source=source)
        try:
            self.hook.on_detections(parsed["boxes"])
        except Exception:
            pass

    def _run_ws(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self.store.fail(f"Detection socket stopped: {exc}")

    async def _serve(self) -> None:
        import websockets

        async def handler(websocket):
            async for raw in websocket:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                self._accept(raw)

        async with websockets.serve(
            handler, self.ws_host, self.ws_port, max_size=8_000_000,
        ):
            while not self._stop.is_set():
                await asyncio.sleep(0.2)

    def _start_mqtt(self) -> None:
        parsed = urlparse(self.mqtt_url)
        host = parsed.hostname
        if not host:
            self.store.fail(f"MQTT url has no host: {self.mqtt_url}")
            return
        port = parsed.port or 1883
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            self.store.fail(str(exc))
            return
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        def on_message(_client, _userdata, message):
            self._accept(message.payload.decode("utf-8", errors="replace"), source="mqtt")

        client.on_message = on_message
        try:
            client.connect_async(host, port, keepalive=15)
            client.subscribe(self.mqtt_topic)
            client.loop_start()
        except Exception as exc:
            self.store.fail(str(exc))
            return
        self._client = client
