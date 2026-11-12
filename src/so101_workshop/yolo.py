"""Local YOLO OBB inference from workshop camera frames.

The detector runs on its own thread so the camera loop can keep streaming.
Only one frame waits in the queue; a slower model drops stale frames.
"""

from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread, current_thread

import numpy as np

from so101_workshop.detections import DetectionStore

DEFAULT_CONFIDENCE = 0.25


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _label(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def obb_boxes(result, names, width: int, height: int) -> dict:
    """Convert one Ultralytics OBB result into the box format the page reads."""
    resolution = [int(width), int(height)]
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return {"boxes": [], "resolution": resolution}
    polygons = _numpy(obb.xyxyxyxy)
    scores = _numpy(obb.conf).reshape(-1)
    classes = _numpy(obb.cls).reshape(-1)
    boxes = []
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=float).reshape(4, 2)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        x1, x2 = min(max(0.0, x1), width), min(max(0.0, x2), width)
        y1, y2 = min(max(0.0, y1), height), min(max(0.0, y2), height)
        class_id = int(classes[index]) if index < len(classes) else 0
        score = float(scores[index]) if index < len(scores) else 0.0
        boxes.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "score": score * 100.0,
            "class_id": class_id,
            "label": _label(names, class_id),
            "polygon": [[float(x), float(y)] for x, y in points],
        })
    return {"boxes": boxes, "resolution": resolution}


def draw_boxes(image, boxes):
    """Copy of `image` with oriented boxes and labels drawn on it."""
    import cv2

    drawn = image.copy()
    for box in boxes:
        polygon = box.get("polygon")
        if polygon:
            points = np.asarray(polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(drawn, [points], True, (0, 220, 120), 2)
            anchor = (int(points[0, 0, 0]), int(points[0, 0, 1]) - 8)
        else:
            anchor = (int(box["x1"]), int(box["y1"]) - 8)
            cv2.rectangle(
                drawn, (int(box["x1"]), int(box["y1"])), (int(box["x2"]), int(box["y2"])),
                (0, 220, 120), 2,
            )
        cv2.putText(
            drawn, f"{box['label']} {float(box['score']):.0f}", anchor,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 1, cv2.LINE_AA,
        )
    return drawn


class YoloDetector:
    """An Ultralytics OBB model, loaded on first use."""

    def __init__(self, model_path, *, confidence: float = DEFAULT_CONFIDENCE):
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self._model = None
        self._names = {}

    def load(self) -> "YoloDetector":
        if self._model is not None:
            return self
        if not self.model_path.is_file():
            raise RuntimeError(f"YOLO model not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; install the workshop dependencies"
            ) from exc
        self._model = YOLO(str(self.model_path), verbose=False)
        self._names = getattr(self._model, "names", {}) or {}
        return self

    def detect(self, image) -> dict:
        self.load()
        results = self._model.predict(
            image, task="obb", conf=self.confidence, verbose=False,
        )
        height, width = image.shape[:2]
        if not results:
            return {"boxes": [], "resolution": [int(width), int(height)]}
        return obb_boxes(results[0], self._names, width, height)


class YoloWorker:
    """Detect on frames submitted by the one camera loop."""

    def __init__(self, store: DetectionStore, hook, *, model_path,
                 confidence: float = DEFAULT_CONFIDENCE):
        self.store = store
        self.hook = hook
        self.detector = YoloDetector(model_path, confidence=confidence)
        self._queue: Queue = Queue(maxsize=1)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        self.store.listening = True
        self._thread = Thread(target=self._run, name="workshop-yolo", daemon=True)
        self._thread.start()

    def submit(self, image) -> None:
        """Queue one frame. If the model is behind, the older frame is dropped."""
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(image)
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(image)
        except Full:
            pass

    def close(self) -> None:
        self._stop.set()
        self._wake()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2)
        self.store.listening = False

    def _wake(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass

    def _run(self) -> None:
        try:
            self.detector.load()
        except Exception as exc:
            self.store.fail(str(exc))
            self.store.listening = False
            return
        while not self._stop.is_set():
            try:
                image = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if image is None:
                break
            try:
                parsed = self.detector.detect(image)
            except Exception as exc:
                self.store.fail(str(exc))
                continue
            self.store.update(parsed, source="yolo")
            try:
                self.hook.on_detections(parsed["boxes"])
            except Exception:
                pass
