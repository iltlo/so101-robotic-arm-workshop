"""Detection boxes for the page. The camera and YOLO never touch the arm."""

from threading import Lock
import time


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
