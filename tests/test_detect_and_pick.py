"""The detect-and-pick script turns reCamera payloads into pickable boxes."""

import importlib.util
import json
from pathlib import Path

from so101_workshop.detections import DetectionStore

_SCRIPT = Path(__file__).resolve().parents[1] / "application" / "detect_and_pick.py"


def _load():
    spec = importlib.util.spec_from_file_location("detect_and_pick", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_payloads_become_boxes_without_an_image():
    module = _load()
    store = DetectionStore()
    raw = json.dumps({
        "data": {
            "boxes": [[10, 20, 30, 40, 77, 1]],
            "labels": ["cup"],
            "resolution": [640, 640],
            "image": "not-kept",
        },
    }).encode()
    module._store_payload(store, raw)
    snapshot = store.snapshot()
    assert snapshot["source"] == "recamera"
    assert snapshot["resolution"] == [640, 640]
    assert snapshot["boxes"][0]["label"] == "cup"


def test_bad_camera_payloads_are_ignored():
    module = _load()
    store = DetectionStore()
    module._store_payload(store, b"{not json")
    module._store_payload(store, b"[]")
    assert store.snapshot()["boxes"] == []
