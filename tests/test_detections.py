"""YOLO OBB results become the box list the page and pick scripts read."""

from types import SimpleNamespace

import numpy as np
import pytest

from so101_utils.click_app import box_centre, select_box
from so101_workshop.detections import DetectionStore
from so101_workshop.yolo import draw_boxes, obb_boxes


class _Obb:
    def __init__(self, polygons, scores, classes):
        self.xyxyxyxy = np.asarray(polygons, dtype=float)
        self.conf = np.asarray(scores, dtype=float)
        self.cls = np.asarray(classes, dtype=float)

    def __len__(self):
        return len(self.conf)


def _result(polygons=(), scores=(), classes=()):
    return SimpleNamespace(obb=_Obb(polygons, scores, classes))


def _box(x1, y1, x2, y2, label="cup"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 90, "class_id": 1, "label": label}


def test_obb_result_becomes_an_axis_aligned_box():
    parsed = obb_boxes(
        _result([[[10, 5], [20, 10], [15, 20], [5, 15]]], [0.5], [3]),
        {3: "screw"}, 640, 480,
    )
    assert parsed["resolution"] == [640, 480]
    box = parsed["boxes"][0]
    assert box["label"] == "screw"
    assert box["class_id"] == 3
    assert box["score"] == pytest.approx(50.0)
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (5, 5, 20, 20)
    assert box["polygon"][0] == [10.0, 5.0]


def test_obb_box_is_clamped_to_the_frame():
    parsed = obb_boxes(
        _result([[[-10, -5], [700, -5], [700, 500], [-10, 500]]], [0.9], [0]),
        ["screw"], 640, 480,
    )
    box = parsed["boxes"][0]
    assert box["label"] == "screw"
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (0, 0, 640, 480)


def test_empty_obb_is_an_empty_box_list():
    parsed = obb_boxes(_result(), {}, 640, 480)
    assert parsed == {"boxes": [], "resolution": [640, 480]}


def test_draw_boxes_leaves_the_source_frame_alone():
    image = np.zeros((48, 64, 3), np.uint8)
    boxes = [{
        "x1": 8, "y1": 8, "x2": 30, "y2": 30,
        "score": 90, "class_id": 0, "label": "screw",
    }]
    drawn = draw_boxes(image, boxes)
    assert drawn is not image
    assert not image.any()
    assert drawn.any()


def test_detection_store_tracks_the_latest_source():
    store = DetectionStore()
    store.update({"boxes": [{"label": "screw"}], "resolution": [64, 48]}, source="yolo")
    snapshot = store.snapshot()
    assert snapshot["source"] == "yolo"
    assert snapshot["resolution"] == [64, 48]
    assert snapshot["boxes"][0]["label"] == "screw"
    assert snapshot["age_s"] is not None
    store.fail("model missing")
    assert "model" in store.snapshot()["error"]
    store.update({"boxes": [], "resolution": None}, source="yolo")
    assert store.snapshot()["error"] is None


def test_select_box_prefers_the_smallest_box_under_the_click():
    intrinsics = SimpleNamespace(width=640, height=480)
    boxes = [_box(0, 0, 600, 400, "table"), _box(100, 100, 200, 200, "cup")]
    index, box, centre = select_box((150, 150), boxes, [640, 480], intrinsics)
    assert index == 1
    assert box["label"] == "cup"
    assert centre == (150, 150)


def test_select_box_scales_the_detection_resolution_to_the_frame():
    intrinsics = SimpleNamespace(width=1000, height=500)
    boxes = [_box(100, 50, 200, 150)]
    index, box, centre = select_box((250, 180), boxes, [500, 250], intrinsics)
    assert index == 0
    assert centre == (300, 200)
    assert box_centre(box, [500, 250], intrinsics) == centre


def test_select_box_needs_a_click_inside_a_box():
    intrinsics = SimpleNamespace(width=640, height=480)
    with pytest.raises(ValueError, match="inside a detection box"):
        select_box((5, 5), [_box(100, 100, 200, 200)], [640, 480], intrinsics)
