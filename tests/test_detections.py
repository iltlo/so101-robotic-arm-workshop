"""reCamera model payloads become a box list. The image field is dropped."""

from types import SimpleNamespace

import pytest

from so101_utils.click_app import select_box
from so101_workshop.detections import parse_detections


def _box(x1, y1, x2, y2, label="cup"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 90, "class_id": 1, "label": label}


def test_parse_reads_centre_based_boxes_and_drops_the_image():
    parsed = parse_detections({
        "code": 0,
        "data": {
            "boxes": [[100, 120, 60, 40, 70, 0]],
            "labels": ["person"],
            "resolution": [640, 640],
            "image": "not-kept",
            "perf": [[0, 38, 11]],
        },
    })
    assert "image" not in parsed
    assert parsed["resolution"] == [640, 640]
    box = parsed["boxes"][0]
    assert box["label"] == "person"
    assert box["score"] == 70
    assert box["class_id"] == 0
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (70, 100, 130, 140)


def test_parse_clamps_a_box_that_runs_past_the_frame():
    parsed = parse_detections({
        "boxes": [[600, 700, 100, 100, 80, 2]],
        "labels": ["keyboard"],
        "resolution": [640, 480],
    })
    box = parsed["boxes"][0]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (550, 480, 640, 480)


def test_parse_accepts_a_json_string():
    parsed = parse_detections('{"boxes": [[1, 2, 3, 4, 80, 2]], "labels": []}')
    assert parsed["boxes"][0]["label"] == "2"
    assert parsed["boxes"][0]["class_id"] == 2
    assert (parsed["boxes"][0]["x1"], parsed["boxes"][0]["y2"]) == (0, 4)


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
    index, _, centre = select_box((250, 180), boxes, [500, 250], intrinsics)
    assert index == 0
    assert centre == (300, 200)


def test_select_box_needs_a_click_inside_a_box():
    intrinsics = SimpleNamespace(width=640, height=480)
    with pytest.raises(ValueError, match="inside a detection box"):
        select_box((5, 5), [_box(100, 100, 200, 200)], [640, 480], intrinsics)
