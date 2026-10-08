"""Local click UI only exchanges pixels and RGB frames, never robot commands."""

from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json

import numpy as np
import pytest

from so101_control.click_view import click_view


def test_local_click_ui_returns_image_and_queues_validated_pixel():
    with click_view(64, 48) as (viewer, address):
        assert address.startswith("http://127.0.0.1:")
        assert b"SO-101 RGB" in urlopen(address).read()
        viewer.publish(np.zeros((48, 64, 3), dtype=np.uint8))
        assert urlopen(address + "frame").read().startswith(b"\xff\xd8")
        viewer.report("Preview only")
        payload = json.dumps({"x": 20, "y": 11}).encode()
        request = Request(address + "click", payload, {"Content-Type": "application/json"})
        assert "Preview only" in json.load(urlopen(request))["message"]
        assert viewer.consume() == [(20, 11, 1)]
        assert viewer.consume() == []
        assert viewer.frame_seq == 1


@pytest.mark.parametrize("payload", [{"x": -1, "y": 0}, {"x": 64, "y": 0},
                                     {"x": True, "y": 2}, {"x": 1.1, "y": 2},
                                     {"x": 1, "y": 0, "unsafe": 1}])
def test_local_click_ui_rejects_bad_coordinates(payload):
    with click_view(64, 48) as (viewer, address):
        request = Request(address + "click", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 400
        assert viewer.consume() == []