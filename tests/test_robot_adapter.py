import sys
import types

import pytest

from so101_control.robot import connect_live


@pytest.mark.parametrize("fail_in_connect,calibrated", [(True, False), (False, False)])
def test_failed_live_connection_closes_partial_bus(monkeypatch, fail_in_connect, calibrated):
    state = {"bus_closed": False}

    class Bus:
        is_connected = False

        def disconnect(self, disable_torque_on_disconnect=True):
            state["bus_closed"] = True
            self.is_connected = False

    class Config:
        def __init__(self, **values):
            state["config"] = values

    class Follower:
        def __init__(self, config):
            self.bus = Bus()
            self.is_calibrated = calibrated

        def connect(self, calibrate=True):
            state["calibrate"] = calibrate
            self.bus.is_connected = True
            if fail_in_connect:
                raise RuntimeError("connection failed after opening bus")

        def disconnect(self):
            self.bus.disconnect()

    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.robots", types.ModuleType("lerobot.robots"))
    so_follower = types.ModuleType("lerobot.robots.so_follower")
    so_follower.SO101Follower = Follower
    so_follower.SO101FollowerConfig = Config
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", so_follower)

    with pytest.raises(RuntimeError):
        connect_live({"port": "/dev/ttyFake", "arm_id": "test"}, enable_live=True)
    assert state["config"]["use_degrees"] is True
    assert state["calibrate"] is False
    assert state["bus_closed"] is True