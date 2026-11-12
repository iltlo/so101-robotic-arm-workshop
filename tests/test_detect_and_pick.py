"""The detect-and-pick scripts run the local YOLO model."""

import importlib.util
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "application"


def _load(name):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), APP / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_demo_model_is_in_the_checkout():
    module = _load("detect_and_pick.py")
    assert module.DEFAULT_MODEL.is_file()
    assert module.DEFAULT_MODEL.stat().st_size > 1_000_000


def test_the_workshop_pick_parameters_are_editable_constants():
    module = _load("detect_and_pick.py")
    assert 0.0 < module.APPROACH_HEIGHT_OFFSET_M <= 0.1
    assert module.PICK_HEIGHT_OFFSET_M >= 0.0
    assert module.GRIP_CLOSE_DELAY_S > 0.0
    assert module.GRIP_OPEN_DELAY_S > 0.0
    assert f"{module.APPROACH_HEIGHT_OFFSET_M * 1000:.0f} mm" in module.PAGE_TEXT
