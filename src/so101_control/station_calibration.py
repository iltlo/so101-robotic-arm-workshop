"""Accepted per-station calibration persistence and camera identity checks."""

from pathlib import Path
import numpy as np
import yaml

from .config import transform


def calibration_file(config_path):
    return Path(config_path).resolve().parent / "calibration.yaml"


def load_calibration(config_path, *, require_plane=False):
    filename = calibration_file(config_path)
    data = yaml.safe_load(filename.read_text())
    if not isinstance(data, dict) or data.get("schema") != 1 or not data.get("accepted"):
        raise ValueError("Accepted station calibration required")
    transform(data["base_T_camera"])
    if require_plane and "plane" not in data:
        raise ValueError("Table-plane calibration required")
    return data


def save_calibration(config_path, data):
    calibration_file(config_path).write_text(yaml.safe_dump(data, sort_keys=False))


def intrinsics_data(intr):
    return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                cx=intr.cx, cy=intr.cy, distortion=list(intr.distortion),
                distortion_model=intr.distortion_model)


def matches_camera(data, intr, config):
    if data.get("arm_id") != config.get("arm_id") or data.get("urdf") != config.get("urdf"):
        raise ValueError("Calibration belongs to another arm or URDF")
    if data.get("camera_serial") != config.get("camera_serial"):
        raise ValueError("Camera serial differs from calibration")
    old = data.get("intrinsics", {})
    for key, value in intrinsics_data(intr).items():
        saved = old.get(key)
        if saved is None or (saved != value if isinstance(value, str) else
                      np.shape(saved) != np.shape(value) or not np.allclose(saved, value, atol=1e-4)):
            raise ValueError(f"Camera {key} differs from calibrated image")