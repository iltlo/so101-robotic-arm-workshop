"""Load the three workshop yaml files. Paths in yaml are relative to the repo root."""

import argparse
from pathlib import Path
import math

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from so101_utils.model import JOINTS
from so101_utils.tags import Tag
from so101_utils.transforms import Intrinsics, transform

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


def set_station(root: str | Path) -> None:
    """Point yaml reads and writes at one station directory.

    The command-line scripts leave this at the repository root. The workshop
    server calls it before opening the arm or the camera.
    """
    global ROOT, CONFIG
    ROOT = Path(root).resolve()
    CONFIG = ROOT / "config"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a mapping")
    return data


def _resolve(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def load_robot(path: str | Path | None = None) -> dict:
    """port, arm_id, urdf, joint limits, station joint offsets, and the tag mount."""
    data = _load(Path(path) if path else CONFIG / "robot.yaml")
    if "urdf" not in data or "port" not in data:
        raise ValueError("robot.yaml needs port and urdf")
    data["urdf"] = _resolve(data["urdf"])
    data["tag"] = tag_config(data)
    return data


def tag_config(config: dict) -> Tag:
    """Validate the AprilTag section of robot.yaml."""
    raw = config.get("tag")
    if not isinstance(raw, dict):
        raise ValueError("robot.yaml needs a tag section")
    try:
        family = str(raw["family"])
        tag_id = int(raw["id"])
        size_m = float(raw["size_m"])
        xyz = np.asarray(raw["mount_xyz_m"], dtype=float)
        ry = float(raw["mount_ry_deg"])
        rz = float(raw["mount_rz_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tag section in robot.yaml: {exc}") from exc
    if (not family or tag_id < 0 or not np.isfinite([size_m, ry, rz]).all()
            or size_m <= 0 or xyz.shape != (3,) or not np.isfinite(xyz).all()):
        raise ValueError("Invalid tag family, id, size, or mount")
    mount = np.eye(4)
    mount[:3, 3] = xyz
    mount[:3, :3] = (Rotation.from_euler("y", ry, degrees=True)
                     * Rotation.from_euler("z", rz, degrees=True)).as_matrix()
    return Tag(family, tag_id, size_m, transform(mount))


def joint_offsets(config: dict) -> dict[str, float]:
    """Arm zero corrections in degrees. Missing joints are 0."""
    values = config.get("joint_offsets_deg") or {}
    if not isinstance(values, dict) or set(values) - set(JOINTS):
        raise ValueError("joint_offsets_deg must contain only the five arm joints")
    offsets = {}
    for name in JOINTS:
        value = values.get(name, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 30:
            raise ValueError(f"Invalid joint offset for {name} (max ±30°)")
        offsets[name] = float(value)
    return offsets


def validated_limits(config: dict) -> dict[str, tuple[float, float]]:
    limits = config.get("joint_limits_deg") or {}
    if set(limits) != set(JOINTS):
        raise ValueError(f"joint_limits_deg must list {list(JOINTS)}")
    result = {}
    for name in JOINTS:
        pair = np.asarray(limits[name], dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
            raise ValueError(f"Invalid limits for {name}")
        result[name] = (float(pair[0]), float(pair[1]))
    return result


def load_camera(path: str | Path | None = None) -> dict:
    data = _load(Path(path) if path else CONFIG / "camera.yaml")
    if str(data.get("backend") or "realsense") == "realsense":
        # A RealSense reports its intrinsics when the colour stream opens. Any
        # copy in an older profile is ignored, so swapping a D405 for a D435
        # cannot leave the wrong focal lengths in use.
        data["intrinsics"] = None
        return data
    intr = data.get("intrinsics") or {}
    data["intrinsics"] = Intrinsics(
        int(data["width"]),
        int(data["height"]),
        float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]),
        tuple(intr.get("distortion") or ()),
        str(intr.get("distortion_model") or "brown_conrady"),
    )
    return data


def save_camera_intrinsics(intr: Intrinsics, *, serial=None, path: str | Path | None = None) -> None:
    """Refresh resolution and intrinsics from the live camera.

    `path` writes the active camera profile. It defaults to camera.yaml, which is
    what the command-line scripts use.
    """
    path = Path(path) if path else CONFIG / "camera.yaml"
    current = _load(path)
    if str(current.get("backend") or "realsense") == "realsense":
        raise ValueError("RealSense intrinsics come from the device; nothing to save")
    current["serial"] = serial if serial is not None else current.get("serial")
    current["width"] = int(intr.width)
    current["height"] = int(intr.height)
    current["intrinsics"] = {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.cx), "cy": float(intr.cy),
        "distortion": [float(v) for v in intr.distortion],
        "distortion_model": intr.distortion_model,
    }
    path.write_text("# RGB camera. A profile camera gets its intrinsics from calibration/intrinsics.py.\n"
                    + yaml.safe_dump(current, sort_keys=False))


def update_workshop(**values) -> dict:
    """Merge top-level keys into workshop.yaml and return the result.

    Used for the active camera profile. The file is small and edited by hand,
    so it is rewritten with a short header rather than patched line by line.
    """
    path = CONFIG / "workshop.yaml"
    data = _load(path) if path.is_file() else {}
    data.update(values)
    path.write_text(
        "# Workshop flags. The command-line scripts ignore this file.\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    return data


def load_hand_eye() -> dict:
    data = _load(CONFIG / "hand_eye.yaml")
    if "base_T_camera" not in data:
        raise ValueError("hand_eye.yaml has no base_T_camera; run calibration/hand_eye.py")
    data["base_T_camera"] = transform(data["base_T_camera"])
    return data


def save_hand_eye(base_T_camera, *, residual_m: float, residual_deg: float) -> None:
    payload = {
        "base_T_camera": np.asarray(base_T_camera, dtype=float).tolist(),
        "max_residual_m": float(residual_m),
        "max_residual_deg": float(residual_deg),
    }
    text = ("# Fixed camera in the robot base frame. Metres.\n"
            "# p_base = base_T_camera @ p_camera\n")
    (CONFIG / "hand_eye.yaml").write_text(text + yaml.safe_dump(payload, sort_keys=False))


def arm_parser(description: str) -> argparse.ArgumentParser:
    """Scripts require either --mock or --live. The description keeps its line breaks."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="run the in-memory arm")
    group.add_argument("--live", action="store_true", help="connect to the follower")
    return parser
