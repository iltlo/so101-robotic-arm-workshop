"""Station configuration. Missing physical measurements intentionally fail closed."""

from pathlib import Path
import yaml
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_JOINTS = (*JOINTS, "gripper")


def load_station(path: str | Path) -> dict:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Station config must be a mapping")
    if "urdf" in data:
        data["urdf"] = str((path.parent.parent / data["urdf"]).resolve())
    return data


def validated_limits(config: dict, joints=JOINTS) -> dict[str, tuple[float, float]]:
    limits = config.get("joint_limits_deg") or {}
    if set(limits) != set(joints):
        raise ValueError(f"Provide measured joint_limits_deg for all {list(joints)}")
    result = {}
    for name in joints:
        pair = np.asarray(limits[name], dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
            raise ValueError(f"Invalid limits for {name}")
        result[name] = (float(pair[0]), float(pair[1]))
    return result


def transform(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("Invalid transform rotation")
    if np.linalg.det(matrix[:3, :3]) < 0.999:
        raise ValueError("Reflection is not a valid rotation")
    return matrix