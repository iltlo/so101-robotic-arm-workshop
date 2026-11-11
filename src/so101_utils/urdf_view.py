"""Log the SO-101 URDF into Rerun 0.26.

Rerun 0.27 dropped Intel Mac wheels, and `UrdfTree` arrived only in 0.28.
0.26.2 is the last release with wheels for Intel macOS, Apple silicon, Linux,
and Windows. Joint motion there is a `Transform3D` on the child link, and the
link's meshes hang under that entity so they move with it.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class _Visual:
    origin: np.ndarray
    mesh: Path
    color: tuple[int, int, int, int] | None


@dataclass
class UrdfJoint:
    """One URDF joint. `entity` is the child link in the Rerun hierarchy."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit_lower: float
    limit_upper: float
    entity: str = ""

    def compute_transform(self, angle: float, clamp: bool = False):
        """Child link pose in the parent link, as a Rerun transform."""
        import rerun as rr

        pose = self.origin
        if self.joint_type == "revolute":
            value = float(angle)
            if not np.isfinite(value):
                raise ValueError(f"Joint {self.name} angle is not finite")
            if clamp:
                value = min(self.limit_upper, max(self.limit_lower, value))
            rotation = Rotation.from_rotvec(self.axis * value).as_matrix()
            turned = np.eye(4)
            turned[:3, :3] = rotation
            pose = self.origin @ turned
        elif self.joint_type != "fixed":
            raise ValueError(f"Joint {self.name} has unsupported type {self.joint_type!r}")
        return rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3])


class UrdfScene:
    """Visual meshes and joints for one URDF file."""

    def __init__(self, name: str, joints: tuple[UrdfJoint, ...],
                 visuals: dict[str, tuple[_Visual, ...]], links: dict[str, str]):
        self.name = name
        self._joints = {joint.name: joint for joint in joints}
        self._visuals = visuals
        self._links = links

    @classmethod
    def from_file_path(cls, path: str | Path) -> "UrdfScene":
        urdf = Path(path).resolve()
        root = ET.parse(urdf).getroot()
        name = root.get("name") or urdf.stem
        materials = _materials(root)
        visuals: dict[str, list[_Visual]] = {}
        for link in root.findall("link"):
            link_name = link.get("name") or ""
            if not link_name:
                raise ValueError("URDF link is missing a name")
            visuals[link_name] = _link_visuals(link, urdf.parent, materials)
        joints = [_parse_joint(element) for element in root.findall("joint")]
        links = _link_paths(name, visuals, joints)
        for joint in joints:
            joint.entity = links[joint.child]
        return cls(name, tuple(joints), {key: tuple(value) for key, value in visuals.items()}, links)

    def get_joint_by_name(self, name: str) -> UrdfJoint | None:
        return self._joints.get(name)

    def link_entity(self, link: str) -> str:
        try:
            return self._links[link]
        except KeyError:
            raise ValueError(f"URDF has no link {link}") from None

    def log_to_recording(self, rr) -> None:
        """Log meshes once, then park every joint at the URDF zero pose."""
        for link, visuals in self._visuals.items():
            parent = self._links[link]
            for index, visual in enumerate(visuals):
                entity = f"{parent}/visual_{index}"
                mesh = rr.Asset3D(path=visual.mesh)
                if visual.color is not None:
                    mesh = rr.Asset3D(path=visual.mesh, albedo_factor=visual.color)
                rr.log(
                    entity,
                    rr.Transform3D(translation=visual.origin[:3, 3], mat3x3=visual.origin[:3, :3]),
                    mesh,
                    static=True,
                )
        rr.set_time("tick", sequence=0)
        for joint in self._joints.values():
            # A fixed joint never gets another sample. Static keeps it in place
            # after the time cursor moves on; a revolute joint is updated live.
            transform = joint.compute_transform(0.0)
            if joint.joint_type == "fixed":
                rr.log(joint.entity, transform, static=True)
            else:
                rr.log(joint.entity, transform)


def _materials(root) -> dict[str, tuple[int, int, int, int]]:
    found = {}
    for element in root.findall("material"):
        name = element.get("name")
        color = element.find("color")
        if name and color is not None:
            found[name] = _rgba(color.get("rgba"))
    return found


def _rgba(text: str | None) -> tuple[int, int, int, int]:
    parts = [float(part) for part in (text or "").split()]
    if len(parts) != 4:
        raise ValueError(f"URDF color needs 4 numbers, got {text!r}")
    return tuple(int(round(min(1.0, max(0.0, part)) * 255)) for part in parts)


def _vec3(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3)
    values = np.array([float(part) for part in text.split()], dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 numbers, got {text!r}")
    return values


def _origin(element) -> np.ndarray:
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = Rotation.from_euler("xyz", _vec3(element.get("rpy"))).as_matrix()
    matrix[:3, 3] = _vec3(element.get("xyz"))
    return matrix


def _link_visuals(link, urdf_dir: Path, materials) -> list[_Visual]:
    visuals = []
    for element in link.findall("visual"):
        geometry = element.find("geometry")
        mesh = None if geometry is None else geometry.find("mesh")
        if mesh is None:
            raise ValueError(f"Link {link.get('name')} has a visual without a mesh")
        filename = mesh.get("filename")
        if not filename:
            raise ValueError(f"Link {link.get('name')} mesh is missing a filename")
        path = (urdf_dir / filename).resolve()
        if not path.is_file():
            raise ValueError(f"URDF mesh not found: {path}")
        material = element.find("material")
        color = None
        if material is not None:
            inline = material.find("color")
            if inline is not None:
                color = _rgba(inline.get("rgba"))
            else:
                color = materials.get(material.get("name") or "")
        visuals.append(_Visual(_origin(element.find("origin")), path, color))
    return visuals


def _parse_joint(element) -> UrdfJoint:
    name = element.get("name") or ""
    if not name:
        raise ValueError("URDF joint is missing a name")
    kind = element.get("type") or ""
    if kind not in {"revolute", "fixed"}:
        raise ValueError(f"Joint {name} has unsupported type {kind!r}")
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None or not parent.get("link") or not child.get("link"):
        raise ValueError(f"Joint {name} is missing a link")
    lower, upper = 0.0, 0.0
    axis = np.zeros(3)
    if kind == "revolute":
        limit = element.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Joint {name} is missing a limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        axis_element = element.find("axis")
        # URDF defaults a revolute joint with no axis element to +X.
        axis = _vec3("1 0 0" if axis_element is None else axis_element.get("xyz"))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            raise ValueError(f"Joint {name} has no axis")
        axis = axis / norm
    return UrdfJoint(
        name=name, joint_type=kind, parent=parent.get("link"), child=child.get("link"),
        origin=_origin(element.find("origin")), axis=axis,
        limit_lower=lower, limit_upper=upper,
    )


def _link_paths(robot: str, visuals, joints: list[UrdfJoint]) -> dict[str, str]:
    children = {joint.child for joint in joints}
    if len(children) != len(joints):
        raise ValueError("URDF links have more than one parent joint")
    known = set(visuals) | {joint.parent for joint in joints} | children
    roots = {link for link in known if link not in children}
    if len(roots) != 1:
        raise ValueError(f"URDF must have one base link, found {sorted(roots)}")
    base = roots.pop()
    paths = {base: f"/{robot}/{base}"}
    pending = list(joints)
    while pending:
        ready = [joint for joint in pending if joint.parent in paths]
        if not ready:
            raise ValueError("URDF joints do not form a tree")
        for joint in ready:
            paths[joint.child] = f"{paths[joint.parent]}/{joint.child}"
        ready_names = {joint.name for joint in ready}
        pending = [joint for joint in pending if joint.name not in ready_names]
    missing = [link for link in visuals if link not in paths]
    if missing:
        raise ValueError(f"URDF links are not connected to the base: {missing}")
    return paths
