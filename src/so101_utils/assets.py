"""Pinned SO-101 URDF and meshes. The setup script is a thin wrapper around this."""

import io
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import sysconfig
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
URL = f"https://github.com/TheRobotStudio/SO-ARM100/archive/{COMMIT}.zip"
DEST = Path(__file__).resolve().parents[2] / "assets" / "SO101"
URDF = "so101_new_calib.urdf"
PREFIX = f"SO-ARM100-{COMMIT}/"
SOURCE = "Simulation/SO101/"

# Only files known to have been installed by the old, whole-directory downloader.
# In particular, local leader files and any other unexpected files are retained.
OLD_TOP_LEVEL = {
    "README.md", "joints_properties.xml", "scene.xml",
    "so101_leader_new_calib.urdf",
    "so101_new_calib_camera.urdf", "so101_new_calib_camera.xml",
    "so101_new_calib.xml", "so101_old_calib.urdf", "so101_old_calib.xml",
}
OLD_MESHES = {
    "Handle_SO101.stl", "Trigger_SO101.stl", "Wrist_Roll_SO101.stl",
    "wrist_camera_mount_so101_v1.stl", "wrist_camera_so101_v1.stl",
}
OLD_PARTS = {
    "base_motor_holder_so101_v1", "base_so101_v2",
    "motor_holder_so101_base_v1", "motor_holder_so101_wrist_v1",
    "moving_jaw_so101_v1", "rotation_pitch_so101_v1",
    "sts3215_03a_no_horn_v1", "sts3215_03a_v1",
    "under_arm_so101_v1", "upper_arm_so101_v1",
    "waveshare_mounting_plate_so101_v2", "wrist_roll_follower_so101_v1",
    "wrist_roll_pitch_so101_v2",
}
OLD_DOWNLOADS = OLD_TOP_LEVEL | {
    f"assets/{name}" for name in OLD_MESHES
} | {f"assets/{name}.part" for name in OLD_PARTS}

# Wheels install the pinned files here so an installed run never needs GitHub.
BUNDLED_SHARE = Path("share") / "so101-utils" / "SO101"


def _parse_urdf(urdf: bytes) -> ET.Element:
    try:
        root = ET.fromstring(urdf)
    except ET.ParseError as exc:
        raise ValueError("Invalid pinned URDF XML") from exc
    if root.tag != "robot":
        raise ValueError("Pinned URDF has no robot root")
    return root


def _mesh_paths(urdf: bytes) -> set[str]:
    root = _parse_urdf(urdf)
    meshes = root.findall(".//mesh")
    paths = set()
    for mesh in meshes:
        filename = mesh.get("filename", "")
        path = PurePosixPath(filename)
        if (len(path.parts) != 2 or path.parts[0] != "assets"
                or path.suffix.lower() != ".stl" or "\\" in filename
                or any(part in (".", "..") for part in filename.split("/"))):
            raise ValueError(f"Unsafe or unsupported URDF mesh reference: {filename!r}")
        paths.add(filename)
    if len(paths) != 13 or not meshes:
        raise ValueError(f"Expected 13 distinct STL meshes, found {len(paths)}")
    return paths


def _tool_frame_x_out_fingers(urdf: bytes) -> bytes:
    """Upstream gripper_frame has Z out the fingers. Turn it -90 deg about Y.

    The installed frame has X out the fingers so base XYZ Euler pitch +90
    points them down, matching the workshop tool convention.
    """
    root = _parse_urdf(urdf)
    joint = root.find("./joint[@name='gripper_frame_joint']")
    if joint is None:
        raise ValueError("Pinned URDF is missing gripper_frame_joint")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError("gripper_frame_joint has no origin")
    if origin.get("rpy") != "0 3.14159 0":
        raise ValueError(
            f"Unexpected gripper_frame_joint rpy {origin.get('rpy')!r}; "
            "expected upstream Z-out-fingers '0 3.14159 0'")
    origin.set("rpy", "0 1.5708 0")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _validate_stl(data: bytes, name: str) -> None:
    # The pinned upstream STLs are binary. Check their declared triangle count
    # against the entire payload, not just whether a file exists.
    if len(data) < 84:
        raise ValueError(f"Incomplete STL: {name}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if triangles == 0 or len(data) != 84 + 50 * triangles:
        raise ValueError(f"Invalid or incomplete STL: {name}")


def _archive_files(archive: zipfile.ZipFile) -> dict[str, bytes]:
    urdf_name = PREFIX + SOURCE + URDF
    license_name = PREFIX + "LICENSE"
    urdf = _read_unique(archive, urdf_name)
    license_bytes = _read_unique(archive, license_name)
    if not urdf or not license_bytes:
        raise ValueError("Pinned archive contains an empty URDF or license")
    _mesh_paths(urdf)  # validate before rewriting the tool frame
    files = {URDF: _tool_frame_x_out_fingers(urdf), "LICENSE.upstream": license_bytes}
    for path in _mesh_paths(files[URDF]):
        data = _read_unique(archive, PREFIX + SOURCE + path)
        _validate_stl(data, path)
        files[path] = data
    return files


def _read_unique(archive: zipfile.ZipFile, name: str) -> bytes:
    matches = [entry for entry in archive.infolist() if entry.filename == name and not entry.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Pinned archive needs exactly one {name!r}")
    return archive.read(matches[0])


def _install(files: dict[str, bytes], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
        raise ValueError(f"Asset destination is not a directory: {dest}")
    # The workspace is staged on the same filesystem, so renames can commit
    # the entire directory and restore the previous version on failure.
    with tempfile.TemporaryDirectory(prefix=".so101-assets-", dir=dest.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace / "staged"
        backup = workspace / "backup"
        if dest.exists():
            shutil.copytree(dest, staged, symlinks=True)
        else:
            staged.mkdir()
        if (staged / "assets").is_symlink():
            raise ValueError("Cannot update assets through a symlink")
        for name in OLD_DOWNLOADS:
            stale = staged / name
            if stale.is_symlink() or stale.is_file():
                stale.unlink()
        for name, data in files.items():
            target = staged / name
            if target.is_symlink() or target.parent.is_symlink():
                raise ValueError(f"Cannot overwrite symlink: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if _mesh_paths((staged / URDF).read_bytes()) != set(files) - {URDF, "LICENSE.upstream"}:
            raise ValueError("Staged URDF references do not match the staged meshes")
        for name, data in files.items():
            if not data or (staged / name).read_bytes() != data:
                raise ValueError(f"Incomplete staged asset: {name}")
        had_old = dest.exists()
        if had_old:
            os.replace(dest, backup)
        try:
            os.replace(staged, dest)
        except BaseException:
            if had_old:
                os.replace(backup, dest)
            raise


def assets_ready(dest: Path) -> bool:
    """True when the URDF and every mesh it names are already on disk."""
    urdf = dest / URDF
    if not urdf.is_file():
        return False
    try:
        meshes = _mesh_paths(urdf.read_bytes())
    except (OSError, ValueError):
        return False
    return all((dest / name).is_file() for name in meshes)


def bundled_assets() -> Path | None:
    """The pinned copy installed by the wheel, if this environment has one."""
    cargo = Path(sysconfig.get_path("data")) / BUNDLED_SHARE
    return cargo if assets_ready(cargo) else None


def _source_files(source: Path) -> dict[str, bytes]:
    license_path = source / "LICENSE.upstream"
    if not license_path.is_file():
        raise ValueError(f"Asset source is missing LICENSE.upstream: {source}")
    urdf = (source / URDF).read_bytes()
    files = {URDF: urdf, "LICENSE.upstream": license_path.read_bytes()}
    for path in _mesh_paths(urdf):
        data = (source / path).read_bytes()
        _validate_stl(data, path)
        files[path] = data
    return files


def download_assets(dest: Path) -> Path:
    try:
        with urlopen(URL, timeout=90) as response:
            payload = response.read()
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the SO-101 URDF from {URL}: {exc}. "
            "Installed packages ship these files; otherwise copy a complete "
            f"assets/SO101 folder into {dest.parent}."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = _archive_files(archive)
    _install(files, dest)
    print(f"Downloaded {dest / URDF} from {COMMIT}")
    return dest / URDF


def ensure_assets(dest: Path | None = None) -> Path:
    """Install the pinned URDF and meshes, preferring copies that ship with us.

    Wheels carry the files under ``share/so101-utils``, and a checkout keeps
    them in ``assets/SO101``. Both avoid the network; the pinned GitHub archive
    is only fetched when neither copy is available.
    """
    target = Path(dest) if dest is not None else DEST
    if assets_ready(target):
        return target / URDF
    checkout = DEST if assets_ready(DEST) else None
    for source in (bundled_assets(), checkout):
        if source is not None and source.resolve() != target.resolve():
            _install(_source_files(source), target)
            return target / URDF
    return download_assets(target)


def main() -> None:
    download_assets(DEST)
