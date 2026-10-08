"""Install the pinned SO-101 follower URDF, its meshes, and upstream license."""

import io
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
URL = f"https://github.com/TheRobotStudio/SO-ARM100/archive/{COMMIT}.zip"
DEST = Path(__file__).resolve().parents[1] / "assets" / "SO101"
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


def _mesh_paths(urdf: bytes) -> set[str]:
    try:
        root = ET.fromstring(urdf)
    except ET.ParseError as exc:
        raise ValueError("Invalid pinned URDF XML") from exc
    if root.tag != "robot":
        raise ValueError("Pinned URDF has no robot root")
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
    files = {URDF: urdf, "LICENSE.upstream": license_bytes}
    for path in _mesh_paths(urdf):
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


def main() -> None:
    with urlopen(URL, timeout=90) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            files = _archive_files(archive)
    _install(files, DEST)
    print(f"Downloaded {DEST / URDF} from {COMMIT}")


if __name__ == "__main__":
    main()
