"""Pinned asset downloader: exact selection, validation, and safe replacement."""

import io
from pathlib import Path
import struct
import importlib.util
import zipfile

import pytest

import so101_utils.assets as packaged

_PATH = Path(__file__).parents[1] / "scripts" / "fetch_so101_assets.py"
_SPEC = importlib.util.spec_from_file_location("fetch_so101_assets", _PATH)
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)


MESHES = [f"assets/mesh_{index}.stl" for index in range(13)]
STL = bytes(80) + struct.pack("<I", 1) + bytes(50)


def make_archive(*, references=MESHES, omit=(), replacements=None):
    urdf = ("<robot name='so101'>" + "".join(
        f"<link name='{index}'><visual><geometry><mesh filename='{path}'/>"
        "</geometry></visual></link>" for index, path in enumerate(references)
    ) + "<joint name='gripper_frame_joint' type='fixed'>"
        "<origin xyz='0 0 0' rpy='0 3.14159 0'/>"
        "</joint></robot>").encode()
    members = {
        fetch.PREFIX + fetch.SOURCE + fetch.URDF: urdf,
        fetch.PREFIX + "LICENSE": b"Apache-2.0",
        **{fetch.PREFIX + fetch.SOURCE + mesh: STL for mesh in MESHES},
        fetch.PREFIX + fetch.SOURCE + "so101_old_calib.urdf": b"not selected",
        fetch.PREFIX + "Simulation/SO100/secret.txt": b"not selected",
    }
    members.update(replacements or {})
    for path in omit:
        members.pop(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, data in members.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def invoke(monkeypatch, dest, data):
    monkeypatch.setattr(fetch, "DEST", dest)
    monkeypatch.setattr(fetch, "urlopen", lambda url, timeout: io.BytesIO(data))
    fetch.main()


def snapshot(dest):
    return {p.relative_to(dest).as_posix(): p.read_bytes()
            for p in dest.rglob("*") if p.is_file()}


def test_installs_only_urdf_thirteen_meshes_and_license(monkeypatch, tmp_path):
    dest = tmp_path / "assets" / "SO101"
    invoke(monkeypatch, dest, make_archive())
    assert set(snapshot(dest)) == {fetch.URDF, "LICENSE.upstream", *MESHES}
    assert snapshot(dest)["LICENSE.upstream"] == b"Apache-2.0"
    assert fetch._mesh_paths((dest / fetch.URDF).read_bytes()) == set(MESHES)
    joint = fetch.ET.fromstring(snapshot(dest)[fetch.URDF]).find(
        "./joint[@name='gripper_frame_joint']/origin")
    assert joint.get("rpy") == "0 1.5708 0"


def test_tool_frame_patch_requires_upstream_z_out_fingers():
    good = make_archive()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        urdf = archive.read(fetch.PREFIX + fetch.SOURCE + fetch.URDF)
    patched = fetch._tool_frame_x_out_fingers(urdf)
    origin = fetch.ET.fromstring(patched).find("./joint[@name='gripper_frame_joint']/origin")
    assert origin.get("rpy") == "0 1.5708 0"
    with pytest.raises(ValueError, match="Unexpected gripper_frame_joint rpy"):
        fetch._tool_frame_x_out_fingers(patched)


def test_success_cleans_only_known_previous_downloads(monkeypatch, tmp_path):
    dest = tmp_path / "assets" / "SO101"
    (dest / "assets").mkdir(parents=True)
    for name in (fetch.URDF, "LICENSE.upstream", "scene.xml",
                 "assets/base_so101_v2.part", "assets/wrist_camera_so101_v1.stl",
                 "so101_leader_new_calib.urdf", "assets/Handle_SO101.stl",
                 "assets/custom.stl", "custom-notes.txt"):
        (dest / name).write_bytes(b"old")
    outside = tmp_path / "reference" / "so101_new_calib.urdf"
    outside.parent.mkdir()
    outside.write_bytes(b"reference unchanged")
    invoke(monkeypatch, dest, make_archive())
    assert set(snapshot(dest)) == {fetch.URDF, "LICENSE.upstream", *MESHES,
                                  "assets/custom.stl", "custom-notes.txt"}
    assert (dest / "custom-notes.txt").read_bytes() == b"old"
    assert outside.read_bytes() == b"reference unchanged"


@pytest.mark.parametrize("changes", [
    {"omit": {fetch.PREFIX + "LICENSE"}},
    {"omit": {fetch.PREFIX + fetch.SOURCE + fetch.URDF}},
    {"omit": {fetch.PREFIX + fetch.SOURCE + MESHES[0]}},
    {"replacements": {fetch.PREFIX + fetch.SOURCE + MESHES[0]: STL[:-1]}},
    {"replacements": {fetch.PREFIX + "LICENSE": b""}},
    {"replacements": {fetch.PREFIX + fetch.SOURCE + fetch.URDF: b"<invalid"}},
    {"references": [*MESHES[:-1], "../escape.stl"]},
    {"references": [*MESHES[:-1], "assets/../escape.stl"]},
    {"references": [*MESHES[:-1], "assets\\escape.stl"]},
    {"references": MESHES[:-1]},
])
def test_bad_archives_do_not_change_old_assets(monkeypatch, tmp_path, changes):
    dest = tmp_path / "assets" / "SO101"
    dest.mkdir(parents=True)
    (dest / fetch.URDF).write_bytes(b"original")
    (dest / "scene.xml").write_bytes(b"original extra")
    before = snapshot(dest)
    with pytest.raises((ValueError, KeyError)):
        invoke(monkeypatch, dest, make_archive(**changes))
    assert snapshot(dest) == before


def test_actual_managed_urdf_has_thirteen_complete_binary_meshes():
    dest = Path(__file__).parents[1] / "assets" / "SO101"
    urdf = (dest / fetch.URDF).read_bytes()
    paths = fetch._mesh_paths(urdf)
    assert len(paths) == 13
    for name in paths:
        fetch._validate_stl((dest / name).read_bytes(), name)
    origin = fetch.ET.fromstring(urdf).find("./joint[@name='gripper_frame_joint']/origin")
    assert origin.get("rpy") == "0 1.5708 0"


def test_commit_error_restores_previous_directory(monkeypatch, tmp_path):
    dest = tmp_path / "assets" / "SO101"
    dest.mkdir(parents=True)
    (dest / fetch.URDF).write_bytes(b"original")
    (dest / "scene.xml").write_bytes(b"original extra")
    before = snapshot(dest)
    replace = fetch.os.replace

    def fail_second_rename(src, dst):
        if Path(src).name == "staged":
            raise OSError("simulated commit error")
        return replace(src, dst)

    monkeypatch.setattr(fetch.os, "replace", fail_second_rename)
    with pytest.raises(OSError, match="simulated commit error"):
        invoke(monkeypatch, dest, make_archive())
    assert snapshot(dest) == before


def test_symlink_cannot_redirect_written_mesh(monkeypatch, tmp_path):
    dest = tmp_path / "assets" / "SO101"
    dest.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "wrist_camera_so101_v1.stl").write_bytes(b"outside unchanged")
    (dest / "assets").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        invoke(monkeypatch, dest, make_archive())
    assert (outside / "wrist_camera_so101_v1.stl").read_bytes() == b"outside unchanged"
    assert (dest / "assets").is_symlink()


def ready_directory(root):
    """A directory assets_ready accepts, without the 16 MB payload."""
    (root / "assets").mkdir(parents=True)
    urdf = ("<robot name='so101'>" + "".join(
        f"<link name='{index}'><visual><geometry><mesh filename='assets/mesh_{index}.stl'/>"
        "</geometry></visual></link>" for index in range(13)
    ) + "</robot>").encode()
    (root / packaged.URDF).write_bytes(urdf)
    for index in range(13):
        (root / f"assets/mesh_{index}.stl").write_bytes(b"mesh")
    return root


def test_bundled_assets_locate_the_installed_share_directory(monkeypatch, tmp_path):
    root = tmp_path / "environment"
    share = ready_directory(root / "share" / "so101-utils" / "SO101")
    monkeypatch.setattr(packaged.sysconfig, "get_path", lambda name: str(root))
    assert packaged.bundled_assets() == share
    (share / packaged.URDF).unlink()
    assert packaged.bundled_assets() is None


def test_ensure_assets_copies_bundled_files_without_network(monkeypatch, tmp_path):
    source = Path(__file__).parents[1] / "assets" / "SO101"
    target = tmp_path / "station" / "assets" / "SO101"
    monkeypatch.setattr(packaged, "bundled_assets", lambda: source)
    monkeypatch.setattr(packaged, "DEST", tmp_path / "missing-checkout")
    monkeypatch.setattr(
        packaged, "urlopen",
        lambda *args, **kwargs: pytest.fail("ensure_assets used the network"),
    )
    assert packaged.ensure_assets(target) == target / packaged.URDF
    assert snapshot(target) == snapshot(source)
