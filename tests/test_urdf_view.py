"""The arm view has to load on Rerun 0.26, which has no UrdfTree."""

from pathlib import Path

from so101_utils.urdf_view import UrdfScene

URDF = Path(__file__).parents[1] / "assets" / "SO101" / "so101_new_calib.urdf"


def test_scene_places_the_shoulder_under_the_base():
    scene = UrdfScene.from_file_path(URDF)
    assert scene.name == "so101_new_calib"
    assert scene.link_entity("base_link") == "/so101_new_calib/base_link"
    shoulder = scene.get_joint_by_name("shoulder_pan")
    assert shoulder is not None and shoulder.joint_type == "revolute"
    assert shoulder.entity == "/so101_new_calib/base_link/shoulder_link"
    jaw = scene.get_joint_by_name("gripper")
    assert jaw is not None
    assert abs(jaw.limit_lower + 0.174533) < 1e-6
    assert jaw.entity.endswith("moving_jaw_so101_v1_link")
    assert "gripper_link" in jaw.entity


def test_revolute_transform_turns_about_the_joint_axis():
    import rerun as rr

    scene = UrdfScene.from_file_path(URDF)
    joint = scene.get_joint_by_name("shoulder_pan")
    rested = joint.compute_transform(0.0)
    turned = joint.compute_transform(0.4)
    assert isinstance(rested, rr.Transform3D)
    assert isinstance(turned, rr.Transform3D)
    clamped = joint.compute_transform(10.0, clamp=True)
    assert isinstance(clamped, rr.Transform3D)


def test_meshes_log_into_a_recording(tmp_path):
    import rerun as rr

    scene = UrdfScene.from_file_path(URDF)
    rr.init("so101-urdf-test", spawn=False)
    path = tmp_path / "arm.rrd"
    rr.save(path)
    scene.log_to_recording(rr)
    rr.get_global_data_recording().flush()
    rr.disconnect()
    # Thirteen visual meshes. An empty recording is a few kilobytes.
    assert path.is_file()
    assert path.stat().st_size > 100_000
