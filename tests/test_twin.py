"""Exercise the real Rerun URDF importer and one-model animated recording."""

from pathlib import Path
import sys

import numpy as np
import pytest

from so101_control.config import JOINTS
from so101_control.twin import Twin


def test_pinned_model_named_frames_and_three_pose_recording(tmp_path):
    rr = pytest.importorskip("rerun")
    urdf = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"
    if not urdf.exists():
        pytest.skip("Install pinned follower assets before running twin tests")
    recording = tmp_path / "so101.rrd"
    viewer = Twin(urdf, spawn=False, recording=str(recording))
    shoulder = viewer.joints["shoulder_pan"]
    assert shoulder.parent_link == "base_link"
    assert shoulder.child_link == "shoulder_link"
    assert np.linalg.norm(shoulder.origin_xyz) > 0.01
    assert viewer.jaw.parent_link == "gripper_link"
    assert viewer.jaw.child_link == "moving_jaw_so101_v1_link"
    assert len(viewer.tree.get_visual_geometry_paths(viewer.tree.root_link())) >= 1
    for angle, gripper in ((0.0, 0.0), (20.0, 50.0), (-10.0, 100.0)):
        state = {name: 0.0 for name in JOINTS}
        state["shoulder_pan"] = angle
        state["gripper"] = gripper
        viewer.update(state, ee_pose=np.eye(4))
    rr.get_global_data_recording().flush()
    assert viewer.tick == 3
    assert recording.stat().st_size > 100_000
    assert recording.with_suffix(".rbl").stat().st_size > 1000
    # A valid recording alone cannot establish that FK visuals share the
    # importer's named base frame. Inspect the actual recorded archetypes.
    import subprocess

    result = subprocess.run(
        [str(Path(sys.executable).parent / "rerun"), "rrd", "print", str(recording), "-v"],
        capture_output=True, text=True, check=True,
    )
    assert "/so101_new_calib/ee_pose" in result.stdout
    assert "/transforms/fk_ee_pose" in result.stdout
    pose_entity = subprocess.run(
        [str(Path(sys.executable).parent / "rerun"), "rrd", "print", str(recording),
         "--entity", "/so101_new_calib/ee_pose", "-vvv"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "CoordinateFrame:frame" in pose_entity
    assert "TransformAxes3D:axis_length" in pose_entity
    assert "fk_ee_pose" in pose_entity
    transform_entity = subprocess.run(
        [str(Path(sys.executable).parent / "rerun"), "rrd", "print", str(recording),
         "--entity", "/transforms/fk_ee_pose", "-vvv"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "Transform3D:parent_frame" in transform_entity
    assert "Transform3D:child_frame" in transform_entity
    assert "base_link" in transform_entity
    assert "fk_ee_pose" in transform_entity


def test_hover_goal_is_a_labeled_base_frame_point_in_recording(tmp_path):
    rr = pytest.importorskip("rerun")
    urdf = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"
    if not urdf.exists():
        pytest.skip("Install pinned follower assets before running twin tests")
    recording = tmp_path / "goal.rrd"
    viewer = Twin(urdf, spawn=False, recording=str(recording))
    viewer.show_hover_goal([0.1, -0.05, 0.04])
    rr.get_global_data_recording().flush()
    import subprocess

    command = [str(Path(sys.executable).parent / "rerun"), "rrd", "print", str(recording)]
    root = f"/{viewer.tree.name}/hover_goal"
    point = subprocess.run([*command, "--entity", root, "-vvv"],
                           capture_output=True, text=True, check=True).stdout
    assert "Points3D" in point
    assert "candidate TCP (preview only)" in point
    pose = subprocess.run([*command, "--entity", "/transforms/hover_goal", "-vvv"],
                          capture_output=True, text=True, check=True).stdout
    assert "base_link" in pose and "hover_goal" in pose
    assert "0.1" in pose and "0.04" in pose


def test_jaw_interpolation_and_fk_pose_transform():
    calls = []

    class Joint:
        limit_lower = -0.17
        limit_upper = 1.74

        def compute_transform(self, value, clamp):
            return value

    class Recording:
        def set_time(self, *args, **kwargs):
            pass

        def log(self, path, value):
            calls.append((path, value))

        class Scalars:
            def __init__(self, value):
                pass

        class Transform3D:
            def __init__(self, *, translation, mat3x3, parent_frame, child_frame):
                self.translation = np.asarray(translation)
                self.rotation = np.asarray(mat3x3)
                self.parent_frame = parent_frame
                self.child_frame = child_frame

    viewer = Twin.__new__(Twin)
    viewer.rr = Recording()
    viewer.tick = 0
    viewer.joints = {name: Joint() for name in JOINTS}
    viewer.jaw = Joint()
    viewer.tree = type("Tree", (), {"name": "so101_new_calib"})()
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    viewer.update({**dict.fromkeys(JOINTS, 0.0), "gripper": 0.0}, ee_pose=pose)
    viewer.update({**dict.fromkeys(JOINTS, 0.0), "gripper": 100.0}, ee_pose=pose)
    jaw_logs = [value for path, value in calls if path == "transforms/gripper"]
    assert jaw_logs == [viewer.jaw.limit_lower, viewer.jaw.limit_upper]
    frames = [value for path, value in calls if path == "transforms/fk_ee_pose"]
    assert len(frames) == 2
    assert np.allclose(frames[0].translation, pose[:3, 3])
    assert np.allclose(frames[0].rotation, pose[:3, :3])
    assert (frames[0].parent_frame, frames[0].child_frame) == ("base_link", "fk_ee_pose")


def test_every_arm_joint_has_its_own_dynamic_transform_path(monkeypatch):
    calls = []

    class Recording:
        def set_time(self, *args, **kwargs):
            pass

        def log(self, path, value):
            calls.append(path)

        class Scalars:
            def __init__(self, value):
                pass

    from so101_control.twin import Twin

    viewer = Twin.__new__(Twin)
    viewer.rr = Recording()
    viewer.tick = 0
    viewer.joints = {joint: type("Joint", (), {"compute_transform": lambda self, value, clamp: object()})()
                     for joint in JOINTS}
    viewer.jaw = type("Jaw", (), {"limit_lower": 0.0, "limit_upper": 1.0,
                                  "compute_transform": lambda self, value, clamp: object()})()
    viewer.update({**{joint: 0.0 for joint in JOINTS}, "gripper": 0.0})
    assert {path for path in calls if path.startswith("transforms/")} == {
        f"transforms/{joint}" for joint in (*JOINTS, "gripper")
    }


def test_torque_disabled_without_opt_in():
    from so101_control.torque import torque_toggle_listener

    with torque_toggle_listener(False) as keys:
        assert keys.consume() is False
        keys.request()
        keys.request()  # repeated press before next loop iteration is one toggle
        assert keys.consume() is True
        assert keys.consume() is False


def test_torque_toggle_writes_without_prompt(monkeypatch):
    from so101_control.torque import process_torque_toggle

    writes = []

    class Bus:
        def disable_torque(self):
            writes.append("off")

        def enable_torque(self):
            writes.append("on")

    class Robot:
        bus = Bus()

    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(AssertionError("prompt blocked viewer")))
    assert process_torque_toggle(Robot(), True) is False
    assert process_torque_toggle(Robot(), False) is True
    assert writes == ["off", "on"]


def test_live_loop_logs_changed_feedback_after_torque_off(monkeypatch):
    import importlib.util

    script = Path(__file__).parents[1] / "teleop" / "twin.py"
    spec = importlib.util.spec_from_file_location("workshop_live_twin", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feedback = iter([5.0, 25.0, 40.0])
    updates = []
    writes = []

    class Robot:
        class Bus:
            def disable_torque(self):
                writes.append("off")

            def enable_torque(self):
                writes.append("on")

        bus = Bus()

        def get_observation(self):
            value = next(feedback)
            return {f"{joint}.pos": value if joint == "shoulder_pan" else 0.0
                    for joint in (*JOINTS, "gripper")}

    class Keys:
        def __init__(self):
            self.first = True

        def consume(self):
            if self.first:
                self.first = False
                return True
            return False

    class Kinematics:
        def fk(self, state):
            pose = np.eye(4)
            pose[0, 3] = state["shoulder_pan"] / 100
            return pose

    class Twin:
        def update(self, state, *, ee_pose):
            updates.append((state["shoulder_pan"], ee_pose[0, 3]))

    def stop_after_three(_):
        if len(updates) == 3:
            raise KeyboardInterrupt()

    monkeypatch.setattr(module.time, "sleep", stop_after_three)
    with pytest.raises(KeyboardInterrupt):
        module._live_loop(Robot(), Kinematics(), Twin(), Keys(), toggle_torque=True)
    assert writes == ["off"]
    assert updates == [(5.0, 0.05), (25.0, 0.25), (40.0, 0.4)]