"""URDF variants and workspace-edge solves for the handwritten IK.

The real URDF is the reference. Each variant is the same geometry written the
way another export might write it, so the base-frame pose must not change.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.kinematics import (
    DOWNWARD_Z, TURN_TOLERANCE_M, Kinematics, _load_chain,
)
from so101_utils.config import load_robot, validated_limits
from so101_utils.model import JOINTS, spin_gripper, tool_rpy

URDF = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"

_FK_STATES = (
    {name: 0.0 for name in JOINTS},
    {"shoulder_pan": 90.0, "shoulder_lift": -40.0, "elbow_flex": 50.0,
     "wrist_flex": -30.0, "wrist_roll": 20.0},
    {"shoulder_pan": -60.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
     "wrist_flex": 80.0, "wrist_roll": 15.0},
    {"shoulder_pan": 35.0, "shoulder_lift": 25.0, "elbow_flex": -70.0,
     "wrist_flex": 55.0, "wrist_roll": -120.0},
)


@pytest.fixture(scope="module")
def urdf_text():
    if not URDF.is_file():
        pytest.skip("URDF not downloaded")
    return URDF.read_text()


@pytest.fixture(scope="module")
def reference(urdf_text):
    return Kinematics(str(URDF))


@pytest.fixture(scope="module")
def kin(urdf_text):
    return Kinematics(str(URDF), ee_tool_offset=0.02)


@pytest.fixture(scope="module")
def limits():
    return validated_limits(load_robot())


def _joint(root, name):
    for element in root.findall("joint"):
        if element.get("name") == name:
            return element
    raise AssertionError(f"No joint {name}")


def _write(tmp_path, urdf_text, change, name="variant.urdf"):
    root = ET.fromstring(urdf_text)
    change(root)
    path = tmp_path / name
    path.write_text(ET.tostring(root, encoding="unicode"))
    return str(path)


def _assert_same_fk(reference, variant):
    for state in _FK_STATES:
        assert variant.fk(state) == pytest.approx(reference.fk(state), abs=1e-9)


def test_an_extra_fixed_joint_does_not_change_the_tool_frame(reference, urdf_text, tmp_path):
    """A spacer between the wrist and the frame must compose out of the pose."""
    spacer_xyz = np.array([0.010, -0.020, 0.005])

    def change(root):
        frame = _joint(root, "gripper_frame_joint")
        origin = frame.find("origin")
        xyz = np.array([float(value) for value in origin.get("xyz").split()])
        origin.set("xyz", " ".join(str(value) for value in xyz - spacer_xyz))
        spacer = ET.SubElement(root, "joint", {"name": "wrist_spacer", "type": "fixed"})
        ET.SubElement(spacer, "origin", {"xyz": " ".join(str(v) for v in spacer_xyz),
                                         "rpy": "0 0 0"})
        ET.SubElement(spacer, "parent", {"link": "gripper_link"})
        ET.SubElement(spacer, "child", {"link": "spacer_link"})
        frame.find("parent").set("link", "spacer_link")

    variant = Kinematics(_write(tmp_path, urdf_text, change))
    _assert_same_fk(reference, variant)


def test_a_prismatic_jaw_does_not_change_the_tool_frame(reference, urdf_text, tmp_path):
    """Some exports model the jaw as a slider. At zero travel the pose is the same."""
    def change(root):
        _joint(root, "gripper").set("type", "prismatic")

    variant = Kinematics(_write(tmp_path, urdf_text, change))
    _assert_same_fk(reference, variant)
    assert variant._path_length == pytest.approx(reference._path_length)


def test_a_continuous_joint_is_movable(reference, urdf_text, tmp_path):
    """A continuous wrist has no stops in the URDF. The caller supplies the limits."""
    def change(root):
        _joint(root, "wrist_roll").set("type", "continuous")

    variant = Kinematics(_write(tmp_path, urdf_text, change))
    _assert_same_fk(reference, variant)


def test_a_renamed_frame_joint_does_not_change_the_tool_frame(reference, urdf_text, tmp_path):
    def change(root):
        _joint(root, "gripper_frame_joint").set("name", "tool_mount")

    variant = Kinematics(_write(tmp_path, urdf_text, change))
    _assert_same_fk(reference, variant)


def test_a_world_link_is_rebased_on_the_robot_base(reference, urdf_text, tmp_path):
    """The arm API is in base_link, so a world above it must not move the answer."""
    def change(root):
        ET.SubElement(root, "link", {"name": "world"})
        joint = ET.SubElement(root, "joint", {"name": "world_to_base", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": "1.0 -2.0 0.5", "rpy": "0.3 -0.1 0.4"})
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": "base_link"})

    variant = Kinematics(_write(tmp_path, urdf_text, change))
    _assert_same_fk(reference, variant)
    assert variant._path_length == pytest.approx(reference._path_length)
    # A goal the arm cannot reach must not become reachable through the world offset.
    limits = validated_limits(load_robot())
    with pytest.raises(ValueError, match="outside the arm's reach"):
        variant.position_ik({name: 0.0 for name in JOINTS}, (1.5, -2.0, 0.5), limits)


def test_a_movable_joint_without_an_axis_defaults_to_x(urdf_text, tmp_path):
    def change(root):
        element = _joint(root, "wrist_roll")
        element.remove(element.find("axis"))

    _base, chain = _load_chain(_write(tmp_path, urdf_text, change))
    joint = next(element for element in chain if element.name == "wrist_roll")
    assert joint.axis == pytest.approx((1.0, 0.0, 0.0))


def test_a_movable_joint_with_a_zero_axis_fails(urdf_text, tmp_path):
    def change(root):
        _joint(root, "wrist_roll").find("axis").set("xyz", "0 0 0")

    with pytest.raises(ValueError, match="has no axis"):
        _load_chain(_write(tmp_path, urdf_text, change))


def test_an_unsupported_joint_type_fails(urdf_text, tmp_path):
    def change(root):
        _joint(root, "gripper").set("type", "floating")

    with pytest.raises(ValueError, match="unsupported type"):
        _load_chain(_write(tmp_path, urdf_text, change))


def test_a_urdf_without_the_tool_frame_fails(urdf_text, tmp_path):
    def change(root):
        root.remove(_joint(root, "gripper_frame_joint"))

    with pytest.raises(ValueError, match="gripper_frame_link"):
        Kinematics(_write(tmp_path, urdf_text, change))


def test_downward_reaches_edge_points_inside_the_cone(kin, limits):
    """Edge points need a lean inside the cone, which the twist-free solve finds."""
    start = {name: 0.0 for name in JOINTS}
    for target in ((0.34, -0.12, 0.06), (0.36, 0.0, 0.02), (0.34, 0.15, 0.06)):
        solved = kin.position_ik(start, target, limits, downward=True)
        pose = kin.fk(solved)
        assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015
        assert pose[2, 0] < DOWNWARD_Z


def test_a_fixed_downward_pitch_does_not_replace_the_cone(kin, limits):
    """Pitch 90 pins the twist too; the edge only fits a lean with the twist free."""
    start = {name: 0.0 for name in JOINTS}
    target = (0.34, -0.12, 0.06)
    solved = kin.position_ik(start, target, limits, downward=True)
    assert np.linalg.norm(kin.tool_point(kin.fk(solved)) - target) < 0.015
    with pytest.raises(ValueError):
        kin.position_ik(start, target, limits, rpy=(0.0, 90.0, 0.0))


def test_downward_rejects_a_point_only_reachable_from_the_side(kin, limits):
    """No pose inside the cone gets within tolerance, so the move must not run."""
    start = {name: 0.0 for name in JOINTS}
    target = (0.34, -0.18, 0.06)
    reached = kin.fk(kin.position_ik(start, target, limits))
    assert np.linalg.norm(kin.tool_point(reached) - target) < 0.015
    with pytest.raises(ValueError, match="fingers pointing down"):
        kin.position_ik(start, target, limits, downward=True)


def test_ik_reaches_a_pose_below_and_behind_the_base(kin, limits):
    """A folded arm reaches this point; a uniform seed set rarely lands there."""
    truth = dict(zip(JOINTS, (-67.6, 71.5, 99.9, -63.2, -41.4)))
    pose = kin.fk(truth)
    target = kin.tool_point(pose)
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    solved = kin.position_ik({name: 0.0 for name in JOINTS}, target, limits, rpy=rpy)
    reached = kin.fk(solved)
    assert np.linalg.norm(kin.tool_point(reached) - target) < 0.015
    error = Rotation.from_matrix(pose[:3, :3].T @ reached[:3, :3]).magnitude()
    assert np.rad2deg(error) < 8.0


def test_a_small_roll_turn_near_the_base_axis_holds_the_point(kin, limits):
    """Inside the pan radius the yaw is undefined, so the wrist has to hold it."""
    state = dict(zip(JOINTS, (-67.6, 71.5, 99.9, -63.2, -41.4)))
    anchor = kin.tool_point(kin.fk(state))
    assert np.hypot(anchor[0], anchor[1]) < 0.05
    turned = spin_gripper(kin.fk(state)[:3, :3], 10.0)
    solved = kin.position_ik(state, anchor, limits, rotation=turned, follow_yaw=True,
                             local_only=True, tolerance_m=TURN_TOLERANCE_M)
    assert np.linalg.norm(kin.tool_point(kin.fk(solved)) - anchor) <= TURN_TOLERANCE_M


def test_invalid_goals_and_limits_are_rejected(kin, limits):
    start = {name: 0.0 for name in JOINTS}
    with pytest.raises(ValueError, match="Invalid TCP target"):
        kin.position_ik(start, (0.2, 0.0), limits)
    with pytest.raises(ValueError, match="Invalid TCP target"):
        kin.position_ik(start, (0.2, np.nan, 0.0), limits)
    with pytest.raises(ValueError, match="Invalid RPY target"):
        kin.position_ik(start, (0.28, -0.05, 0.02), limits, rpy=(0.0, 90.0))
    with pytest.raises(ValueError, match="replaces an orientation target"):
        kin.position_ik(start, (0.28, -0.05, 0.02), limits, downward=True,
                        rpy=(0.0, 90.0, 0.0))
    with pytest.raises(ValueError, match="Invalid rotation target"):
        kin.position_ik(start, (0.28, -0.05, 0.02), limits, rotation=np.eye(2))
    with pytest.raises(ValueError, match="not both"):
        kin.position_ik(start, (0.28, -0.05, 0.02), limits, rpy=(0.0, 90.0, 0.0),
                        rotation=np.eye(3))
    incomplete = dict(limits)
    del incomplete["wrist_roll"]
    with pytest.raises(ValueError, match="Limits required"):
        kin.position_ik(start, (0.28, -0.05, 0.02), incomplete)
    with pytest.raises(ValueError, match="ee_tool_offset"):
        Kinematics(str(URDF), ee_tool_offset=float("nan"))
