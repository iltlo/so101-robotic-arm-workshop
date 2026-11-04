"""Position and pose IK on the real URDF. No serial port."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.arm import SO101
from so101_control.kinematics import NEAR_STEP_DEG, Kinematics, _fingers_down
from so101_utils.config import load_robot, validated_limits
from so101_utils.model import JOINTS, gripper_rotation_from_rpy, tool_rpy
from so101_utils.transforms import transform

# PyBullet and PlaCo read this URDF's link frames. The gap on the samples below
# is about 1e-8 m and 1e-5 deg; these bounds are that rounding, not a frame change.
FK_POSITION_TOL_M = 1e-5
FK_ROTATION_TOL_DEG = 1e-4
# Converged PlaCo IK on the same seed and 4x4. The widest gap is 15 deg at the
# elbow on the downward click target. An elbow flip sits well outside this.
IK_JOINT_TOL_DEG = 18.0

URDF = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"


@pytest.fixture(scope="module")
def kin():
    if not URDF.is_file():
        pytest.skip("URDF not downloaded")
    return Kinematics(str(URDF))


def test_ik_matches_a_known_base_pose(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    goal_joints = dict(start)
    goal_joints["shoulder_pan"] = 15.0
    goal_joints["wrist_flex"] = -20.0
    goal_joints["wrist_roll"] = 25.0
    matrix = kin.fk(goal_joints)
    rpy = tool_rpy(matrix[:3, :3])
    solved = kin.position_ik(start, kin.tool_point(matrix), limits, rpy=rpy)
    pose = kin.fk(solved)
    assert np.linalg.norm(kin.tool_point(pose) - kin.tool_point(matrix)) < 0.015
    error = Rotation.from_matrix(matrix[:3, :3].T @ pose[:3, :3]).magnitude()
    assert np.rad2deg(error) < 1.0


def test_facing_forward_is_level_in_the_base(kin):
    rpy = tool_rpy(kin.fk({name: 0.0 for name in JOINTS})[:3, :3])
    assert rpy[0] == pytest.approx(2.8, abs=0.2)
    assert rpy[1] == pytest.approx(0.0, abs=0.2)
    assert rpy[2] == pytest.approx(0.0, abs=0.2)


def test_pitch_90_points_the_fingers_down_and_the_wrist_is_yaw(kin):
    fingers = Rotation.from_euler("xyz", [0.0, 90.0, 40.0], degrees=True).apply([1.0, 0.0, 0.0])
    assert fingers == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert tool_rpy(gripper_rotation_from_rpy((0.0, 90.0, 40.0))) == pytest.approx(
        (0.0, 90.0, 40.0), abs=0.1)

    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["wrist_flex"] = 90.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(20.0)
    assert level[0] == pytest.approx(0.0, abs=0.2)
    assert level[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[0] == pytest.approx(0.0, abs=0.2)
    assert twisted[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[2] - level[2] == pytest.approx(20.0, abs=0.5)


def test_a_few_degrees_off_straight_down_still_puts_the_wrist_in_yaw(kin):
    """The live arm sits a few degrees off vertical. Wrist twist is still yaw."""
    def angles(wrist_roll):
        joints = {name: 0.0 for name in JOINTS}
        joints["shoulder_pan"] = -60.0
        joints["wrist_flex"] = 80.0
        joints["wrist_roll"] = wrist_roll
        return tool_rpy(kin.fk(joints)[:3, :3])

    level = angles(0.0)
    twisted = angles(15.0)
    assert level[0] == pytest.approx(0.0, abs=0.2)
    assert level[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[0] == pytest.approx(0.0, abs=0.2)
    assert twisted[1] == pytest.approx(90.0, abs=0.2)
    assert twisted[2] - level[2] == pytest.approx(15.0, abs=0.5)


def test_a_side_pose_keeps_roll_and_pitch_apart(kin):
    def angles(**overrides):
        joints = {name: 0.0 for name in JOINTS}
        joints["shoulder_pan"] = 90.0
        joints.update(overrides)
        return tool_rpy(kin.fk(joints)[:3, :3])

    base = angles()
    rolled = angles(wrist_roll=20.0)
    flexed = angles(wrist_flex=-30.0)
    assert rolled[0] - base[0] == pytest.approx(-20.0, abs=0.5)
    assert rolled[1:] == pytest.approx(base[1:], abs=0.5)
    assert flexed[1] - base[1] == pytest.approx(-30.0, abs=0.5)
    assert flexed[0] == pytest.approx(base[0], abs=0.5)
    assert flexed[2] == pytest.approx(base[2], abs=0.5)


def test_a_short_translation_stays_near_the_start(kin):
    limits = validated_limits(load_robot())
    start = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -90.0,
        "elbow_flex": 90.0,
        "wrist_flex": 45.0,
        "wrist_roll": 0.0,
    }
    target = kin.tool_point(kin.fk(start)) + np.array([0.005, 0.0, 0.0])
    solved = kin.position_ik(start, target, limits)
    assert max(abs(solved[name] - start[name]) for name in JOINTS) <= NEAR_STEP_DEG
    assert np.linalg.norm(kin.tool_point(kin.fk(solved)) - target) < 0.015


def test_repeated_x_steps_do_not_rotate_the_wrist(kin):
    """Keyboard nudges translate. Position alone leaves five joints a null space."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        start = tool_rpy(kin.fk(state)[:3, :3])
        for _ in range(5):
            target = kin.tool_point(kin.fk(state)) + np.array([0.005, 0.0, 0.0])
            state = kin.position_ik(state, target, limits)
            assert tool_rpy(kin.fk(state)[:3, :3]) == pytest.approx(start, abs=0.5)
            assert state["wrist_roll"] == pytest.approx(0.0, abs=0.5)
    finally:
        kin.ee_tool_offset = previous


def test_downward_steps_keep_the_twist(kin):
    """Fingers down still leaves the twist, so a sideway step must not spin the jaws."""
    limits = validated_limits(load_robot())
    state = kin.position_ik({name: 0.0 for name in JOINTS}, (0.28, -0.05, 0.04),
                            limits, downward=True)
    start = tool_rpy(kin.fk(state)[:3, :3])
    for _ in range(4):
        target = kin.tool_point(kin.fk(state)) + np.array([0.0, 0.005, 0.0])
        state = kin.position_ik(state, target, limits, downward=True)
        assert tool_rpy(kin.fk(state)[:3, :3]) == pytest.approx(start, abs=0.5)


def test_ik_returns_to_a_known_tcp(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    goal_joints = dict(start)
    goal_joints["shoulder_pan"] = 20.0
    goal_joints["elbow_flex"] = -25.0
    target = kin.fk(goal_joints)[:3, 3]
    solved = kin.position_ik(start, target, limits)
    reached = kin.fk(solved)[:3, 3]
    assert np.linalg.norm(reached - target) < 0.015


def test_downward_points_the_fingers_down(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    solved = kin.position_ik(start, target, limits, downward=True)
    pose = kin.fk(solved)
    assert pose[2, 0] < -0.95
    assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015


def test_ee_tool_offset_follows_the_finger_axis(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    target = (0.28, -0.05, 0.04)
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        solved = kin.position_ik(start, target, limits, downward=True)
        pose = kin.fk(solved)
        tool = kin.tool_point(pose)
        assert np.linalg.norm(tool - target) < 0.015
        assert pose[2, 0] < -0.95
        # The tool point sits on gripper +Z.
        offset = tool - pose[:3, 3]
        assert np.dot(offset, pose[:3, 2]) > 0
        assert np.linalg.norm(offset) == pytest.approx(0.020, abs=0.005)
    finally:
        kin.ee_tool_offset = previous


def test_ik_accepts_a_start_just_past_the_stop(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    start["shoulder_lift"] = limits["shoulder_lift"][0] - 0.01
    target = kin.fk({name: 0.0 for name in JOINTS})[:3, 3]
    solved = kin.position_ik(start, target, limits)
    assert np.linalg.norm(kin.fk(solved)[:3, 3] - target) < 0.015
    start["shoulder_lift"] = limits["shoulder_lift"][0] - 5.0
    with pytest.raises(ValueError):
        kin.position_ik(start, target, limits)


def test_ik_recovers_when_the_arm_is_folded(kin):
    """A folded seed can miss a point that the home pose reaches."""
    limits = validated_limits(load_robot())
    folded = {
        "shoulder_pan": 0.0,
        "shoulder_lift": limits["shoulder_lift"][0],
        "elbow_flex": limits["elbow_flex"][0],
        "wrist_flex": limits["wrist_flex"][0],
        "wrist_roll": 0.0,
    }
    target = (0.23428342, 0.0300049, 0.10)
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        solved = kin.position_ik(folded, target, limits)
        pose = kin.fk(solved)
        assert np.linalg.norm(kin.tool_point(pose) - target) < 0.015
    finally:
        kin.ee_tool_offset = previous


def test_ik_rejects_a_point_outside_reach(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    with pytest.raises(ValueError):
        kin.position_ik(start, (5.0, 5.0, 5.0), limits)


def _rotation_error_deg(left, right) -> float:
    delta = Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude()
    return float(np.rad2deg(delta))


def _placo():
    from lerobot.model import RobotKinematics

    return RobotKinematics(
        urdf_path=str(URDF), target_frame_name="gripper_frame_link", joint_names=list(JOINTS))


def _converged_placo(solver, seed, pose) -> np.ndarray:
    """One PlaCo solve step does not reach. Repeat it from the same seed."""
    joints = np.array([seed[name] for name in JOINTS], dtype=float)
    for _ in range(30):
        joints = np.asarray(solver.inverse_kinematics(
            joints, pose, position_weight=1.0, orientation_weight=1.0), dtype=float)
    return joints


def test_fk_matches_placo(kin):
    placo = _placo()
    configs = (
        {name: 0.0 for name in JOINTS},
        {"shoulder_pan": 90.0, "shoulder_lift": -40.0, "elbow_flex": 50.0,
         "wrist_flex": -30.0, "wrist_roll": 20.0},
        {"shoulder_pan": -60.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
         "wrist_flex": 80.0, "wrist_roll": 15.0},
    )
    for joints in configs:
        angles = np.array([joints[name] for name in JOINTS], dtype=float)
        theirs = transform(placo.forward_kinematics(angles))
        ours = kin.fk(joints)
        assert np.linalg.norm(ours[:3, 3] - theirs[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(ours, theirs) < FK_ROTATION_TOL_DEG
        link = kin.fk_gripper_link(joints)
        placo_link = transform(placo.robot.get_T_world_frame("gripper_link"))
        assert np.linalg.norm(link[:3, 3] - placo_link[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(link, placo_link) < FK_ROTATION_TOL_DEG


def test_ik_stays_in_the_placo_configuration(kin):
    """Same seeds PlaCo gets: a reachable pose, a 5 mm step, and a downward click."""
    placo = _placo()
    limits = validated_limits(load_robot())
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.0
    try:
        seed = {name: 0.0 for name in JOINTS}
        goal = dict(seed)
        goal["shoulder_pan"] = 15.0
        goal["wrist_flex"] = -20.0
        goal["wrist_roll"] = 25.0
        matrix = kin.fk(goal)
        rpy = tool_rpy(matrix[:3, :3])
        cases = [(
            seed, kin.tool_point(matrix), gripper_rotation_from_rpy(rpy),
            {"rpy": tuple(float(angle) for angle in rpy)},
        )]
        start = {
            "shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
            "wrist_flex": 45.0, "wrist_roll": 0.0,
        }
        pose = kin.fk(start)
        cases.append((start, kin.tool_point(pose) + np.array([0.005, 0.0, 0.0]), pose[:3, :3], {}))
        home = kin.fk(seed)
        down = np.array([0.28, -0.05, 0.04])
        cases.append((seed, down, _fingers_down(home[:3, :3]), {"downward": True}))

        for joints, target, rotation, options in cases:
            solved = kin.position_ik(joints, target, limits, **options)
            reached = kin.fk(solved)
            assert np.linalg.norm(kin.tool_point(reached) - target) < 0.015
            if options.get("downward"):
                assert reached[2, 0] < -0.95
            goal_pose = np.eye(4)
            goal_pose[:3, :3] = rotation
            goal_pose[:3, 3] = target
            reference = _converged_placo(placo, joints, goal_pose)
            reference_pose = transform(placo.forward_kinematics(reference))
            assert np.linalg.norm(reference_pose[:3, 3] - target) < 0.015
            delta = [abs(solved[name] - reference[index]) for index, name in enumerate(JOINTS)]
            assert max(delta) <= IK_JOINT_TOL_DEG
    finally:
        kin.ee_tool_offset = previous


def test_workshop_entrypoints_keep_the_arm_api():
    source = inspect.getsource(SO101.move_to)
    assert "position_ik" in source
    assert "self._kin.fk" in source
    root = Path(__file__).parents[1]
    for path in ("scripts/keyboard_control.py", "application/click_and_go.py",
                 "application/pick_and_place.py"):
        text = (root / path).read_text()
        assert "move_to(" in text
        spec = importlib.util.spec_from_file_location(path.replace("/", "_"), root / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert "downward=True" in (root / "application/click_and_go.py").read_text()
