"""Position and pose IK on the real URDF. No serial port."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from so101_control.arm import SO101
from so101_control.kinematics import (
    NEAR_STEP_DEG, TURN_TOLERANCE_M, IKError, Kinematics,
)
from so101_utils.config import load_robot, validated_limits
from so101_utils.model import (
    JOINTS, gripper_rotation_from_rpy, rotate_gripper, tool_rpy,
)

# Recorded from PlaCo before that extra was removed. The live gap was about
# 1e-8 m and 1e-5 deg; these bounds are that rounding, not a frame change.
FK_POSITION_TOL_M = 1e-5
FK_ROTATION_TOL_DEG = 1e-4
# Joint answers of this solver, and the PyBullet answers they replace.
# The known pose is checked against the joints that generated it. PyBullet
# stopped at [14.9980, 0.5335, -1.2774, -19.2381, 24.9915], 1.3 deg short
# on the elbow. The 5 mm step and the downward click have no single answer on
# a five-joint arm; PyBullet's are the second array. This solver stays in that
# family: within 7 deg on the step and 11 deg on the click. PlaCo's soft solve
# of the known pose left the family (about 42 deg at the lift) and still fails.
IK_JOINT_TOL_DEG = 0.2

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


def test_repeated_y_steps_move_sideways_and_keep_the_tilt(kin):
    """A five-joint arm cannot hold yaw while the pan carries the tool sideways."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    previous = kin.ee_tool_offset
    kin.ee_tool_offset = 0.020
    try:
        start = tool_rpy(kin.fk(state)[:3, :3])
        for _ in range(5):
            before = kin.tool_point(kin.fk(state))
            target = before + np.array([0.0, 0.005, 0.0])
            state = kin.position_ik(state, target, limits)
            after = kin.tool_point(kin.fk(state))
            assert after[1] - before[1] > 0.0035
            roll, pitch, _ = tool_rpy(kin.fk(state)[:3, :3])
            assert (roll, pitch) == pytest.approx((start[0], start[1]), abs=0.5)
    finally:
        kin.ee_tool_offset = previous


def test_a_turn_holds_the_tool_point(kin):
    """F and V turn the gripper while the reported xyz stays put."""
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    anchor = kin.tool_point(kin.fk(state))
    for _ in range(4):
        rpy = rotate_gripper(tool_rpy(kin.fk(state)[:3, :3]), pitch=5.0)
        goal = gripper_rotation_from_rpy(rpy)
        state = kin.position_ik(state, anchor, limits, rpy=rpy,
                                follow_yaw=True, local_only=True)
        pose = kin.fk(state)
        assert np.linalg.norm(kin.tool_point(pose) - anchor) <= TURN_TOLERANCE_M
        # World-Z rotation is the direction this arm cannot hold; roll and pitch
        # are the ones a pitch key asks for.
        error = Rotation.from_matrix(goal @ pose[:3, :3].T).as_rotvec()
        assert np.rad2deg(np.linalg.norm(error[:2])) < 0.5
    assert tool_rpy(kin.fk(state)[:3, :3])[1] == pytest.approx(20.0, abs=1.0)


def test_a_turn_raises_instead_of_jumping_when_it_cannot_hold(kin):
    limits = validated_limits(load_robot())
    state = {name: 0.0 for name in JOINTS}
    anchor = kin.tool_point(kin.fk(state))
    rpy = rotate_gripper(tool_rpy(kin.fk(state)[:3, :3]), pitch=60.0)
    with pytest.raises(IKError):
        kin.position_ik(state, anchor, limits, rpy=rpy,
                        follow_yaw=True, local_only=True)


def test_a_turn_survives_a_mid_workspace_pose(kin):
    """Away from full extension the arm can pitch further while holding the point."""
    limits = validated_limits(load_robot())
    state = kin.position_ik({name: 0.0 for name in JOINTS}, (0.25, 0.0, 0.20),
                            limits)
    anchor = kin.tool_point(kin.fk(state))
    for _ in range(8):
        rpy = rotate_gripper(tool_rpy(kin.fk(state)[:3, :3]), pitch=5.0)
        state = kin.position_ik(state, anchor, limits, rpy=rpy,
                                follow_yaw=True, local_only=True)
        assert np.linalg.norm(kin.tool_point(kin.fk(state)) - anchor) <= TURN_TOLERANCE_M
    assert tool_rpy(kin.fk(state)[:3, :3])[1] == pytest.approx(40.0, abs=1.5)


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


# gripper_frame_link, then gripper_link, recorded from PlaCo on this URDF.
_RECORDED_POSES = (
    (
        {name: 0.0 for name in JOINTS},
        np.array([
            [1.0, -0.00001030, -0.00000234, 0.39136147],
            [0.00001017, 0.99881526, -0.04866293, -0.00000921],
            [0.00000284, 0.04866293, 0.99881526, 0.22646971],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [-0.00000601, -0.00001030, -1.0, 0.29323402],
            [-0.04866293, 0.99881526, -0.00001000, -0.00017677],
            [0.99881526, 0.04866293, -0.00000651, 0.23437033],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        {"shoulder_pan": 90.0, "shoulder_lift": -40.0, "elbow_flex": 50.0,
         "wrist_flex": -30.0, "wrist_roll": 20.0},
        np.array([
            [0.00000853, 0.95522288, 0.29588721, 0.03611355],
            [-0.93969250, -0.10119183, 0.32670815, -0.26542692],
            [0.34202047, -0.27804578, 0.89761492, 0.24995594],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [0.29588721, 0.95522288, -0.00000961, 0.03865847],
            [0.32671160, -0.10119183, 0.93969130, -0.17065851],
            [0.89761366, -0.27804578, -0.34202376, 0.22342454],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
    (
        {"shoulder_pan": -60.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
         "wrist_flex": 80.0, "wrist_roll": 15.0},
        np.array([
            [0.08682207, -0.95058286, 0.29808415, 0.14731489],
            [0.15039083, 0.30829123, 0.93932908, 0.18377148],
            [-0.98480685, -0.03672537, 0.16972543, 0.07622953],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [0.29808383, -0.95058286, -0.08682316, 0.14094268],
            [0.93932853, 0.30829123, -0.15039428, 0.17650162],
            [0.16972904, -0.03672537, 0.98480622, 0.17419886],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ),
)


def test_fk_matches_the_recorded_frames(kin):
    for joints, frame, link in _RECORDED_POSES:
        ours = kin.fk(joints)
        assert np.linalg.norm(ours[:3, 3] - frame[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(ours, frame) < FK_ROTATION_TOL_DEG
        got = kin.fk_gripper_link(joints)
        assert np.linalg.norm(got[:3, 3] - link[:3, 3]) < FK_POSITION_TOL_M
        assert _rotation_error_deg(got, link) < FK_ROTATION_TOL_DEG


def test_ik_stays_near_the_recorded_configuration(kin):
    """A reachable pose, a 5 mm step, and a downward click, against recorded joints."""
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
            seed, kin.tool_point(matrix),
            {"rpy": tuple(float(angle) for angle in rpy)},
            np.array([15.0, 0.0, 0.0, -20.0, 25.0]),
            np.array([14.9980, 0.5335, -1.2774, -19.2381, 24.9915]),
            2.0,
        )]
        start = {
            "shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 90.0,
            "wrist_flex": 45.0, "wrist_roll": 0.0,
        }
        pose = kin.fk(start)
        cases.append((
            start, kin.tool_point(pose) + np.array([0.005, 0.0, 0.0]), {},
            np.array([-0.0000, -83.5025, 88.8648, 39.6377, 0.0000]),
            np.array([0.0, -87.8258, 89.5194, 43.2864, 0.0]),
            7.0,
        ))
        cases.append((
            seed, np.array([0.28, -0.05, 0.04]), {"downward": True},
            np.array([11.2623, 35.9458, -37.0912, 91.1456, 14.0516]),
            np.array([11.2604, 30.3480, -26.8648, 86.4439, 14.0504]),
            11.0,
        ))
        for joints, target, options, reference, pybullet, family_deg in cases:
            solved = kin.position_ik(joints, target, limits, **options)
            reached = kin.fk(solved)
            assert np.linalg.norm(kin.tool_point(reached) - target) < 0.015
            if options.get("downward"):
                assert reached[2, 0] < -0.95
            got = np.array([solved[name] for name in JOINTS])
            assert max(abs(got - reference)) <= IK_JOINT_TOL_DEG
            assert max(abs(got - pybullet)) <= family_deg
    finally:
        kin.ee_tool_offset = previous


def test_workshop_entrypoints_keep_the_arm_api():
    source = inspect.getsource(SO101.move_to)
    assert "position_ik" in source
    assert "self._kin.fk" in source
    root = Path(__file__).parents[1]
    for path in ("scripts/keyboard_control.py", "application/click_and_go.py",
                 "application/pick_and_place.py"):
        spec = importlib.util.spec_from_file_location(path.replace("/", "_"), root / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert "move_to(" in (root / "src/so101_control/teleop.py").read_text()
    moves = (root / "src/so101_utils/moves.py").read_text()
    assert "move_to(" in moves
    assert "downward=True" in moves
    assert "move_to(" in (root / "application/pick_and_place.py").read_text()
