"""Position IK on the real URDF. No serial port."""

from pathlib import Path

import numpy as np
import pytest

from so101_control.kinematics import JOINTS, Kinematics
from so101_utils.config import load_robot, validated_limits

URDF = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"


@pytest.fixture(scope="module")
def kin():
    if not URDF.is_file():
        pytest.skip("URDF not downloaded")
    return Kinematics(str(URDF))


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


def test_ik_rejects_a_point_outside_reach(kin):
    limits = validated_limits(load_robot())
    start = {name: 0.0 for name in JOINTS}
    with pytest.raises(ValueError):
        kin.position_ik(start, (5.0, 5.0, 5.0), limits)
