"""One settled hand-eye sample. The CLI and the workshop page both call this."""

import time

import numpy as np

from so101_utils.transforms import Sample

# Joints must hold this close across a short wait before a sample counts.
SETTLE_DEG = 0.3
SETTLE_WAIT_S = 0.25


def joints_settled(before, after, limit=SETTLE_DEG) -> bool:
    return max(abs(before[name] - after[name]) for name in before) <= limit


def record_sample(arm, camera_T_tag, read_tag, *, settle_s: float = SETTLE_WAIT_S):
    """Return a Sample, or raise ValueError when the arm or the tag moved.

    `read_tag` is called once more after the settle wait and returns a 4x4
    camera_T_tag, or None when the tag left the frame.
    """
    if not arm.torque_on:
        raise ValueError("Turn torque on, let the arm settle, then record.")
    if camera_T_tag is None:
        raise ValueError("No tag in view.")
    before = arm.get_joints()
    time.sleep(settle_s)
    after = arm.get_joints()
    if not joints_settled(before, after):
        raise ValueError("Arm moved. Wait and try again.")
    again = read_tag()
    if again is None or not joints_settled(after, arm.get_joints()):
        raise ValueError("Tag or arm moved during capture.")
    return Sample(arm.gripper_link_pose(), np.asarray(again, dtype=float))
