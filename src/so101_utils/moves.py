"""Click-and-go and click-and-pick. Both call the arm; neither opens a window."""

import time

from so101_control.arm import REST_JOINTS
from so101_utils.model import JOINTS

# Metres above the click, which lies on the base plane (z = 0).
HOVER_M = 0.02
# Metres along gripper_frame_link +Z. Zero puts the click goal on that link's origin.
CLICK_TOOL_OFFSET_M = 0.0
# Metres above the click for a pick. The tool point is PICK_TOOL_OFFSET_M out the fingers.
PICK_HEIGHT_M = 0.00
APPROACH_M = 0.080
DWELL_S = 0.3
PICK_TOOL_OFFSET_M = 0.025
# Metres above the place tool point before descending (after the pick).
PLACE_JUMP_M = 0.05
# The pick scripts pass their own approach, height, and delays; these values are
# what the web page uses.


def click_ready(arm, view=None) -> None:
    """Fold to the rest pose. A click waits until that move is sent."""
    if view is not None:
        view.report("Going to rest")
    arm.rest(wait=False)


def go_to(arm, view, target) -> bool:
    """Command one hover point. False when a newer click has taken over."""
    x, y, z = (float(value) for value in target)
    where = f"{x:.3f}, {y:.3f}, {z:.3f} m"
    view.report(f"Moving to {where}")
    arrived = arm.move_to(
        x, y, z, wait=False, cancel=view.has_pending, downward=True,
    )
    if arrived:
        view.report(f"Moved to {where}")
    return arrived


def pick_ready(arm, view=None) -> None:
    """Fold to the rest pose and open the gripper.

    `rest` closes the jaw, so the open has to come after the fold.
    """
    if view is not None:
        view.report("Going to rest")
    arm.rest(wait=False)
    arm.release()


def pick_and_return(arm, view, target, dwell_s: float = DWELL_S, *,
                    approach_m: float = APPROACH_M,
                    close_delay_s: float = DWELL_S,
                    open_delay_s: float = DWELL_S,
                    place_joints: dict[str, float] | None = None,
                    place_jump_m: float = PLACE_JUMP_M) -> bool:
    """Open the gripper, hover above target, move down, rest, and open.

    Both poses point the fingers straight down; the twist is re-solved on the way
    down. `approach_m` is the hover height above the target. `close_delay_s` and
    `open_delay_s` pause after the grip and after the release. A newer click
    skips the close.
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    hover_z = z + approach_m
    view.report("Opening gripper")
    arm.release()
    view.report(f"Moving above the pick: {x:.3f}, {y:.3f}, {hover_z:.3f} m")
    if not _down_to(arm, view, x, y, hover_z):
        return False
    view.report(f"Moving down to {x:.3f}, {y:.3f}, {z:.3f} m")
    if not _down_to(arm, view, x, y, z):
        return False
    view.report("Holding the pick")
    hold(arm, dwell_s)
    view.report("Closing gripper")
    arm.grip()
    view.report("Holding the grasp")
    hold(arm, close_delay_s)
    if not _place_at(arm, view, place_joints, place_jump_m):
        return False
    arm.release()
    view.report("Released at place")
    hold(arm, open_delay_s)
    return True


def hold(arm, seconds: float) -> None:
    """Stay at the current command. The camera keeps updating."""
    deadline = time.monotonic() + seconds
    while True:
        arm.refresh()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def _place_at(arm, view, place_joints: dict[str, float] | None, jump_m: float) -> bool:
    """Jump above the place point, lower, then fold to the joint targets."""
    joints = dict(place_joints or REST_JOINTS)
    view.report("Moving to place")
    if float(jump_m) <= 0:
        return _fold_to(arm, view, joints)
    sample = arm.get_joints()
    for name in JOINTS:
        sample[name] = float(joints[name])
    goal = arm.pose_at(sample)
    if not _jump_to(arm, view, goal.x, goal.y, goal.z, float(jump_m)):
        return False
    return _fold_to(arm, view, joints)


def _jump_to(arm, view, x: float, y: float, z: float, jump_m: float) -> bool:
    """Lift, travel above the target, then descend with fingers down."""
    cancel = view.has_pending
    pose = arm.get_pose()
    lift_z = max(pose.z, z) + jump_m
    mm = jump_m * 1000.0
    view.report(f"Lifting {mm:.0f} mm before place")
    if not arm.move_to(
            pose.x, pose.y, lift_z, wait=False, cancel=cancel) or cancel():
        view.report("Did not reach the place")
        return False
    view.report(f"Above place at {lift_z:.3f} m")
    if not arm.move_to(
            x, y, lift_z, wait=False, cancel=cancel) or cancel():
        view.report("Did not reach the place")
        return False
    view.report(f"Lowering to place at {z:.3f} m")
    if not arm.move_to(
            x, y, z, wait=False, cancel=cancel, downward=True) or cancel():
        view.report("Did not reach the place")
        return False
    return True


def _fold_to(arm, view, joints: dict[str, float]) -> bool:
    """Fold to joint targets. Return False if a newer click wins."""
    if not arm.fold(joints, wait=False, cancel=view.has_pending) or view.has_pending():
        view.report("Did not reach the place")
        return False
    return True


def _down_to(arm, view, x: float, y: float, z: float) -> bool:
    """Move with the fingers straight down. Return False if a newer click wins."""
    if not arm.move_to(
            x, y, z, wait=False, cancel=view.has_pending, downward=True,
    ) or view.has_pending():
        view.report("Did not reach the pick")
        return False
    return True


def pick_page_text() -> str:
    millimetres = PICK_HEIGHT_M * 1000.0
    clearance = APPROACH_M * 1000.0
    return (
        "Starts at the rest pose with the gripper open. "
        f"Click a point to pick {millimetres:.0f} mm above the table. "
        f"The arm stops {clearance:.0f} mm above that, fingers pointing down, "
        "then moves down, pauses, closes, pauses, lifts above the place point, folds in, and releases."
    )
