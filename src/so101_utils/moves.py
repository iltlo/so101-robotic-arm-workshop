"""Click-and-go and click-and-pick. Both call the arm; neither opens a window."""

import time

# Metres above the click, which lies on the base plane (z = 0).
HOVER_M = 0.02
# Metres along gripper_frame_link +Z. Zero puts the click goal on that link's origin.
CLICK_TOOL_OFFSET_M = 0.0
# Metres above the click for a pick. The tool point is PICK_TOOL_OFFSET_M out the fingers.
PICK_HEIGHT_M = 0.00
APPROACH_M = 0.080
DWELL_S = 0.3
PICK_TOOL_OFFSET_M = 0.025
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
                    open_delay_s: float = DWELL_S) -> bool:
    """Hover above target, move down, rest, and open.

    Both poses point the fingers straight down; the twist is re-solved on the way
    down. `approach_m` is the hover height above the target. `close_delay_s` and
    `open_delay_s` pause after the grip and after the release. A newer click
    skips the close.
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    hover_z = z + approach_m
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
    view.report("Returning to rest")
    arm.rest(wait=False)
    arm.release()
    view.report("Released at rest")
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
        "then moves down, pauses, closes, pauses, folds to the rest pose, and releases."
    )
