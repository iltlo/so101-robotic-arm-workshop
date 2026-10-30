"""Click the camera image to pick there, then release at the rest pose.

The click is a ray through the pixel, hitting z = 0. The tool point
(EE_TOOL_OFFSET_M along gripper_frame_link +Z) stops PICK_HEIGHT_M above it,
approached from APPROACH_M higher with the fingers down. Click in the browser
page; Rerun cannot forward clicks on its image.

    uv run python application/click_and_pick.py --mock
    uv run python application/click_and_pick.py --live
"""

import time

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import attach_camera, click_target, frame_viewer, read_frame
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.views import watch_arm

# Metres above the click, which lies on the base plane (z = 0).
PICK_HEIGHT_M = 0.00
# Metres above the pick. The arm stops here, then moves straight down.
APPROACH_M = 0.080
# Hold the pick before the jaw closes, and hold the grasp before rest.
DWELL_S = 0.3
# Metres along gripper_frame_link +Z.
EE_TOOL_OFFSET_M = 0.025


def ready(arm, view=None) -> None:
    """Fold to the rest pose and open the gripper.

    `rest` closes the jaw, so the open has to come after the fold.
    """
    if view is not None:
        view.report("Going to rest")
    arm.rest(wait=False)
    arm.release()


def pick_and_return(arm, view, target) -> bool:
    """Hover above target, move down, rest, and open.

    Both poses point the fingers straight down; the twist is re-solved on the way
    down. A newer click skips the close.
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    hover_z = z + APPROACH_M
    view.report(f"Moving above the pick: {x:.3f}, {y:.3f}, {hover_z:.3f} m")
    if not _down_to(arm, view, x, y, hover_z):
        return False
    view.report(f"Moving down to {x:.3f}, {y:.3f}, {z:.3f} m")
    if not _down_to(arm, view, x, y, z):
        return False
    view.report("Holding the pick")
    _hold(arm, DWELL_S)
    view.report("Closing gripper")
    arm.grip()
    view.report("Holding the grasp")
    _hold(arm, DWELL_S)
    view.report("Returning to rest")
    arm.rest(wait=False)
    arm.release()
    view.report("Released at rest")
    return True


def _hold(arm, seconds: float) -> None:
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


def _page_text() -> str:
    millimetres = PICK_HEIGHT_M * 1000.0
    clearance = APPROACH_M * 1000.0
    return (
        "Starts at the rest pose with the gripper open. "
        f"Click a point to pick {millimetres:.0f} mm above the table. "
        f"The arm stops {clearance:.0f} mm above that, fingers pointing down, "
        "then moves down, pauses, closes, pauses, folds to the rest pose, and releases."
    )


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            title="Click and pick", text=_page_text(),
        )
        print(view.start())
        print("The camera is also in Rerun. Click the browser page above; Rerun cannot take the click.")
        print("Ctrl-C stops.")

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, view, show)
        print("Going to rest, gripper open.")
        ready(arm, view)
        try:
            while True:
                read_frame(cam, view, show)
                click = view.consume()
                if click is None:
                    continue
                try:
                    target = click_target(click, cam.intrinsics, hand["base_T_camera"], PICK_HEIGHT_M)
                except ValueError as exc:
                    view.report(str(exc))
                    print(exc)
                    continue
                print(f"pick {target}")
                try:
                    pick_and_return(arm, view, target)
                except (ValueError, RuntimeError) as exc:
                    view.report(str(exc))
                    print(exc)
                # Clicks made while the pick ran are dropped, not queued.
                view.consume()
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            view.close()
