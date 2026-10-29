"""Click the camera image to pick there, then release at the rest pose.

The click is a ray through the pixel. It hits z = 0, and the tool point stops
PICK_HEIGHT_M metres above that point. The tool point is 25 mm along gripper
−X. The arm first stops APPROACH_M higher, fingers pointing straight down,
then moves down to the pick. Each pose solves
that aim on its own. This does not use the depth camera.

The arm starts at the rest pose with the gripper open. A click moves to the
pick, holds there, closes the gripper, holds again, folds to the rest pose,
and opens. The next
click is taken only once that cycle has finished.

The same image is shown in Rerun next to the arm. This Rerun version cannot
send a click on that image back to Python, so the click still happens in the
browser page printed at startup. --mock moves the arm in that Rerun window.

    uv run python application/click_and_pick.py --mock
    uv run python application/click_and_pick.py --live
"""

import time

import numpy as np

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.transforms import base_link_plane, intersect_pixel
from so101_utils.twin import watch_arm

# Metres above the base plane. The click itself is the z = 0 intersection.
# Raise this if the gripper would hit the table.
PICK_HEIGHT_M = 0.00
# Metres above the pick. The arm stops here, then moves straight down.
APPROACH_M = 0.080
# Hold the pick before the jaw closes, and hold the grasp before rest.
DWELL_S = 0.3
# Metres along gripper X. Negative is opposite the red arrow. 25 mm.
EE_TOOL_OFFSET_M = -0.025


def ready(arm, view=None) -> None:
    """Fold to the rest pose and open the gripper.

    A click waits until that move is sent. `rest` closes the jaw, so the open
    comes after the fold. The first pick would otherwise approach already shut.
    The settle wait is skipped. A click used to cancel this move and leave the
    solver on the halfway joints, so every later click in that process missed.
    """
    if view is not None:
        view.report("Going to rest")
    arm.rest(wait=False)
    arm.release()


def attach_camera(arm, cam, view, show) -> None:
    """Publish the camera during moves, and one frame before returning.

    Resting used to block the page loop, so /frame stayed empty
    and the browser kept the "Live RGB" placeholder.
    """

    def paint() -> None:
        frame = cam.poll()
        if frame is not None:
            view.publish(frame.rgb_bgr)
        show(frame)

    arm.track(paint)
    first = cam.read()
    view.publish(first.rgb_bgr)
    show(first)


def pick_and_return(arm, view, target) -> bool:
    """Hover above target, move down, then rest and open.

    Both poses point gripper +Z straight down. The twist about that axis is
    solved again on the way down. A newer click skips the close. Moves finish
    when the goal is sent. They do not wait for the 2° settle. The arm then
    holds the pick, closes, holds the grasp, and folds to rest.
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    hover_z = z + APPROACH_M
    view.report(f"Moving above {x:.3f}, {y:.3f}, {hover_z:.3f} m")
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
        refresh = getattr(arm, "_on_motion", None)
        if refresh is not None:
            refresh()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def _down_to(arm, view, x: float, y: float, z: float) -> bool:
    """Move with +Z straight down. Return False if a newer click wins."""
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
        f"The arm stops {clearance:.0f} mm above that, gripper Z axis pointing down, "
        "then moves down, pauses, closes, pauses, folds to the rest pose, and releases."
    )


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    plane = base_link_plane()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], camera=True)
        twin.mark_ee_tool(arm.ee_tool_offset)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            title="Click and pick", text=_page_text(),
        )
        print(view.start())
        print("The camera is also in Rerun. Click the browser page above; Rerun cannot take the click.")
        print("Ctrl-C stops.")

        def show(frame) -> None:
            if frame is not None:
                twin.log_camera(frame.rgb_bgr)
            twin.show(arm)
            twin.rr.get_global_data_recording().flush(timeout_sec=0)

        attach_camera(arm, cam, view, show)
        print("Going to rest, gripper open.")
        ready(arm, view)
        try:
            while True:
                frame = cam.read()
                view.publish(frame.rgb_bgr)
                show(frame)
                click = view.consume()
                if click is None:
                    continue
                x, y = click
                try:
                    # Ray from the camera through the pixel, onto z = 0.
                    point = intersect_pixel(
                        (x, y), cam.intrinsics, hand["base_T_camera"],
                        plane["point"], plane["normal"])
                    target = point + np.array([0.0, 0.0, PICK_HEIGHT_M])
                except ValueError as exc:
                    view.report(str(exc))
                    print(exc)
                    continue
                print(f"plane {point}, pick {target}")
                try:
                    pick_and_return(arm, view, target)
                except (ValueError, RuntimeError) as exc:
                    view.report(str(exc))
                    print(exc)
                view.consume()
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            view.close()
