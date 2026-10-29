"""Click the camera image to move the gripper frame onto the base-link plane.

The click is a ray through the pixel. It hits z = 0, then the gripper frame
stops HOVER_M metres above that point, with gripper +Z pointing straight down.
The tool offset is 0, so that point is the frame origin. This does not use
the depth camera.

The arm starts at the rest pose. The same image is shown in Rerun next to the
arm. This Rerun version cannot send a click on that image back to Python, so
the click still happens in the browser page printed at startup. --mock moves
the arm in that Rerun window.

A new click replaces a move that has not finished. The page polls /status,
so the line under the image leaves "Queued" without another click.

    uv run python application/click_and_go.py --mock
    uv run python application/click_and_go.py --live
"""

import numpy as np

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.transforms import base_link_plane, intersect_pixel
from so101_utils.twin import watch_arm

# Metres above the base plane. The click itself is the z = 0 intersection.
HOVER_M = 0.02
# Metres along gripper +Z. Zero puts the goal on the gripper frame origin.
EE_TOOL_OFFSET_M = 0.0


def ready(arm, view=None) -> None:
    """Fold to the rest pose. A click waits until that move is sent.

    The settle wait is skipped. A click used to cancel this fold and leave the
    solver on the halfway joints, so every later click in that process missed.
    """
    if view is not None:
        view.report("Going to rest")
    arm.rest(wait=False)


def attach_camera(arm, cam, view, show) -> None:
    """Publish the camera during the rest move, and one frame before returning.

    Resting used to run first and block the page loop, so /frame stayed empty
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


def go_to(arm, view, target) -> bool:
    """Command one hover point. Return False when a newer click has taken over.

    The move is not waited out. Live feedback sometimes stays outside the
    settle band, and the next click used to sit there until the 60s timeout.
    """
    view.report(f"Moving to {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
    arrived = arm.move_to(
        float(target[0]), float(target[1]), float(target[2]),
        wait=False, cancel=view.has_pending, downward=True,
    )
    if arrived:
        view.report(f"Moved to {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
    return arrived


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    plane = base_link_plane()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], camera=True)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            text=(
                "Starts at the rest pose. Click a point on the table. "
                "The gripper frame moves to that spot, stopping 2 cm above it. "
                "A new click takes over from one that is still moving."
            ),
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
        print("Going to rest.")
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
                    target = point + np.array([0.0, 0.0, HOVER_M])
                except ValueError as exc:
                    view.report(str(exc))
                    print(exc)
                    continue
                print(f"plane {point}, hover {target}")
                try:
                    go_to(arm, view, target)
                except (ValueError, RuntimeError) as exc:
                    view.report(str(exc))
                    print(exc)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            view.close()
