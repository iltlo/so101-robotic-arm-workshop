"""Click the camera image to move the TCP onto the base-link plane.

The click is a ray through the pixel. It hits z = 0, then the arm stops
HOVER_M metres above that point. This does not use the depth camera.

The same image is shown in Rerun next to the arm. This Rerun version cannot
send a click on that image back to Python, so the click still happens in the
browser page printed at startup. --mock moves the arm in that Rerun window.

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


def go_to(arm, view, target) -> bool:
    """Command one hover point. Return False when a newer click has taken over.

    The move is not waited out. Live feedback sometimes stays outside the
    settle band, and the next click used to sit there until the 60s timeout.
    """
    view.report(f"Moving to {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
    arrived = arm.move_to(
        float(target[0]), float(target[1]), float(target[2]),
        wait=False, cancel=view.has_pending,
    )
    if arrived:
        view.report(f"Moved to {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
    return arrived


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    plane = base_link_plane()
    config = load_robot()
    with SO101.connect(mock=args.mock) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], camera=True)
        view = ClickView(cam.intrinsics.width, cam.intrinsics.height)
        print(view.start())
        print("The camera is also in Rerun. Click the browser page above; Rerun cannot take the click.")
        print("Ctrl-C stops.")
        try:
            while True:
                frame = cam.read()
                view.publish(frame.rgb_bgr)
                twin.show(arm)
                twin.log_camera(frame.rgb_bgr)
                twin.rr.get_global_data_recording().flush(timeout_sec=0)
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
                twin.show_hover_goal(target)
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
