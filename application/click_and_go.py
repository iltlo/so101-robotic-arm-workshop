"""Click the camera image to move the TCP onto the base-link plane.

The click is a ray through the pixel. It hits z = 0, then the arm stops
HOVER_M metres above that point. This does not use the depth camera.

    uv run python application/click_and_go.py --mock
    uv run python application/click_and_go.py --live
"""

import numpy as np

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.transforms import base_link_plane, intersect_pixel
from so101_utils.twin import Twin

# Metres above the base plane. The click itself is the z = 0 intersection.
HOVER_M = 0.02


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    plane = base_link_plane()
    config = load_robot()
    with SO101.connect(mock=args.mock) as arm, open_camera() as cam:
        arm.set_speed(10)
        twin = Twin(config["urdf"])
        view = ClickView(cam.intrinsics.width, cam.intrinsics.height)
        print(view.start())
        print("Click the image. Ctrl-C stops.")
        try:
            while True:
                frame = cam.read()
                view.publish(frame.rgb_bgr)
                twin.show(arm)
                for x, y in view.consume():
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
                        arm.move_to(*target)
                    except (ValueError, RuntimeError) as exc:
                        view.report(str(exc))
                        print(exc)
                    else:
                        view.report(f"Moved to {target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f} m")
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            view.close()
