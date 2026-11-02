"""Click the camera image in the native Rerun viewer to move the gripper.

The click is a ray through the pixel. It hits z = 0 and the frame stops HOVER_M
above that point with the fingers pointing down. The pixel is
`rr.poll_image_click` on the Camera panel, via `watch_arm(..., clicks=True)`.

Requires the Rerun fork submodule (`rerun`; see the README).

    uv run python application/click_and_go.py --mock
    uv run python application/click_and_go.py --live
"""

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import attach_camera, click_target, frame_viewer, read_frame
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.views import watch_arm

# Metres above the click, which lies on the base plane (z = 0).
HOVER_M = 0.02
# Metres along gripper_frame_link +Z. Zero puts the goal on that link's origin.
EE_TOOL_OFFSET_M = 0.0


def ready(arm) -> None:
    """Fold to the rest pose. A click waits until that move is sent."""
    print("Going to rest", flush=True)
    arm.rest(wait=False)


def go_to(arm, clicks, target) -> bool:
    """Command one hover point. False when a newer click has taken over."""
    x, y, z = (float(value) for value in target)
    where = f"{x:.3f}, {y:.3f}, {z:.3f} m"
    print(f"Moving to {where}", flush=True)
    arrived = arm.move_to(
        x, y, z, wait=False, cancel=clicks.has_pending, downward=True,
    )
    if arrived:
        print(f"Moved to {where}", flush=True)
    return arrived


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics, clicks=True)
        print("Click the Camera panel in Rerun. Ctrl-C stops.", flush=True)

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, show)
        ready(arm)
        try:
            while True:
                read_frame(cam, show)
                click = twin.clicks.poll()
                if click is None:
                    continue
                try:
                    target = click_target(click, cam.intrinsics, hand["base_T_camera"], HOVER_M)
                except ValueError as exc:
                    print(exc, flush=True)
                    continue
                print(f"hover {target}", flush=True)
                try:
                    go_to(arm, twin.clicks, target)
                except (ValueError, RuntimeError) as exc:
                    print(exc, flush=True)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            twin.close()
