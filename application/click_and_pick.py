"""Click the camera image in the native Rerun viewer to pick, then release at rest.

The click is a ray through the pixel, hitting z = 0. The tool point
(EE_TOOL_OFFSET_M along gripper_frame_link +Z) stops PICK_HEIGHT_M above it,
approached from APPROACH_M higher with the fingers down. Clicks come from the
Camera panel through `watch_arm(..., clicks=True)`.

    uv run python application/click_and_pick.py --mock
    uv run python application/click_and_pick.py --live
"""

import time

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import attach_camera, click_target, frame_viewer, read_frame
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


def ready(arm) -> None:
    """Fold to the rest pose and open the gripper.

    `rest` closes the jaw, so the open has to come after the fold.
    """
    print("Going to rest", flush=True)
    arm.rest(wait=False)
    arm.release()


def pick_and_return(arm, clicks, target) -> bool:
    """Hover above target, move down, rest, and open.

    Both poses point the fingers straight down; the twist is re-solved on the way
    down. A newer click skips the close.
    """
    x, y, z = float(target[0]), float(target[1]), float(target[2])
    hover_z = z + APPROACH_M
    print(f"Moving above the pick: {x:.3f}, {y:.3f}, {hover_z:.3f} m", flush=True)
    if not _down_to(arm, clicks, x, y, hover_z):
        return False
    print(f"Moving down to {x:.3f}, {y:.3f}, {z:.3f} m", flush=True)
    if not _down_to(arm, clicks, x, y, z):
        return False
    print("Holding the pick", flush=True)
    _hold(arm, DWELL_S)
    print("Closing gripper", flush=True)
    arm.grip()
    print("Holding the grasp", flush=True)
    _hold(arm, DWELL_S)
    print("Returning to rest", flush=True)
    arm.rest(wait=False)
    arm.release()
    print("Released at rest", flush=True)
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


def _down_to(arm, clicks, x: float, y: float, z: float) -> bool:
    """Move with the fingers straight down. Return False if a newer click wins."""
    if not arm.move_to(
            x, y, z, wait=False, cancel=clicks.has_pending, downward=True,
    ) or clicks.has_pending():
        print("Did not reach the pick", flush=True)
        return False
    return True


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics, clicks=True)
        print("Click the Camera panel in Rerun. Ctrl-C stops.", flush=True)

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, show)
        print("Going to rest, gripper open.", flush=True)
        ready(arm)
        try:
            while True:
                read_frame(cam, show)
                click = twin.clicks.poll()
                if click is None:
                    continue
                try:
                    target = click_target(click, cam.intrinsics, hand["base_T_camera"], PICK_HEIGHT_M)
                except ValueError as exc:
                    print(exc, flush=True)
                    continue
                print(f"pick {target}", flush=True)
                try:
                    pick_and_return(arm, twin.clicks, target)
                except (ValueError, RuntimeError) as exc:
                    print(exc, flush=True)
                # Clicks made while the pick ran are dropped, not queued.
                twin.clicks.poll()
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            twin.close()
