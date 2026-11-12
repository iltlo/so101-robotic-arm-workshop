"""Click the camera image to pick there, then release at the rest pose.

The click is a ray through the pixel, hitting z = 0. The tool point
(EE_TOOL_OFFSET_M along gripper_frame_link +Z) stops PICK_HEIGHT_M above it,
approached from APPROACH_M higher with the fingers down. Click in the browser
page; Rerun cannot forward clicks on its image.

    uv run python application/click_and_pick.py --mock
    uv run python application/click_and_pick.py --live
"""

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import attach_camera, click_target, frame_viewer, run_click_loop
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.interrupts import ignore_sigint
from so101_utils.moves import APPROACH_M, DWELL_S, PICK_HEIGHT_M
from so101_utils.moves import PICK_TOOL_OFFSET_M as EE_TOOL_OFFSET_M
from so101_utils.moves import hold as _hold
from so101_utils.moves import pick_and_return as _pick_and_return
from so101_utils.moves import pick_page_text as _page_text
from so101_utils.moves import pick_ready as ready
from so101_utils.views import watch_arm


def pick_and_return(arm, view, target) -> bool:
    """Use this module's DWELL_S so a test can shorten the pause."""
    return _pick_and_return(arm, view, target, dwell_s=DWELL_S)


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

        def on_click(click):
            target = click_target(click, cam.intrinsics, hand["base_T_camera"], PICK_HEIGHT_M)
            print(f"pick {target}")
            try:
                pick_and_return(arm, view, target)
            except (ValueError, RuntimeError) as exc:
                view.report(str(exc))
                print(exc)
            # Clicks made while the pick ran are dropped, not queued.
            view.consume()

        run_click_loop(cam, view, show, on_click)
        print("Going to rest.")
        ignore_sigint()
        arm.rest()
