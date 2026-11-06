"""Click the camera image to move the gripper frame onto the base-link plane.

The click is a ray through the pixel. It hits z = 0 and the frame stops HOVER_M
above that point with the fingers pointing down. Click in the browser page; Rerun
cannot forward clicks on its image. --mock still needs the camera.

    uv run python application/click_and_go.py --mock
    uv run python application/click_and_go.py --live
"""

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import attach_camera, click_target, frame_viewer, run_click_loop
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.moves import CLICK_TOOL_OFFSET_M as EE_TOOL_OFFSET_M
from so101_utils.moves import HOVER_M, click_ready as ready, go_to
from so101_utils.views import watch_arm

if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    config = load_robot()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            text=(
                "Starts at the rest pose. Click a point on the table. "
                "The gripper frame moves to that spot, stopping "
                f"{HOVER_M * 1000:.0f} mm above it. "
                "A new click takes over from one that is still moving."
            ),
        )
        print(view.start())
        print("The camera is also in Rerun. Click the browser page above; Rerun cannot take the click.")
        print("Ctrl-C stops.")

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, view, show)
        print("Going to rest.")
        ready(arm, view)

        def on_click(click):
            target = click_target(click, cam.intrinsics, hand["base_T_camera"], HOVER_M)
            print(f"hover {target}")
            go_to(arm, view, target)

        run_click_loop(cam, view, show, on_click)
