"""Click a detection box to pick its centre, then release at the rest pose.

The reCamera sends YOLO boxes beside the JPEG. A click selects the smallest
box under it, and the arm goes to that box's centre rather than to the click
point. Rerun cannot forward clicks, so click in the browser page.

    uv run python application/detect_and_pick.py --live
    uv run python application/detect_and_pick.py --mock
"""

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.click_app import (
    attach_camera,
    click_target,
    frame_viewer,
    run_click_loop,
    select_box,
)
from so101_utils.click_view import ClickView
from so101_utils.config import arm_parser, load_hand_eye, load_robot
from so101_utils.moves import DWELL_S, PICK_HEIGHT_M
from so101_utils.moves import PICK_TOOL_OFFSET_M as EE_TOOL_OFFSET_M
from so101_utils.moves import pick_and_return as _pick_and_return
from so101_utils.moves import pick_ready as ready
from so101_utils.views import watch_arm
from so101_workshop.detections import DetectionStore, parse_detections


PAGE_TEXT = (
    "Click inside a detected box. The arm picks the centre of that box, not the "
    "click point. Boxes come from the camera model; a click selects the smallest "
    "box under it. A new click takes over from one that is still moving."
)


def pick_and_return(arm, view, target) -> bool:
    """Use this module's DWELL_S so a test can shorten the pause."""
    return _pick_and_return(arm, view, target, dwell_s=DWELL_S)


def _store_payload(store: DetectionStore, raw) -> None:
    try:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        store.update(parse_detections(text), source="recamera")
    except (UnicodeDecodeError, TypeError, ValueError):
        return


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    hand = load_hand_eye()
    config = load_robot()
    detections = DetectionStore()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, \
            open_camera(on_payload=lambda raw: _store_payload(detections, raw)) as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            title="Detect and pick", text=PAGE_TEXT,
        )
        print(view.start())
        print("Click a detection box in the browser page; Rerun cannot take the click.")
        print("Ctrl-C stops.")

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, view, show)
        print("Going to rest, gripper open.")
        ready(arm, view)

        def on_click(click):
            snapshot = detections.snapshot()
            if not snapshot["boxes"]:
                view.report("No detection boxes yet")
                return
            index, box, centre = select_box(
                click, snapshot["boxes"], snapshot["resolution"], cam.intrinsics,
            )
            print(f"pick box {index} ({box['label']}) centre {centre}")
            try:
                target = click_target(
                    centre, cam.intrinsics, hand["base_T_camera"], PICK_HEIGHT_M,
                )
                if pick_and_return(arm, view, target):
                    view.report(f"Picked {box['label']} at the box centre {centre}")
            except (ValueError, RuntimeError) as exc:
                view.report(str(exc))
                print(exc)
            # Clicks made while the pick ran are dropped, not queued.
            view.consume()

        run_click_loop(cam, view, show, on_click)
