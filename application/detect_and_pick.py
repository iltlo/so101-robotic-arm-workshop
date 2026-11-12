"""Screw click pick: click a detected box and release at the rest pose.

Dobot's `screw_click_pick.py`. A local YOLO OBB model runs on the RealSense
frames and draws the boxes on the published image. A click selects the smallest
box under it, and the arm goes to that box's centre rather than to the click
point. Rerun cannot forward clicks, so click in the browser page.

    uv run python application/detect_and_pick.py --live
    uv run python application/detect_and_pick.py --mock
"""

from pathlib import Path

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
from so101_utils.interrupts import ignore_sigint
from so101_utils.moves import DWELL_S
from so101_utils.moves import PICK_TOOL_OFFSET_M as EE_TOOL_OFFSET_M
from so101_utils.moves import pick_and_return as _pick_and_return
from so101_utils.moves import pick_ready as ready
from so101_utils.views import watch_arm
from so101_workshop.detections import DetectionStore
from so101_workshop.yolo import YoloDetector, draw_boxes

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "detection" / "model" / "demo.pt"

# ============== CONFIGURATION CONSTANTS ==============

# Metres above the pick point. The arm lifts there before moving down.
APPROACH_HEIGHT_OFFSET_M = 0.050

# Metres above the table plane for the tool point. The plane stands in for the
# object's top, so keep this near zero unless the table needs clearance.
PICK_HEIGHT_OFFSET_M = 0.000

# Seconds to wait after closing and after opening the gripper.
GRIP_CLOSE_DELAY_S = 1.5
GRIP_OPEN_DELAY_S = 0.5

# Minimum YOLO confidence.
DETECTION_CONFIDENCE_THRESHOLD = 0.25

# ============== END CONFIGURATION ==============

PAGE_TEXT = (
    "Click inside a detected box. The arm picks the centre of that box, not the "
    f"click point, {PICK_HEIGHT_OFFSET_M * 1000:.0f} mm above the table. It stops "
    f"{APPROACH_HEIGHT_OFFSET_M * 1000:.0f} mm above that, then moves down, "
    "closes, and folds back to rest. A click selects the smallest box under it; "
    "a new click takes over from one that is still moving."
)


def pick_and_return(arm, view, target) -> bool:
    """Here so a test can shorten the pauses."""
    return _pick_and_return(
        arm, view, target,
        dwell_s=DWELL_S,
        approach_m=APPROACH_HEIGHT_OFFSET_M,
        close_delay_s=GRIP_CLOSE_DELAY_S,
        open_delay_s=GRIP_OPEN_DELAY_S,
    )


def main(model: str = str(DEFAULT_MODEL),
         confidence: float = DETECTION_CONFIDENCE_THRESHOLD) -> None:
    parser = arm_parser(__doc__)
    parser.add_argument("--model", default=model, help="YOLO OBB weights")
    parser.add_argument("--conf", type=float, default=confidence, help="minimum confidence")
    args = parser.parse_args()

    detector = YoloDetector(args.model, confidence=args.conf)
    detector.load()
    hand = load_hand_eye()
    config = load_robot()
    detections = DetectionStore()
    with SO101.connect(mock=args.mock, ee_tool_offset=EE_TOOL_OFFSET_M) as arm, \
            open_camera() as cam:
        twin = watch_arm(arm, config["urdf"], intrinsics=cam.intrinsics)
        view = ClickView(
            cam.intrinsics.width, cam.intrinsics.height,
            title="Screw click pick", text=PAGE_TEXT,
        )
        print(view.start())
        print("Click a detection box in the browser page; Rerun cannot take the click.")
        print("Ctrl-C stops.")

        show = frame_viewer(arm, twin)
        attach_camera(arm, cam, view, show)
        print("Going to rest, gripper open.")
        ready(arm, view)

        def prepare(image):
            parsed = detector.detect(image)
            detections.update(parsed, source="yolo")
            return draw_boxes(image, parsed["boxes"])

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
                    centre, cam.intrinsics, hand["base_T_camera"], PICK_HEIGHT_OFFSET_M,
                )
                if pick_and_return(arm, view, target):
                    view.report(f"Picked {box['label']} at the box centre {centre}")
            except (ValueError, RuntimeError) as exc:
                view.report(str(exc))
                print(exc)
            # Clicks made while the pick ran are dropped, not queued.
            view.consume()

        run_click_loop(cam, view, show, on_click, prepare)
        print("Going to rest.")
        ignore_sigint()
        arm.rest()


if __name__ == "__main__":
    main()
