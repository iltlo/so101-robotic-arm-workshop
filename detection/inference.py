"""Live YOLO OBB inference on the configured camera.

Uses the same RealSense (or demo) profile as the workshop, draws the oriented
boxes in the Rerun window, and prints the count. Ctrl-C stops.

    uv run python detection/inference.py
    uv run python detection/inference.py --model detection/model/demo.pt --conf 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from so101_utils.camera import open_camera
from so101_workshop.yolo import YoloDetector, draw_boxes

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "detection" / "model" / "demo.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence")
    args = parser.parse_args()

    detector = YoloDetector(args.model, confidence=args.conf)
    detector.load()
    print("Ctrl-C stops. The boxes are drawn in Rerun, not on a desktop window.")
    rr.init("SO-101 YOLO", spawn=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial2DView(name="YOLO", origin="camera/rgb"),
        auto_views=False,
    ))
    with open_camera() as cam:
        try:
            while True:
                frame = cam.read()
                parsed = detector.detect(frame.rgb_bgr)
                annotated = draw_boxes(frame.rgb_bgr, parsed["boxes"])
                rr.log("camera/rgb", rr.Image(np.ascontiguousarray(annotated[:, :, ::-1])))
                rr.get_global_data_recording().flush(timeout_sec=0)
                labels = ", ".join(box["label"] for box in parsed["boxes"])
                print(f"\r{len(parsed['boxes'])} boxes {labels}   ", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
