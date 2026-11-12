"""Run a YOLO OBB model on one of the raw images and save an annotated copy.

    uv run python detection/test.py
    uv run python detection/test.py --image raw_image/IMG_01.jpg
    uv run python detection/test.py --model detection/model/demo.pt --conf 0.5
"""

import argparse
from pathlib import Path

import cv2

from so101_workshop.yolo import YoloDetector, draw_boxes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "raw_image" / "IMG_09.jpg"
DEFAULT_MODEL = ROOT / "detection" / "model" / "demo.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="image to inspect")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence")
    parser.add_argument("--out", help="annotated output path")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    detector = YoloDetector(args.model, confidence=args.conf)
    parsed = detector.detect(image)
    annotated = draw_boxes(image, parsed["boxes"])

    out_path = Path(args.out) if args.out else image_path.with_name(
        f"{image_path.stem}_annotated{image_path.suffix}"
    )
    cv2.imwrite(str(out_path), annotated)
    labels = ", ".join(
        f"{box['label']} {box['score']:.1f}%" for box in parsed["boxes"]
    )
    print(f"{len(parsed['boxes'])} detections: {labels or 'none'}")
    print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
