"""Measure RGB intrinsics from a chessboard and write config/camera.yaml.

The lab board has 8 by 11 inner corners (9 by 12 squares). Square size is
not required: it does not change fx, fy, or the distortion. Keys (X11):
Space captures a view, Enter solves and saves, Q quits. Move the board
between captures.
"""

import cv2

from so101_utils.camera import open_camera
from so101_utils.config import save_camera_intrinsics
from so101_utils.intrinsics import (
    INNER_CORNERS,
    WORKSHOP_MIN_VIEWS,
    capture_chessboard,
    draw_chessboard,
    find_chessboard,
    solve_captured,
)
from so101_utils.keys import keyboard
from so101_utils.views import ChessboardView

MIN_VIEWS = WORKSHOP_MIN_VIEWS
# Rerun at 1920x1080 filled its log and made the picture late.
_PREVIEW_WIDTH = 960


def _preview(image):
    height, width = image.shape[:2]
    if width <= _PREVIEW_WIDTH:
        return image
    preview_height = int(round(height * _PREVIEW_WIDTH / width))
    return cv2.resize(image, (_PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    samples = []
    saved = False
    cols, rows = INNER_CORNERS
    print(f"Show the {cols}x{rows} inner-corner chessboard. Space captures. "
          f"Enter solves after {MIN_VIEWS} views. Q quits.")
    with open_camera() as cam, keyboard(("space", "enter", "q")) as keys:
        view = ChessboardView()
        while True:
            frame = cam.read().rgb_bgr
            found = find_chessboard(frame, INNER_CORNERS)
            corners, pattern = (None, INNER_CORNERS) if found is None else found
            count = len(samples)
            status = f"{count} views" + ("  board found" if corners is not None else "  no board")
            view.update(_preview(draw_chessboard(frame, corners, status, pattern)))
            command = keys.consume()
            if command == "q":
                print("Saved." if saved else f"Quit. Discarded {count} views.")
                break
            if command == "space":
                try:
                    samples.append(capture_chessboard(frame))
                except ValueError as exc:
                    print(exc)
                    continue
                print(f"Captured {len(samples)} views.")
                continue
            if command != "enter":
                continue
            height, width = frame.shape[:2]
            try:
                intr, rms = solve_captured(samples, (width, height), minimum=MIN_VIEWS)
            except (ValueError, RuntimeError, cv2.error) as exc:
                print(f"Calibration failed: {exc}")
                continue
            save_camera_intrinsics(intr, serial=None)
            saved = True
            print(f"Saved {width}x{height}  fx={intr.fx:.2f} fy={intr.fy:.2f} "
                  f"cx={intr.cx:.2f} cy={intr.cy:.2f}  RMS {rms:.3f} px")
            print("Rerun hand-eye before a live click. The old base_T_camera is the RealSense pose.")
