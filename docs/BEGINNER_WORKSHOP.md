# Beginner workshop commands

The [Dobot arm basic workshop](https://github.com/HKUArmStrong/dobot-robotic-arm-workshop) on the SO-101 follower. Same slide order; the commands and units differ.

Setup: `git lfs pull` for `detection/model/demo.pt`, set the port once with `uv run python scripts/check_port.py`, then start the web UI from the repository root with `uvx --python 3.13 --from . so101-workshop run --port 8032` (or `uv run so101-workshop run --port 8032` after `uv sync`). The published package does not have this branch's changes.

## Commands

- **02.1 API** — `arm.home()`, `arm.get_pose()`, `arm.move_to(x, y, z, wait=True)`, `arm.grip()` / `arm.release()`. Metres and degrees; add `downward=True` for a top-down pick.
- **02.2 Move the arm** — `uv run python scripts/arm_move.py --live`
- **02.3 Read the pose** — `uv run python scripts/get_pose.py --live`
- **02.4 Gripper** — `uv run python scripts/gripper.py --live` (toggles until Ctrl-C)
- **02.5 Custom task** — edit `PICK_LOCATION` and `PLACE_LOCATION` in `application/pick_and_place.py`, then `uv run python application/pick_and_place.py --live`
- **03.1 AprilTag view** — `uv run python calibration/apriltag.py`
- **03.2 Hand-eye calibration** — clear the table, then `uv run python calibration/auto_hand_eye.py --live`
- **03.3 Click and go** — `uv run python application/click_and_go.py --live`, then click in the browser page
- **04.1 YOLO inference** — `uv run python detection/inference.py`
- **04.2 Screw click pick** — `uv run python application/detect_and_pick.py --live`, then click a box

## Notes

- `--mock` runs an arm script without the USB arm; camera scripts still open the camera.
- Every script that moves the arm ends folded at rest. Ctrl-C is not an e-stop.
- The web UI runs the same tasks without a terminal, with the e-stop in the top bar.
