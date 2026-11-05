# SO-101 workshop

A short path from a fresh checkout to moving the follower. Distances are **metres**. Joint angles are **degrees**. The arm class is `SO101`. Camera, transforms, and the Rerun viewer live in `so101_utils` and stay out of the scripts.

## Features

- [x] Digital twin (`scripts/twin.py`)
- [x] Read the TCP pose (`scripts/get_pose.py`)
- [x] Move to a few points (`scripts/arm_move.py`)
- [x] Gripper open / close (`scripts/gripper.py`)
- [x] Joint jog (`scripts/joint_jog.py`)
- [x] Keyboard teleop (`scripts/keyboard_control.py`)
- [x] AprilTag in the camera frame (`calibration/apriltag.py`)
- [x] Chessboard intrinsics (`calibration/intrinsics.py`)
- [x] Eye-to-hand calibration (`calibration/hand_eye.py`)
- [x] Click-and-go on the base plane z = 0 (`application/click_and_go.py`)
- [x] Click and pick: pick at the click, release at rest (`application/click_and_pick.py`)
- [x] Pick and place (`application/pick_and_place.py`)

Not in this workshop: YOLO, depth picking, and the interactive motor-calibration UI. Motor midpoint and ranges are recorded by `scripts/calibrate_motors.py` into `config/motors.json`.

## Safety

Clear the table before any `--live` command. Joint speed defaults to 90 deg/s, which is also the maximum. Change it with `arm.set_speed(...)`. Torque off (`T` in the twin, or during hand-eye) lets the arm fall: support it, and keep a power cutoff within reach. Ctrl-C and the keyboard are not emergency stops. There is no collision checker.

## 1. Install

From the repository root, with [uv](https://docs.astral.sh/uv/) and Python 3.12. This checkout also has an older `.venv`; point uv at `.uv-so101` so it does not install into that one. Repeat the `export` in each new terminal.

```sh
export UV_PROJECT_ENVIRONMENT=.uv-so101
unset PYTHONPATH
uv sync --locked --all-extras
uv run python scripts/fetch_so101_assets.py
uv run python -m pytest -q
```

The asset script downloads the SO-101 URDF and its meshes. It needs a network once.

## 2. Find the USB port

Plug in the USB adapter. Motors can stay off. The script asks you to unplug it and writes `port` in `config/robot.yaml`.

```sh
uv run python scripts/check_port.py
```

## 3. Calibrate the motors

Do this once, before any live motion. Use the port from `config/robot.yaml`.

```sh
uv run python scripts/calibrate_motors.py
```

Torque turns off and the arm is moved by hand, so support it. The midpoint and each joint range are saved to `config/motors.json`. Wrist roll keeps a full turn, so it is not swept. A later connect loads that file onto the servos. `joint_limits_deg` and `joint_offsets_deg` in `config/robot.yaml` stay the limits and the small zero corrections for this arm. Edit the offsets only if you re-measure them. 0° is the calibrated midpoint.

## 4. See the arm

Mock first. It plays five poses in Rerun and does not open the serial port.

```sh
uv run python scripts/twin.py --mock
uv run python scripts/twin.py --live
```

`--live` only reads joint feedback. Add `--relax` if you want `T` to turn torque off while you pose the arm by hand.

## 5. Read the pose, move, and use the gripper

```sh
uv run python scripts/get_pose.py --mock
uv run python scripts/get_pose.py --live

uv run python scripts/arm_move.py --mock
uv run python scripts/arm_move.py --live

uv run python scripts/gripper.py --mock
uv run python scripts/gripper.py --live
```

`arm_move.py` homes (every arm joint to 0°) and visits three points. Change `POINTS` in that file if you want different ones. Gripper percent is 0 closed and 100 fully open. The open command is 50. `--mock` on these scripts opens Rerun and moves the drawn arm.

To step a single joint by at most 2°:

```sh
uv run python scripts/joint_jog.py --mock
```

Type `shoulder_pan 1`, then `q`.

Keyboard teleop (WASD / arrows move x and y, Q/E move z, and those translation keys keep the current orientation. Z/C change base roll, F/V change base pitch, R rests, H homes, space toggles the gripper, Esc quits). Rerun shows the arm, joint angles, and end-effector pose. `--mock` moves that drawing and does not open the serial port.

```sh
uv run python scripts/keyboard_control.py --mock
uv run python scripts/keyboard_control.py --live
```

## Camera view

The live picture only, in Rerun. Q quits. `calibration/intrinsics.py` searches every frame for the chessboard, so that view updates more slowly.

```sh
uv run python scripts/camera_view.py
```

## 6. AprilTag

Camera only. No arm. The tag is tag36h11, id 0, 40 mm black-square edge, as set in the `tag:` section of `config/robot.yaml`. Ctrl-C stops.

```sh
uv run python calibration/apriltag.py
```

## Chessboard intrinsics

Run this before hand-eye. The webcam and the reCamera do not report fx, fy, or distortion. The lab board has 8 × 11 inner corners (9 × 12 squares). Square size is not needed. Hold the board in view, move it between captures (tilt it, and use different parts of the image), and take at least ten views.

**Space** captures the current view. **Enter** solves and writes `config/camera.yaml`. **Q** quits. The fit is Brown-Conrady, so later rays use OpenCV undistortion.

```sh
uv run python calibration/intrinsics.py
```

## 7. Hand-eye

The camera stays fixed. The same tag is mounted on `gripper_link` at [-37.2, 0, -28.4] mm, with Ry(90°) then Rz(180°). That mount is the `tag:` section of `config/robot.yaml` (`mount_xyz_m`, `mount_ry_deg`, `mount_rz_deg`).

Each sample is:

```text
base_T_camera = base_T_gripper @ gripper_T_tag @ inv(camera_T_tag)
```

Press **Enter** at four or more settled poses, then **C**. The result is the average of those samples (mean translation, short-arc SLERP for rotation) and is written to `config/hand_eye.yaml`. **T** turns torque off so you can pose the arm; support it first. **Q** quits without saving.

```sh
uv run python calibration/hand_eye.py
```

Or drive the arm through the fixed poses in `calibration/poses.md` (recorded side plus the XZ-plane mirror, roll fixed at −170°):

```sh
uv run python calibration/auto_hand_eye.py --live
uv run python calibration/auto_hand_eye.py --mock
```

## 8. Click and go

A click is a ray through that pixel. It hits the base-link plane **z = 0**. The arm starts at the rest pose, then moves the gripper frame (tool offset 0) to that point plus `HOVER_M` (0.02 m) along +Z. Depth is not used.

Rerun shows the arm, joint angles, end-effector pose, and the camera image together. `--mock` moves the drawn arm. The click itself is on the printed browser page: this Rerun version does not report image clicks back to the script.

```sh
uv run python application/click_and_go.py --mock
uv run python application/click_and_go.py --live
```

`--mock` still needs the camera, so the image and the ray are real. Only the arm is simulated.

## Click and pick

Same click as click-and-go. The arm starts at the rest pose with the gripper open. The tool point is 25 mm along gripper +Z. It picks `PICK_HEIGHT_M` above the click (default 0.00 m), closes, folds to the rest pose, and opens. Edit that constant at the top of `application/click_and_pick.py`.

```sh
uv run python application/click_and_pick.py --mock
uv run python application/click_and_pick.py --live
```

## 9. Pick and place

Edit `PICK` and `PLACE` in `application/pick_and_place.py` before you run it live. They are metres in the base frame.

```sh
uv run python application/pick_and_place.py --mock
uv run python application/pick_and_place.py --live
```

`--mock` opens Rerun and plays the pick and place on the drawn arm.

## Config

The default camera is the webcam in `config/camera.yaml` (`backend: webcamera`). `device` is the camera index: `/dev/videoN` on Linux, DirectShow on Windows, and AVFoundation on macOS. Set `backend: recamera` for the reCamera supervisor websocket (`url`, `ws://<host>:8090/ws/boxes`), or `backend: realsense` for a RealSense, and set `width`, `height`, and `fps` to a mode that camera supports. The webcam `fx` and `fy` are placeholders until you run `calibration/intrinsics.py`, and `config/hand_eye.yaml` is still the RealSense pose. Measure the intrinsics, then rerun `calibration/hand_eye.py`, before any live click.

| File | What you edit |
| --- | --- |
| `config/robot.yaml` | port, arm id, URDF, joint limits, joint offsets, tag family/id/size and mount |
| `config/camera.yaml` | backend (`webcamera`, `recamera`, or `realsense`), device, url, resolution, fps, intrinsics, distortion |
| `config/hand_eye.yaml` | `base_T_camera`, written by the calibration script |

## Using the arm from Python

```python
from so101_control import SO101

with SO101.connect(mock=True) as arm:
    arm.home()
    arm.rest()
    arm.set_speed(60)
    arm.move_to(0.20, 0.00, 0.10)
    print(arm.get_pose())
    arm.grip()
    arm.release()
```

`SO101.connect(mock=False)` opens the follower on the port in `config/robot.yaml`.
