# SO-101 workshop

A short path from a fresh checkout to moving the follower. Distances are **metres**. Joint angles are **degrees**. The arm class is `SO101`. Camera, transforms, and the Rerun viewer live in `so101_utils` and stay out of the scripts.

## Run the local UI

Install [uv](https://docs.astral.sh/uv/), then start one local process and open the URL it prints (`http://127.0.0.1:8020`). The server binds to loopback. Inside this repository it uses `config/` and `assets/`. A published `uvx` run uses `~/.so101-workshop/` (override with `--config-dir` or `SO101_WORKSHOP_HOME`) and downloads the URDF meshes on first kinematics load.

macOS and Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx so101-workshop@latest run
```

Windows PowerShell. Use native Windows so the USB gadget at `192.168.42.1` is visible:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uvx so101-workshop@latest run
```

Until the package is on PyPI, from this checkout:

```sh
uvx --from . so101-workshop run
```

`uvx` caches the wheel it builds for a local path and only rebuilds when
`pyproject.toml` changes. After editing the source, run `touch pyproject.toml`
first, or use `uv run python -m so101_workshop.cli run`, which always runs the
checkout.

The page groups the work into **Hardware** (arm, camera, USB discovery, motor calibration), **Hand-eye** and **Intrinsics** (tag samples, the automatic tour, and chessboard calibration), and **Tasks** (teleop, click and go, click and pick, detect and pick). It teleops from the keyboard or a keypad arranged like the keyboard, fills the card width, and lights the key that is pressed. The USB finder and motor calibration sit under **Arm connection** as optional sections; motor calibration stays disabled until you press **Start motor calibration**, so one stray click cannot rewrite the motor values. Teleop keys turn on with the Teleop tab and off when you leave it; the **Key teleop** button in the top bar shows on/off and keeps them on across tabs. The YOLO box table is on **Detect and pick**. The hand-eye view prints the tag's XYZ in millimetres and RPY in degrees while the tag is detected, and **Capture chessboard** turns green only while the board is in view. The control cards and the Rerun panel share the width in two equal columns; only the control column scrolls, and the Rerun panel stays fixed while you work. Rerun's joint and pose plots follow the newest sample at the right edge. Below 1200 px or in a portrait (taller than wide) window the columns stack and the page scrolls normally. The E-stop is a local request on this machine and turns torque off; it also clears the teleop target and motion cache, so torque-on starts from the measured pose. Detection messages cannot move the arm. Use `--mock` to try the whole page without hardware and `--no-viewer` to skip the Rerun panel.

- Linux: add your user to the `dialout` group. The arm is typically `/dev/ttyACM0`. The automated tests run on Linux.
- Windows: serial ports are `COM*`.
- macOS: the arm is `/dev/cu.usbmodem*`.

Wheels checked on PyPI for Python 3.12:

| Package | Windows x64 | macOS | Linux |
| --- | --- | --- | --- |
| opencv-python-headless 4.13.0.92 | `win_amd64` | arm64 and x86_64 | manylinux x86_64 and aarch64 |
| pupil-apriltags 1.0.4.post11 | `win_amd64` | arm64 | manylinux x86_64 |
| feetech-servo-sdk 1.0.0 | pure-Python sdist | pure-Python sdist | pure-Python sdist |
| pybullet 3.2.7 | no cp312 wheel | no cp312 wheel | manylinux wheels stop at cp311 |

`pybullet` 3.2.7 has no Python 3.12 wheel, so uv builds the sdist. That needs a C++ compiler. If the build fails, stop there. This project does not fork Bullet.

The default install talks to the STS3215 servos with the Feetech SDK. Motor calibration writes the midpoint homing offset and the min/max range, then saves `config/motors.json` in the station directory.

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
- [x] Detect and pick: click a YOLO box, pick its centre (`application/detect_and_pick.py`)
- [x] Pick and place (`application/pick_and_place.py`)
- [x] Local workshop UI (`so101-workshop run`)

Depth picking is not in this workshop. The local UI lists on-device YOLO boxes and can write motor midpoints and ranges.

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

Do this once before any live motion. Torque turns off, so support the arm. Pose every joint at the middle of its travel and press Enter. Then move every joint except wrist roll through its whole travel, including the gripper, and press Enter again. Wrist roll keeps a full turn. The same two steps are on the workshop Calibration page.

```sh
uv run python scripts/calibrate_motors.py
```

That writes `config/motors.json`. Every later connect loads that file and writes it to the servos when their homing offset or limits differ. `joint_limits_deg` and `joint_offsets_deg` in `config/robot.yaml` are the limits and the small zero corrections for this arm. Edit the offsets only if you re-measure them. 0° is the calibrated midpoint.

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

Keyboard teleop (WASD / arrows move x and y, Q/E move z, and those translation keys keep the current orientation. Z/C change base roll, F/V change base pitch, R rests, H homes, space toggles the gripper, P prints the pose, +/- change the step, and Esc is E-stop). The web page keypad also has Home, Rest, Gripper, Pose, and step buttons. Rerun shows the arm, joint angles, and end-effector pose. `--mock` moves that drawing and does not open the serial port.

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

The same tour runs from the **Hand-eye** page. With the arm and camera connected, torque on, and the tag in view, **Run auto calibration** visits all ten poses, records the samples that see the tag, solves, and saves `config/hand_eye.yaml` with its residual. The button returns at once and the progress bar follows the arm. **Stop** and the E-stop abort between poses and during the settle wait; teleop, manual samples, reset, solve, and disconnect stay locked until the tour ends. Clear the table first, because the arm moves by itself.

## 8. Click and go

A click is a ray through that pixel. It hits the base-link plane **z = 0**. The arm starts at the rest pose, then moves the gripper frame (tool offset 0) to that point plus `HOVER_M` (0.02 m) along +Z. Depth is not used.

Rerun shows the arm, joint angles, end-effector pose, and the camera image together, with the 3D arm next to the camera image. `--mock` moves the drawn arm. The click itself is on the printed browser page: this Rerun version does not report image clicks back to the script.

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

## Detect and pick

The reCamera model sends YOLO boxes beside the JPEG. Each box is `[centre_x, centre_y, width, height]` in the frame named by the payload's `resolution`, which the agent converts to corners before drawing or hit-testing. A click selects the smallest box under it, and the arm picks that box's **centre**, not the click point. The approach and release match click-and-pick: it hovers `APPROACH_M` above the centre, closes, folds to the rest pose, and opens. The web page draws the boxes over the live image and shows the selected box and centre in the result line. Boxes only reach the page: they are never drawn into the MJPEG stream, and they cannot move the arm until you click.

```sh
uv run python application/detect_and_pick.py --mock
uv run python application/detect_and_pick.py --live
```

In the UI, open **Tasks → Detect and pick**, connect the arm and camera, then click anywhere inside a green box. With overlapping boxes, the smallest one wins. A click outside every box does not move the arm.

## 9. Pick and place

Edit `PICK` and `PLACE` in `application/pick_and_place.py` before you run it live. They are metres in the base frame.

```sh
uv run python application/pick_and_place.py --mock
uv run python application/pick_and_place.py --live
```

`--mock` opens Rerun and plays the pick and place on the drawn arm.

## Config

`config/camera*.yaml` holds one profile per camera. The active profile is saved in `config/workshop.yaml` as `camera_profile` and can be switched in **Hardware → Camera connection**; the agent opens only that camera. The browser receives MJPEG from the agent, so the agent stays the only camera client. reCamera boxes arrive on the same WebSocket as the JPEG and are listed in the detection panel, never drawn into the video.

| Profile | Backend | Notes |
| --- | --- | --- |
| `camera.yaml` | `recamera` | Default: `ws://192.168.42.1:8090/ws/boxes`, 1280×720 |
| `camera_recamera720.yaml` | `recamera` | Same WebSocket, 720p |
| `camera_demo.yaml` | `demo` | Synthetic tabletop; no hardware |
| `camera_logi.yaml` | `webcamera` | USB webcam index 0 |
| `camera_usbcam.yaml` | `webcamera` | Second USB webcam |
| `camera_realsense.yaml` | `realsense` | Needs the `vision` extra (`pyrealsense2`) |

For a webcam, `device` is the camera index: `/dev/videoN` on Linux, DirectShow on Windows, and AVFoundation on macOS. An `rtsp://` url keeps its host and uses the reCamera websocket. Set `width`, `height`, and `fps` to a mode the camera supports. Intrinsics live in each profile, so run the chessboard calibration with the camera you will use. `config/hand_eye.yaml` is camera-specific too; redo hand-eye whenever you move the camera.

The Rerun panel streams the existing arm blueprint from a headless `rerun --serve-web` process (gRPC 9881, web 9091). It shows the arm and joint angles on the left and the camera image and end-effector pose on the right, plus AprilTag axes while you work. This is live-only: the viewer keeps a rolling buffer (256 MB viewer, 64 MB server) and drops the oldest frames, and the station stops sending as soon as the viewer's gRPC port closes. Nothing is written to disk. If 9881/9091 are held by another station, the next free pair is used.

| File | What you edit |
| --- | --- |
| `config/robot.yaml` | port, arm id, URDF, joint limits, joint offsets, tag family/id/size and mount |
| `config/camera*.yaml` | backend (`webcamera`, `recamera`, `realsense`, or `demo`), device, url, resolution, fps, intrinsics, distortion |
| `config/workshop.yaml` | feature flags, active camera profile, detection websocket/MQTT |
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
