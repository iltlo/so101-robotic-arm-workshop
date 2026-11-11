# SO-101 workshop

Local control for an SO-101 follower: a browser UI for connection, calibration, teleop, and picking, plus a small Python API. Distances are metres. Joint angles are degrees.

## Run

Install [uv](https://docs.astral.sh/uv/), then start one local process. It prints `http://127.0.0.1:8020` and listens only on this machine. An installed run keeps its config and a copy of the URDF meshes in `~/.so101-workshop/`; the package ships those files, so no download is needed. Override that directory with `--config-dir` or `SO101_WORKSHOP_HOME`.

macOS and Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --python 3.13 so101-workshop@latest run
```

Windows PowerShell. Use native Windows so the reCamera USB gadget at `192.168.42.1` is visible:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uvx --python 3.13 so101-workshop@latest run
```

From a checkout of this repository:

```sh
uvx --python 3.13 --from . so101-workshop run
```

`--mock` runs the page without an arm. `--no-viewer` skips the Rerun panel.

The page is split into **Hardware** (arm, camera, USB port, motor calibration), **Hand-eye** and **Intrinsics**, and **Tasks** (teleop, click and go, click and pick, detect and pick). Motor calibration stays disabled until you press **Start motor calibration**. The E-stop turns torque off.

- Linux: add your user to the `dialout` group. The arm is usually `/dev/ttyACM0`.
- Windows: serial ports are `COM*`.
- macOS: the arm is `/dev/cu.usbmodem*`.

Python 3.13. Without `--python 3.13`, `uvx` uses whatever `python` is on your PATH. Python 3.14 can open the page, but AprilTags has no wheel for it, so hand-eye does not run. On macOS, the AprilTags wheel is Apple silicon only.

The arm panel uses Rerun 0.26.2. Later Rerun releases have no Intel Mac wheel, so this is the version that installs on Intel macOS, Apple silicon, Linux, and Windows.

## Safety

Clear the table before any live motion. Joint speed defaults to 90 deg/s, which is the maximum. Change it with `arm.set_speed(...)`. Torque off lets the arm fall: support it, and keep a power cutoff within reach. Ctrl-C and the keyboard are not emergency stops. There is no collision checker.

## First use

Plug in the USB adapter. Motors can stay off. Under **Arm connection**, find the port, then calibrate the motors once before any live move. Torque turns off, so support the arm. Pose every joint at the middle of its travel and confirm. Then move every joint except wrist roll through its whole travel, including the gripper, and confirm again. Wrist roll keeps a full turn. That writes `motors.json`. Later connects load it and update a servo whose midpoint or range differs. 0° is the calibrated midpoint.

Connect the arm and the camera. The default camera is the reCamera at `ws://192.168.42.1:8090/ws/boxes`.

## Calibration

**Intrinsics.** The webcam and the reCamera do not report fx, fy, or distortion. The board has 8 × 11 inner corners (9 × 12 squares). Hold it in view, move it between captures (tilt it, and use different parts of the image), and take at least ten views. **Capture chessboard** is green only while the board is in view.

**Hand-eye.** The camera stays fixed. The tag is tag36h11, id 0, 40 mm, mounted on the gripper. That mount is the `tag:` section of `robot.yaml`. With the arm and camera connected, torque on, and the tag in view, **Run auto calibration** moves through the poses, keeps the samples that see the tag, and saves `hand_eye.yaml`. **Stop** and the E-stop abort the tour. Clear the table first, because the arm moves by itself. Redo this if you move the camera. The hand-eye view shows the tag's XYZ in millimetres and RPY in degrees while the tag is detected.

## Tasks

**Teleop** uses the keyboard or the on-screen keypad. WASD or the arrows move x and y. Q/E move z and keep the current orientation. Z/C turn the gripper about its fingers, F/V tilt the fingers up and down about the arm's wrist axis, R rests, H homes, and space toggles the gripper. Keys follow the Teleop tab. **Key teleop** in the top bar keeps them on across tabs.

**Click and go** treats a click as a ray onto the base plane z = 0, then moves the gripper there, 20 mm above the plane.

**Click and pick** starts from rest with the gripper open, picks at the click, closes, returns to rest, and opens.

**Detect and pick** uses the on-device YOLO boxes. Click inside a box and the arm picks that box's centre. Overlapping boxes: the smallest one wins. A click outside every box does not move the arm.

Depth picking is not included.

## Config

Files live in the station directory (`config/` in a checkout, `~/.so101-workshop/config/` when installed).

| File | What you edit |
| --- | --- |
| `robot.yaml` | port, arm id, joint limits, joint offsets, tag family/id/size and mount |
| `camera*.yaml` | backend, device, url, resolution, fps, intrinsics |
| `workshop.yaml` | feature flags, active camera profile, detection websocket and MQTT |
| `hand_eye.yaml` | camera pose in the base frame, written by calibration |
| `motors.json` | servo midpoint and range, written by motor calibration |

Switch camera under **Hardware → Camera connection**. Intrinsics belong to that profile, so calibrate the camera you will use. For a webcam, `device` is the camera index: `/dev/videoN` on Linux, DirectShow on Windows, and AVFoundation on macOS.

| Profile | Backend | Notes |
| --- | --- | --- |
| `camera_recamera.yaml` | `recamera` | Default. `ws://192.168.42.1:8090/ws/boxes`, 1280×720 |
| `camera_logi.yaml` | `webcamera` | USB webcam, index 0 |
| `camera_usbcam.yaml` | `webcamera` | Second USB webcam. Set `device` if that index is not this machine's camera |

RealSense needs the vision extra: `uvx --python 3.13 --with pyrealsense2 so101-workshop run`.

`joint_limits_deg` and `joint_offsets_deg` in `robot.yaml` are this arm's limits and small zero corrections. Edit the offsets only if you re-measure them.

## Scripts

From a checkout, the same work is available as scripts. `--mock` draws the arm in Rerun and does not open the serial port. `--live` uses the arm. Clear the table first.

```sh
uv sync --python 3.13 --locked --all-extras
```

The pinned URDF and meshes are already in `assets/SO101/`, and the package ships them, so there is nothing to fetch. `scripts/fetch_so101_assets.py` only refreshes them when the pinned upstream commit changes.

```sh
uv run python scripts/check_port.py
uv run python scripts/calibrate_motors.py
uv run python scripts/twin.py --mock
uv run python scripts/twin.py --live
uv run python scripts/get_pose.py --live
uv run python scripts/arm_move.py --live
uv run python scripts/gripper.py --live
uv run python scripts/joint_jog.py --mock
uv run python scripts/keyboard_control.py --live
uv run python scripts/camera_view.py
uv run python calibration/apriltag.py
uv run python calibration/intrinsics.py
uv run python calibration/hand_eye.py
uv run python calibration/auto_hand_eye.py --live
uv run python application/click_and_go.py --live
uv run python application/click_and_pick.py --live
uv run python application/detect_and_pick.py --live
uv run python application/pick_and_place.py --live
```

`check_port.py` asks you to unplug the adapter and writes `port` in `config/robot.yaml`. `twin.py --live` only reads joint feedback; `--relax` lets `T` turn torque off while you pose the arm. `arm_move.py` homes (every arm joint to 0°) and visits three points; change `POINTS` in that file for different ones. Gripper percent is 0 closed and 100 fully open. `joint_jog.py` steps one joint by at most 2°: type `shoulder_pan 1`, then `q`.

Keyboard teleop adds P to print the pose, +/- to change the step, and Esc as E-stop. Translation keys keep the current orientation.

`camera_view.py` shows the live picture in Rerun. Q quits. `calibration/intrinsics.py` searches every frame for the chessboard, so that view updates more slowly. **Space** captures the current view, **Enter** solves and writes `config/camera.yaml`, **Q** quits.

`calibration/apriltag.py` is the camera only. Ctrl-C stops. `calibration/hand_eye.py` records settled poses: **Enter** samples, **C** solves and writes `config/hand_eye.yaml` after four or more samples, **T** turns torque off (support the arm first), **Q** quits without saving. `calibration/auto_hand_eye.py` drives the poses in `calibration/poses.md`.

`click_and_go.py --mock` still needs the camera, so the image and the ray are real. Only the arm is simulated. The click is on the printed browser page. Edit `PICK` and `PLACE` in `application/pick_and_place.py` before a live run. They are metres in the base frame.

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

`SO101.connect(mock=False)` opens the follower on the port in `robot.yaml`.
