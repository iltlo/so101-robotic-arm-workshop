# SO-101 workshop

A browser page to connect the arm, calibrate it, and run the tasks. Distances are metres. Joint angles are degrees.

Clear the table before the arm moves. Torque off lets it fall, so support it.

## Start the page

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uvx --python 3.13 so101-workshop@latest run
```

From this folder:

```sh
uvx --python 3.13 --from . so101-workshop run
```

Open the address it prints (`http://127.0.0.1:8020`). `--port 8033` changes the port. `--mock` runs the page with no arm. `--no-viewer` hides the 3D view.

Use Python 3.13. On Linux, add your user to the `dialout` group and install `libgl1`. The arm is `/dev/ttyACM0` on Linux, `COM*` on Windows, and `/dev/cu.usbmodem*` on macOS.

## First time

1. Plug in the USB adapter. Under **Hardware**, find the port.
2. Press **Start motor calibration**. Set every joint to the middle of its travel, then move every joint except wrist roll through its range, including the gripper.
3. Connect the arm and the camera. The default camera is the reCamera.

## Tasks

The **Tasks** tabs are the exercises.

| Tab | What it does |
| --- | --- |
| Teleop | Drive the gripper. WASD moves X and Y, Q/E moves Z, F/V tilts, Z/C twists, space toggles the gripper, H homes, R rests. |
| Click and go | Click the table. The gripper moves 20 mm above that point. |
| Click and pick | Click the table. The gripper opens, picks, returns to rest, and opens. |
| Detect and pick | Click a box. The arm picks the centre of that box. |

**Key teleop** in the top bar keeps the keys working on every tab. The motor control shows the current state and what the next press does: green **Press to torque off** (or **Esc**) while holding, red **Press to enable torque** when off.

Detect and pick uses the boxes from the reCamera. To run YOLO on this computer instead, install the extra and set `detector: local` in `workshop.yaml`:

```sh
uv sync --python 3.13 --extra yolo
```

A RealSense needs `uv sync --python 3.13 --extra vision`. `camera_realsense.yaml` is the D405. `camera_usbcam.yaml` is a USB camera (`device` is the video index). `camera_demo.yaml` is a fake camera for trying the page.

## Run the code

From this folder, once:

```sh
uv sync --python 3.13 --extra twin --extra yolo
```

`--mock` does not move a real arm. `--live` does.

```sh
uv run python scripts/keyboard_control.py --live
uv run python scripts/twin.py --live
uv run python scripts/get_pose.py --live
uv run python scripts/arm_move.py --live
uv run python scripts/gripper.py --live
uv run python application/click_and_go.py --live
uv run python application/click_and_pick.py --live
uv run python application/detect_and_pick.py --live
uv run python application/pick_and_place.py --live
uv run python calibration/hand_eye.py
uv run python calibration/auto_hand_eye.py --live
uv run python detection/inference.py
```

`keyboard_control.py` is the same keys as the page, plus P for the pose and Esc to stop. `twin.py` only shows the arm. `arm_move.py` homes, visits three points, and rests. Change `POINTS` in that file, and the heights and delays at the top of `pick_and_place.py` and `detect_and_pick.py`, before a live run.

```python
from so101_control import SO101

with SO101.connect(mock=True) as arm:
    arm.home()
    arm.move_to(0.20, 0.00, 0.10)
    arm.grip()
    arm.release()
    arm.rest()
```

`SO101.connect(mock=False)` uses the port in `config/robot.yaml`.
