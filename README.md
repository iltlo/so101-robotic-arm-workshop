# SO-101 follower workshop

This repo provides runnable exercises under `setup/`, `teleop/`, `calibration/`, and `vision/`. Shared robot, FK/IK and Rerun code lives in `src/so101_control/`. The `reference/` folders are not required at runtime. **Run every command from the repository root.** No command moves the arm unless it includes `--enable-live` and sends a motion goal; status, FK and the twin only read feedback.

## 1. Install and check offline

Requires `uv` and Python 3.12–3.13. **Use `.uv-so101` for this project.** The older `.venv` is also Python 3.12 and contains a separate 6.9 GB dependency installation; it is intentionally left untouched because other sessions may use it. Without `UV_PROJECT_ENVIRONMENT`, `uv run` silently selects `.venv` instead. Repeat the `export` and `unset` in each new terminal; do **not** activate a venv. `unset PYTHONPATH` removes this host's unrelated ROS Python 3.10 packages from the 3.12 interpreter.

```sh
cd /home/ian/Code_Projects/so101-robotic-arm-workshop
export UV_PROJECT_ENVIRONMENT=.uv-so101
unset PYTHONPATH
uv sync --locked --all-extras
uv run --locked --all-extras python scripts/fetch_so101_assets.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --all-extras python -m pytest -q
```

The asset script downloads **one** official SO-101 follower model (`assets/SO101/so101_new_calib.urdf`), its 13 referenced STLs and the upstream license. It needs network at setup. If `config/station.yaml` does not yet exist, copy `config/station.example.yaml` to it and edit the copy; **do not overwrite an existing station configuration**. The `reference/` trees are never changed by asset setup.

Try the installed scripts without hardware:

```sh
uv run --locked --all-extras python teleop/status.py --config config/station.yaml
uv run --locked --all-extras python teleop/twin.py --config config/station.yaml --recording /tmp/so101_demo.rrd
uv run --locked --all-extras rerun /tmp/so101_demo.rrd /tmp/so101_demo.rbl
```

The twin recording contains **five synthetic poses**, including jaw open/close, and a matching viewer blueprint (`.rbl`). For a live offline viewer instead of recording to disk, omit `--recording`; no motors are connected. Without `--enable-live`, the twin script returns after the demo is logged, so use the saved recording to inspect it later. `rerun rrd verify /tmp/so101_demo.rrd /tmp/so101_demo.rbl` can check the files if playback fails.

## 2. Prepare and inspect the physical follower

Secure the arm, clear the workspace and keep a **physical power cutoff** accessible. Configure motor IDs and calibrate using the [official LeRobot SO-101 instructions](https://huggingface.co/docs/lerobot/so101). For this station, the `port` and `arm_id` in `config/station.yaml` must match those used by LeRobot. To identify the USB port without moving the arm, follow the connect/unplug prompts (then reconnect USB):

```sh
uv run --locked --all-extras python setup/find_port.py --config config/station.yaml
```

To run the workshop's full midpoint-and-range calibration with a **live Rerun twin**, use:

```sh
uv run --locked --all-extras python setup/calibrate_motors.py --config config/station.yaml --enable-live
```

**Support the arm**: torque is disabled, and this script never commands motor positions. Carefully position *all* joints, including the gripper, at their physical midpoints and press **Enter**; this captures physical encoder positions without writing them. Press **Enter** to begin recording. Slowly sweep every joint except `wrist_roll` through its full **safe** range; press **Enter** to finish. In Rerun inspect the proposed joint pose and the printed offsets/min/max. The tool rejects missing travel or a badly centered midpoint. Press **S then Y** to confirm saving to the motor registers, LeRobot JSON and station metadata, or **Q** to discard. All keys use the X11 global listener; do not type terminal commands or rely on the keyboard as an emergency stop. Recalibration backs up old motor and camera files, invalidates previously verified FK and hand-eye, and requires physically rechecking FK and camera pose before any live motion.

Alternatively use the official LeRobot command with your actual port and arm ID (it can disable torque while prompting you to move the arm):

```sh
uv run --locked --all-extras lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=workshop_follower
```

LeRobot stores actual motor offsets and ranges in its own per-arm JSON file (typically `~/.cache/huggingface/lerobot/calibration/robots/so_follower/workshop_follower.json`), **not** in `station.yaml`. The workshop tool also records a snapshot of offsets in `station.yaml`, but the LeRobot file and motor registers remain authoritative. A calibration recording with a joint's `POS` close to `MIN` or `MAX` may indicate it was not moved through its complete physical range; repeat full calibration before trusting FK.

To fine-tune a *small* motor zero offset while viewing the arm in Rerun, after completing official calibration:

```sh
uv run --locked --all-extras python setup/tune_motor_zero.py --config config/station.yaml --enable-live
```

**Support the arm**: this tool disables torque and never sends motion goals. On an X11 desktop, press **1–5** to select an arm joint in `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` order; **6** selects the gripper and **0** deselects. **Up/Down** previews ±0.5° per keypress (up to ±30° absolute station correction), or ±1% for the gripper (up to ±10%). Rerun displays the proposed pose from LeRobot normalized feedback, including readings outside recorded motor travel; that is **not proof the physical range is valid**. **Q** discards previews. **S then Y** saves **only `joint_zero_offsets_deg` and `gripper_zero_offset_percent` in `station.yaml`**; it never changes LeRobot's JSON or motor registers. The script replaces `station.yaml.before-zero-tuning.bak` with the immediately preceding station settings and, if a camera calibration exists, replaces `calibration.yaml.before-zero-tuning.bak` with its previous contents; older backups at those names are overwritten. It clears `fk_tcp_verified` and `base_plane_verified` and invalidates any accepted `config/calibration.yaml`. Live FK/hand-eye feedback and live joint commands apply the saved correction consistently. Verify physical alignment, redo hand-eye, and recheck any workspace/TCP limits before live use. This is not a substitute for recording full joint ranges: if the gripper or any joint's ranges are wrong, redo full motor calibration first. A key listener requires X11; closing Rerun is not a safety stop. Keep a physical power cutoff available.

Then check calibrated readings and the gripper frame (`base_T_gripper_frame_link`, translation in metres):

```sh
uv run --locked --all-extras python teleop/status.py --config config/station.yaml --enable-live
uv run --locked --all-extras python teleop/fk.py --config config/station.yaml --enable-live
```

LeRobot calibration's **raw motor ticks are not joint degrees**. Before any jog, set all five `joint_limits_deg` in `config/station.yaml` to conservative, independently verified **degree** ranges inside the arm's safe working area. The gripper is normalized `0..100`, not degrees. Then run `uv run --locked --all-extras python teleop/joint_jog.py --config config/station.yaml --enable-live`; enter `shoulder_pan 1` (max 2° per step) or `q` to quit. Do not use hard mechanical stops as software limits.

## 3. View the articulated live twin

```sh
uv run --locked --all-extras python teleop/twin.py --config config/station.yaml --enable-live
```

The twin uses **that same URDF** for both Rerun and FK; live feedback updates five arm joints and the jaw. The viewer has one 3-D arm view and **two plots**: joint degrees and EE x/y/z (metres) plus roll/pitch/yaw (degrees). The **FK EE coordinate frame** has red/green/blue X/Y/Z axes: their common origin is the gripper-frame position, and their directions show its orientation in the robot base frame. Jaw movement is **display-only**: LeRobot reports `gripper.pos` from `0` (closed) to `100` (open), but the official URDF supplies no servo-to-jaw angle mapping, so the twin linearly interpolates its joint limits as a visual approximation. Do not use displayed jaw angle to infer opening width or grasp contact. Check a few physical poses against the viewer before trusting FK. Rerun is neither a physics simulator nor a collision checker.

By default the twin **cannot toggle torque**. For an immediate **T** key toggle on a supported desktop, deliberately add `--allow-torque-toggle` to the **live** twin command. It does **not** prompt or pause the feedback loop; after torque is off, the viewer continues reading servo positions as you reposition the arm. **Disabling torque may let the arm fall.** Support it physically first and keep the physical power cutoff accessible. `T` and Ctrl-C are *not* emergency stops; on Wayland a global key listener may not work. Leaving the twin disconnects the follower; it does not home the arm.

## 4. Fixed RGB camera → robot calibration

First check the AprilTag **in the camera optical frame alone** (no arm connection or robot motion):

```sh
uv run --locked --all-extras python calibration/visualize_apriltag.py --config config/station.yaml --tag-id 0 --tag-size-m 0.04
```

Rerun shows live RGB, the camera origin, the detected tag's X/Y/Z axes, and the 4×4 `camera_T_tag` matrix in **metres**. The camera optical frame has X right, Y down, Z forward; `camera_T_tag` maps coordinates in the tag frame into the camera frame. Its translation and rotation should change as the tag moves; if the tag is lost, its pose is cleared rather than held. Measure the **black-square tag edge** before setting `--tag-size-m` (0.04 is only the workshop assumption). Compare the live pose with the tag's physical position and orientation before combining it with arm FK. Stop with Ctrl-C. This script never computes `base_T_camera` and does not validate the tag's `gripper_T_tag` mount.

The camera is a fixed RealSense used for **RGB only**; no depth is read. After physically validating the URDF's **`gripper_link` FK** against the arm, set `fk_tcp_verified: true` in `config/station.yaml` (this flag is **not** auto-validation or a camera calibration). Secure a **40 mm tag36h11 AprilTag ID 0** to the EE and keep the camera fixed. The provisional tag pose relative to `gripper_link` is **[-37.2, 0, -28.4] mm, with +90° about gripper Y followed by 180° about the tag's local Z axis** (`R_y(90°) @ R_z(180°)`); check the tag axes and mount physically before accepting the solution.

```sh
uv run --locked --all-extras python calibration/hand_eye.py --config config/station.yaml --enable-live
```

Rerun continuously shows **RealSense RGB** with a **green outline and center** around the detected ID 0 tag. Red X, green Y and blue Z arrows mark the FK EE and measured tag-mount pose. No terminal commands or typed prompts are used: on an X11 desktop, press keys directly (they are global, including when Rerun has focus):

1. Press **T** to turn torque **OFF**, while physically supporting the arm: gravity may make it fall. Reposition gently; avoid mechanical stops. Press **T** again to hold the *measured current position* before torque turns **ON**. Let the arm settle. If it moves unexpectedly, use the physical power cutoff.
2. Press **Enter** once to register the current visible tag36h11 ID 0 and stable joint feedback. The 3-D view immediately shows the **provisional camera pose** and retains an individual camera-axis hypothesis for that sample. Repeat steps 1–2 for **at least six varied positions with independent rotations**. Captures with torque off, missing tag or moving joints are rejected. Individual camera estimates are previews, **not accepted calibration**.
3. **After the sixth capture, press C** (not Q) to validate the aggregate camera pose and save `config/calibration.yaml` only if pose diversity and residual checks pass (max 20 mm translation / 8° rotation). Camera rotations are averaged by progressive shortest-arc SLERP; translations by arithmetic mean. Inspect the printed residuals and physically cross-check camera/tag frame directions before trusting the result. **Q exits without saving unsaved samples.** A new capture after saving needs C again to validate the expanded set (the previously saved file is not removed).

A 15–20 mm residual may be saved **for inspection only**; `vision/click_to_hover.py` still refuses any saved calibration over **15 mm** translation residual, including preview. Neither passing threshold proves physical accuracy. If C reports an inconsistency, try more diverse settled poses and verify the measured tag size, mount and FK before accepting the result.

The keyboard shortcut is **not** a physical emergency stop. Global key hooks require a supported desktop (typically X11); on Wayland they may be unavailable, so run under X11 rather than relying on typed terminal commands. The mount measurements are provisional: if your ruler measurement or AprilTag printed axis convention differs, correct the transform in `src/so101_control/calibration.py` before collecting samples. Camera movement, remounting the tag or motor recalibration requires collecting new samples. Camera and joint feedback remain live throughout.

Optional, if the physical surface is **not** the base_link XY plane at Z=0: move that **same** ID 0 tag flat onto the tabletop and hide/remove the EE tag so there is only one visible ID 0:

```sh
uv run --locked --all-extras python calibration/table_plane.py --config config/station.yaml
```

Type `YES` at the prompts only after checking the tag pose, plane height and orientation against physical points. The current click-to-hover script uses **base_link Z=0**, not this optional tag-measured table plane; do not use it for live moves on a surface at another height.

## 5. RGB click preview → supervised hover

For **read-only preview**, an accepted hand-eye camera calibration is required, but `gripper_T_tcp`, `workspace_m`, `hover_clearance_m` and table-tag calibration may remain unset. Clicks are intersected with the **assumed base_link XY plane at Z=0**, which is *not* automatically the physical tabletop. The preview prints only this base-plane point, **not** a hover TCP target.

```sh
uv run --locked --all-extras python vision/click_to_hover.py --config config/station.yaml
```

For a **virtual arm and read-only IK preview** with no follower connection or motor commands, run:

```sh
UV_PROJECT_ENVIRONMENT=.uv-so101 uv run --locked --all-extras python vision/click_to_hover.py --config config/station.yaml --simulate
```

Rerun begins at an offline zero-joint pose and animates each successfully solved TCP-position target (not a checked motion trajectory); failed IK leaves the previous simulated pose displayed. In this station config the TCP is currently the URDF `gripper_frame_link` origin near the fixed fingertip, **not** `gripper_link` (which is near the wrist). The URDF does not have an exact named fingertip contact frame: compare the simulated fingertip against the target and physically measure any remaining offset before setting `fk_tcp_verified: true` for live motion. Simulation requires configured TCP, clearance and joint limits, but does not require `workspace_m` and cannot validate physical reachability or collision clearance. `--simulate` and `--enable-live` cannot be combined. Open the printed **localhost URL** in a browser and click an empty base-plane pixel to print its intersection in base coordinates; **Ctrl-C** stops the RGB server. No OpenCV GUI is required. This **does not connect to the follower**. Hand-eye produces `base_T_camera`, *not* the physical tool-center offset: the tag on the gripper is not automatically the picking TCP. For live hover, measure and fill **all three** fields in `config/station.yaml`: `gripper_T_tcp` (4×4 transform in metres from URDF `gripper_frame_link` to actual TCP), `workspace_m` (base-frame `{x: [min,max], y: [min,max], z: [min,max]}`) and `hover_clearance_m` (validated 0.015–0.15 m). Never use Dobot millimetre coordinates. Only after independently checking that the real surface is at base_link Z=0 and verifying camera projection, FK/TCP, workspace and clearance against physical measurements should an operator set `base_plane_verified: true` and run the same command with `--enable-live`. If the physical tabletop is above/below base Z=0 or tilted, **do not enable live hover** with this assumption. With `--enable-live`, Rerun shows the continuously updated arm twin and a red "candidate TCP (preview only)" marker at the clicked hover goal, including when `workspace_m` is unset. This marker is only a visualization; an unset or unverified workspace still rejects the click **before IK, confirmation, or motor commands**. Use the marker to inspect the geometry, then measure and configure safe workspace bounds separately; it does not validate collision clearance. Live hover uses the same bounded position-only TCP solver as simulation, starting from measured arm joint feedback; a post-solve TCP error check still rejects targets more than 15 mm away. Simulation starts from an offline zero pose instead, so its success alone does not establish live reachability or a safe trajectory. Each live hover requires an IK check and a separate terminal `YES` confirmation. A click **never descends, closes the gripper or picks**. No collision detection or full path clearance check is provided: keep the area clear and have physical power disconnect ready.

The SO-101 has five arm-position joints and cannot generally achieve an arbitrary 6-DOF tool pose. Object detection/YOLO, leader teleop and autonomous grasp/release are outside this release. Upstream licenses and attribution: `THIRD_PARTY_NOTICES.md`.