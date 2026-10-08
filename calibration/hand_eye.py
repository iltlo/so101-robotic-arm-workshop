"""Live RGB eye-to-hand: T toggles torque, Enter captures, C validates/saves, Q quits."""

import argparse
import time

from so101_control.calibration import (Sample, camera_candidates, measured_gripper_T_tag,
                                       solve_eye_to_hand_known_tag)
from so101_control.config import ALL_JOINTS, load_station
from so101_control.hand_eye_view import HandEyeView
from so101_control.kinematics import Kinematics
from so101_control.motion import joint_state
from so101_control.session import rgb_camera, station_robot
from so101_control.station_calibration import intrinsics_data, save_calibration
from so101_control.torque import ArmTorque, keyboard_requests


def run_capture(robot, cam, kin, config, config_path, *, view, keys):
    torque = ArmTorque(robot)
    samples = []
    known_mount = measured_gripper_T_tag()
    estimate = None
    saved = False
    print("Tag: gripper_link -> [-37.2, 0, -28.4] mm; +90 deg Y, then 180 deg about tag-local Z (provisional).")
    print("T: torque OFF/ON (support arm); Enter: capture locked pose; C: validate/save >=6 poses; Q: quit.")
    print("Preview camera axes from one pose are NOT validated calibration.")
    while True:
        # Keyboard callbacks enqueue events; all robot/camera IO stays here.
        frame = cam.read()
        state = joint_state(robot)
        base_T_gripper = kin.fk_gripper_link(state)
        view.update(frame, state, base_T_gripper, kin.fk(state), estimate)
        command = keys.consume()
        if command == "q":
            if saved:
                print("Exiting; validated calibration saved.")
            else:
                print(f"Exiting; {len(samples)} unsaved samples discarded. "
                      "Press C after capture to validate and save; Q never saves.")
            return
        if command == "t":
            torque.toggle()
            continue
        if command == "c":
            if not torque.enabled:
                print("Turn torque ON before confirming calibration.")
                continue
            try:
                solved = solve_eye_to_hand_known_tag(samples, known_mount)
            except ValueError as exc:
                print("Cannot confirm calibration:", exc)
                continue
            print(f"Validated camera in base (metres):\n{solved['base_T_camera']}")
            print(f"Max residual: {solved['max_residual_m']:.3f} m / {solved['max_residual_deg']:.1f} deg")
            if solved["max_residual_m"] > 0.015:
                print("WARNING: Calibration may be saved for inspection, but residual exceeds the "
                      "15 mm click-to-hover gate. Do not use it for physical hover; inspect "
                      "tag size/mount and repeat calibration with more diverse poses.")
            save_calibration(config_path, dict(
                schema=1, accepted=True, arm_id=config.get("arm_id"), urdf=config["urdf"],
                camera_serial=config.get("camera_serial"), intrinsics=intrinsics_data(cam.intrinsics),
                base_T_camera=solved["base_T_camera"].tolist(),
                gripper_T_tag=solved["gripper_T_tag"].tolist(),
                max_residual_m=solved["max_residual_m"], max_residual_deg=solved["max_residual_deg"],
            ))
            estimate = solved["base_T_camera"]
            saved = True
            view.show_camera_estimate(estimate, sample_count=len(samples), confirmed=True)
            print("Validated calibration saved. Q exits; do not move the camera/tag mount.")
            continue
        if command != "enter":
            continue
        if not torque.enabled:
            print("Refusing capture with torque OFF. Support the arm; press T to hold it first.")
            continue
        time.sleep(0.25)
        after = joint_state(robot)
        if max(abs(state[name] - after[name]) for name in ALL_JOINTS) > 0.3:
            print("Arm moved during capture; retry after it settles.")
            continue
        captured_frame = cam.read()
        captured_state = joint_state(robot)
        if max(abs(after[name] - captured_state[name]) for name in ALL_JOINTS) > 0.3:
            print("Arm moved during RGB capture; retry.")
            continue
        captured_tag = view.update(captured_frame, captured_state,
                                   kin.fk_gripper_link(captured_state), kin.fk(captured_state), estimate)
        if captured_tag is None:
            print("Tag lost during RGB capture; retry.")
            continue
        samples.append(Sample(kin.fk_gripper_link(captured_state), captured_tag))
        # Single-pose hypotheses are useful for visual feedback, not accepted
        # camera calibration. Only C runs the diverse-pose residual checks.
        estimate = camera_candidates(samples, known_mount)[-1]
        saved = False
        view.show_camera_estimate(estimate, sample_count=len(samples))
        print(f"Captured {len(samples)} poses; provisional camera estimate shown in Rerun. "
              "Press C to validate/save once >=6 diverse poses are collected; Q discards unsaved samples.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    parser.add_argument("--enable-live", action="store_true", help="read the physical follower")
    args = parser.parse_args(argv)
    if not args.enable_live:
        parser.error("hand-eye requires --enable-live")
    config = load_station(args.config)
    if not config.get("fk_tcp_verified"):
        raise ValueError("Validate the gripper_link FK against physical measurements first (fk_tcp_verified)")
    kin = Kinematics(config["urdf"])
    with station_robot(config, True) as robot, rgb_camera(config) as cam:
        view = HandEyeView(config["urdf"], cam.intrinsics, measured_gripper_T_tag())
        try:
            with keyboard_requests() as keys:
                run_capture(robot, cam, kin, config, args.config, view=view, keys=keys)
        except KeyboardInterrupt:
            print("Interrupted; disconnecting follower. Support arm if torque is OFF.")


if __name__ == "__main__":
    main()