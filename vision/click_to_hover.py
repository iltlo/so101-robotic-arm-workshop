"""Preview RGB pixel-to-base-plane points; live hover requires verified measurements."""

import argparse
from contextlib import nullcontext
import time

import numpy as np

from so101_control.camera import require_fresh
from so101_control.click_view import click_view
from so101_control.config import load_station, transform, validated_limits
from so101_control.geometry import base_link_plane, intersect_pixel, preview_hover_target, workspace_check
from so101_control.kinematics import Kinematics
from so101_control.motion import joint_state, move_joints
from so101_control.session import rgb_camera, station_robot
from so101_control.station_calibration import load_calibration, matches_camera
from so101_control.twin import Twin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/station.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enable-live", action="store_true", help="opt in to confirmed physical hover moves")
    mode.add_argument("--simulate", action="store_true", help="solve IK and animate a virtual arm; never connect to the follower")
    args = parser.parse_args(argv)

    config = load_station(args.config)
    data = load_calibration(args.config)
    plane = base_link_plane()
    if data.get("max_residual_m", float("inf")) > 0.015 or data.get("max_residual_deg", float("inf")) > 8:
        raise ValueError("Hand-eye residual exceeds acceptance threshold")
    # A read-only pixel-to-base-plane preview needs camera extrinsics, not TCP,
    # clearance, joint limits, or a measured robot workspace.
    if (args.enable_live or args.simulate) and config.get("gripper_T_tcp") is None:
        raise ValueError("Hand-eye determines base_T_camera, not the physical tool center. "
                         "Measure gripper_frame_link-to-TCP and set gripper_T_tcp in config/station.yaml "
                         "before live hover. Use this command without --enable-live to inspect RGB clicks safely.")
    tcp = transform(config["gripper_T_tcp"]) if args.enable_live or args.simulate else None
    if args.enable_live and not config.get("fk_tcp_verified"):
        raise ValueError("Set fk_tcp_verified: true only after independent physical FK/TCP validation")
    if args.enable_live and not config.get("base_plane_verified"):
        raise ValueError("Set base_plane_verified only after verifying physical surface is at base_link z=0 and checking camera projection")
    limits = validated_limits(config) if args.enable_live or args.simulate else None
    if (args.enable_live or args.simulate) and (config.get("hover_clearance_m") is None or
                                                not 0.015 <= config["hover_clearance_m"] <= 0.15):
        raise ValueError("Measured hover_clearance_m must be 0.015–0.15 m for live motion")

    with rgb_camera(config) as cam:
        matches_camera(data, cam.intrinsics, config)
        with click_view(cam.intrinsics.width, cam.intrinsics.height) as (viewer, address):
            print(f"Open {address} to view RGB and click a pixel. Ctrl-C to stop.")
            with (nullcontext() if args.simulate else station_robot(config, args.enable_live)) as robot:
                kin = Kinematics(config["urdf"]) if args.enable_live or args.simulate else None
                twin = Twin(config["urdf"]) if args.enable_live or args.simulate else None
                if args.simulate:
                    print("SIMULATION ONLY: Rerun starts at zero joint angles; clicks solve IK and animate a virtual arm. "
                          "No follower connection or motor commands.")
                elif twin:
                    print("Rerun shows the live arm and clicked TCP goal (preview only). "
                          "Missing workspace bounds still block motion.")
                simulated_state = {**dict.fromkeys(limits, 0.0), "gripper": 0.0} if args.simulate else None
                if args.simulate:
                    twin.update(simulated_state, ee_pose=kin.fk(simulated_state, tcp))
                try:
                    while True:
                        frame = cam.read()
                        viewer.publish(frame.rgb_bgr)
                        if args.enable_live:
                            state = joint_state(robot)
                            twin.update(state, ee_pose=kin.fk(state, tcp))
                        for x, y, image_seq in viewer.consume():
                            chosen = (x, y)
                            try:
                                if image_seq == 0 or viewer.frame_seq - image_seq > 10:
                                    raise ValueError("Click again after the RGB image appears")
                                require_fresh(frame)
                                if args.enable_live or args.simulate:
                                    target = preview_hover_target(chosen, cam.intrinsics, data["base_T_camera"],
                                                                  plane, config["hover_clearance_m"])
                                    twin.show_hover_goal(target)
                                    if args.simulate:
                                        joints = kin.simulate_tcp_ik(simulated_state, target, tcp, limits)
                                        simulated_state.update(joints)
                                        twin.update(simulated_state, ee_pose=kin.fk(simulated_state, tcp))
                                        message = (f"SIMULATION ONLY: RGB pixel {chosen} candidate TCP (m) {target}; "
                                                   f"IK joints {joints}; no motor commands")
                                        print(message)
                                        viewer.report(message)
                                        continue
                                    workspace_check(target, config.get("workspace_m"))
                                    current_pose = kin.fk(state, tcp)
                                    workspace_check(current_pose[:3, 3], config.get("workspace_m"))
                                    surface = plane["point"]
                                    upward = plane["normal"]
                                    # if np.dot(current_pose[:3, 3] - surface, upward) < config["hover_clearance_m"]:
                                    #     raise ValueError("Current TCP below safe hover height; no motion")
                                    current_pose[:3, 3] = target
                                    joints = kin.hover_ik(state, current_pose, tcp, limits)
                                    message = f"RGB pixel {chosen} hover TCP in base (m) {target}; IK joints {joints}"
                                    print(message)
                                    viewer.report(message + "; terminal YES required for motion")
                                    # if input("Area clear, manual power cutoff ready; hover (never descend)? Type YES to continue: ").strip() == "YES":
                                    move_joints(robot, joints, limits, speed_deg_s=50.0, period_s=0.01)
                                else:
                                    point = intersect_pixel(chosen, cam.intrinsics, data["base_T_camera"],
                                                            plane["point"], plane["normal"])
                                    message = (f"RGB pixel {chosen} base_link z=0 intersection (m) {point} "
                                               "(preview only; not a validated hover TCP)")
                                    print(message)
                                    viewer.report(message)
                            except (ValueError, RuntimeError) as exc:
                                message = f"Rejected click: {exc}"
                                print(message)
                                viewer.report(message)
                        time.sleep(0.03)
                except KeyboardInterrupt:
                    print("RGB click view stopped.")


if __name__ == "__main__":
    main()