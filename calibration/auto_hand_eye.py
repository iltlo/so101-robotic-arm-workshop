"""Automatic eye-to-hand calibration. Drives the arm through fixed poses.

The recorded poses plus their base-XZ mirror live in calibration/poses.md. Roll
stays at −170° so the wrist does not hit the stop. Each step sends move_to
without waiting on encoder settle, dwells, then samples FK(gripper_link) and the
tag wherever the arm stopped. Needs 4 tag captures to save.

    uv run python calibration/auto_hand_eye.py --live
    uv run python calibration/auto_hand_eye.py --mock
"""

import time

import numpy as np

from so101_control import SO101
from so101_control.motion import MotionError
from so101_utils.auto_hand_eye import DWELL_S, calibration_poses
from so101_utils.camera import open_camera
from so101_utils.config import arm_parser, load_robot, save_camera_intrinsics, save_hand_eye
from so101_utils.interrupts import ignore_sigint
from so101_utils.transforms import MIN_POSES, Sample, solve_eye_to_hand
from so101_utils.views import HandEyeView


def capture_sample(arm, cam, view, estimate):
    """One frame plus FK(gripper_link) from the joints at that same moment."""
    time.sleep(DWELL_S)
    frame = cam.read()
    joints = arm.get_joints()
    base_T_gripper = arm.gripper_link_pose(joints)
    camera_T_tag = view.update(frame, arm, estimate)
    if camera_T_tag is None:
        print("  No tag in view; skip.")
        return None, estimate
    sample = Sample(base_T_gripper, camera_T_tag)
    preview = base_T_gripper @ view.tag.gripper_T_tag @ np.linalg.inv(camera_T_tag)
    return sample, preview


def run(arm, cam, view) -> bool:
    """Tour the poses, solve, and save. Returns True when a result was written."""
    if not getattr(cam, "intrinsics_from_device", False):
        save_camera_intrinsics(cam.intrinsics, serial=None)
    poses = calibration_poses()
    samples = []
    estimate = None
    print(f"Auto hand-eye: {len(poses)} poses. Roll fixed at -170 deg.")
    print("Start at rest (wait). Calibration moves do not wait on encoder settle.")
    arm.rest(wait=True)
    for index, (x, y, z, roll, pitch, yaw) in enumerate(poses, start=1):
        print(f"[{index}/{len(poses)}] move_to "
              f"xyz=({x:.2f}, {y:+.2f}, {z:.2f}) rpy=({roll:.0f}, {pitch:.0f}, {yaw:.0f})")
        try:
            # wait=False: send the trajectory without blocking on encoder settle.
            arm.move_to(x, y, z, roll, pitch, yaw, wait=False)
        except (ValueError, MotionError) as exc:
            print(f"  Move failed: {exc}")
            view.update(cam.read(), arm, estimate)
            continue
        sample, estimate = capture_sample(arm, cam, view, estimate)
        if sample is None:
            continue
        samples.append(sample)
        print(f"  Captured ({len(samples)} samples).")

    arm.rest()
    if len(samples) < MIN_POSES:
        print(f"Only {len(samples)} samples; need at least {MIN_POSES}. Not saved.")
        return False
    solved = solve_eye_to_hand(samples, view.tag.gripper_T_tag)
    save_hand_eye(solved["base_T_camera"], residual_m=solved["max_residual_m"],
                  residual_deg=solved["max_residual_deg"])
    print(f"Saved {len(samples)} samples. Residual "
          f"{solved['max_residual_m'] * 1000:.1f} mm, {solved['max_residual_deg']:.1f} deg.")
    return True


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    config = load_robot()
    with SO101.connect(mock=args.mock) as arm, open_camera() as cam:
        view = HandEyeView(config["urdf"], cam.intrinsics, config["tag"])
        try:
            run(arm, cam, view)
        except KeyboardInterrupt:
            print("Stopped.")
            ignore_sigint()
            try:
                arm.rest()
            except Exception:
                pass
