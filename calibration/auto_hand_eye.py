"""Automatic eye-to-hand calibration. Drives the arm through fixed poses.

Uses the neat poses in calibration/poses.md: five recorded poses, then the same
set mirrored through the base XZ plane (y → −y, yaw → −yaw). Roll stays at
−170° so the wrist does not hit the stop.

Each step sends move_to without waiting on encoder settle (so a soft arm is
not blocked), dwells, then samples FK(gripper_link) plus the tag at wherever
the arm actually stopped. Start and end at the rest pose. Needs at least 4
successful tag captures to save.

    uv run python calibration/auto_hand_eye.py --live
    uv run python calibration/auto_hand_eye.py --mock
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from so101_control import SO101
from so101_control.motion import MotionError
from so101_utils.camera import open_camera
from so101_utils.config import arm_parser, load_robot, save_camera_intrinsics, save_hand_eye
from so101_utils.transforms import Sample, solve_eye_to_hand
from so101_utils.twin import HandEyeView

# Same mount as calibration/hand_eye.py.
GRIPPER_T_TAG = np.eye(4)
GRIPPER_T_TAG[:3, 3] = [-0.0372, 0.0, -0.0284]
GRIPPER_T_TAG[:3, :3] = (Rotation.from_euler("y", 90, degrees=True)
                         * Rotation.from_euler("z", 180, degrees=True)).as_matrix()

# x y z roll pitch yaw. Roll fixed at -170.
_RECORDED = (
    (0.24,  0.03, 0.13, -170.0,  3.0,   8.0),
    (0.23,  0.03, 0.20, -170.0,  2.0,   8.0),
    (0.18, -0.15, 0.20, -170.0,  7.0, -48.0),
    (0.21, -0.16, 0.11, -170.0, -6.0, -44.0),
    (0.31, -0.14, 0.09, -170.0, -2.0, -28.0),
)

# After the joint commands are sent, wait for the arm to finish moving.
DWELL_S = 1.5


def calibration_poses():
    """Recorded poses plus the XZ-plane mirror (flip y and yaw)."""
    poses = list(_RECORDED)
    for x, y, z, roll, pitch, yaw in _RECORDED:
        poses.append((x, -y, z, roll, pitch, -yaw))
    return poses


def capture_sample(arm, cam, view, estimate):
    """One photo + FK(gripper_link) from the joints at that same moment.

    The next move_to is not started until this returns, so the arm is not
    heading to another goal while we shoot.
    """
    time.sleep(DWELL_S)
    frame = cam.read()
    joints = arm.get_joints()
    base_T_gripper = arm._kin.fk_gripper_link(joints)
    camera_T_tag = view.update(frame, arm, estimate)
    if camera_T_tag is None:
        print("  No tag in view; skip.")
        return None, estimate
    sample = Sample(base_T_gripper, camera_T_tag)
    preview = base_T_gripper @ GRIPPER_T_TAG @ np.linalg.inv(camera_T_tag)
    return sample, preview


def run(arm, cam, view) -> bool:
    """Tour the poses, solve, and save. Returns True when a result was written."""
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
            # wait=False: finish sending this pose's trajectory, do not block on
            # encoder settle. Capture runs before the next move_to.
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
    if len(samples) < 4:
        print(f"Only {len(samples)} samples; need at least 4. Not saved.")
        return False
    solved = solve_eye_to_hand(samples, GRIPPER_T_TAG)
    save_hand_eye(solved["base_T_camera"], residual_m=solved["max_residual_m"],
                  residual_deg=solved["max_residual_deg"])
    print(f"Saved {len(samples)} samples. Residual "
          f"{solved['max_residual_m'] * 1000:.1f} mm, {solved['max_residual_deg']:.1f} deg.")
    return True


if __name__ == "__main__":
    args = arm_parser(__doc__).parse_args()
    config = load_robot()
    with SO101.connect(mock=args.mock) as arm, open_camera() as cam:
        view = HandEyeView(config["urdf"], cam.intrinsics, GRIPPER_T_TAG)
        try:
            run(arm, cam, view)
        except KeyboardInterrupt:
            print("Stopped.")
            try:
                arm.rest()
            except Exception:
                pass
