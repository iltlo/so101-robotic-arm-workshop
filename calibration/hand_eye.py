"""Eye-to-hand calibration. Fixed camera, AprilTag on the gripper.

The tag is tag36h11 id 0, 40 mm edge, mounted on gripper_link.
Keys (X11): T torque off/on, Enter record, C average and save, Q quit.
Support the arm before T turns torque off.

Saves config/hand_eye.yaml and refreshes config/camera.yaml intrinsics.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from so101_control import SO101
from so101_utils.camera import open_camera
from so101_utils.config import load_robot, save_camera_intrinsics, save_hand_eye
from so101_utils.keys import keyboard
from so101_utils.transforms import Sample, solve_eye_to_hand
from so101_utils.twin import HandEyeView

# Measured mount, metres: [-37.2, 0, -28.4] mm, then Ry(90°) and Rz(180°).
GRIPPER_T_TAG = np.eye(4)
GRIPPER_T_TAG[:3, 3] = [-0.0372, 0.0, -0.0284]
GRIPPER_T_TAG[:3, :3] = (Rotation.from_euler("y", 90, degrees=True)
                         * Rotation.from_euler("z", 180, degrees=True)).as_matrix()


def _settled(before, after, limit=0.3) -> bool:
    return max(abs(before[name] - after[name]) for name in before) <= limit


if __name__ == "__main__":
    config = load_robot()
    samples = []
    estimate = None
    saved = False
    print("T relaxes the arm (support it). Enter records. C solves and saves. Q quits.")
    with SO101.connect(mock=False) as arm, open_camera() as cam:
        save_camera_intrinsics(cam.intrinsics, serial=None)
        view = HandEyeView(config["urdf"], cam.intrinsics, GRIPPER_T_TAG)
        with keyboard(("t", "enter", "c", "q")) as keys:
            while True:
                frame = cam.read()
                camera_T_tag = view.update(frame, arm, estimate)
                command = keys.consume()
                if command == "q":
                    print("Saved." if saved else f"Quit. Discarded {len(samples)} unsaved samples.")
                    break
                if command == "t":
                    arm.relax() if arm.torque_on else arm.hold()
                    continue
                if command == "c":
                    if not arm.torque_on:
                        print("Turn torque on before saving.")
                        continue
                    try:
                        solved = solve_eye_to_hand(samples, GRIPPER_T_TAG)
                    except ValueError as exc:
                        print(exc)
                        continue
                    save_hand_eye(solved["base_T_camera"], residual_m=solved["max_residual_m"],
                                  residual_deg=solved["max_residual_deg"])
                    estimate = solved["base_T_camera"]
                    saved = True
                    print(f"Saved. Residual {solved['max_residual_m']*1000:.1f} mm, "
                          f"{solved['max_residual_deg']:.1f} deg.")
                    continue
                if command != "enter":
                    continue
                if not arm.torque_on:
                    print("Turn torque on, let the arm settle, then press Enter.")
                    continue
                if camera_T_tag is None:
                    print("No tag in view.")
                    continue
                before = arm.get_joints()
                time.sleep(0.25)
                after = arm.get_joints()
                if not _settled(before, after):
                    print("Arm moved. Wait and try again.")
                    continue
                again = view.update(cam.read(), arm, estimate)
                if again is None or not _settled(after, arm.get_joints()):
                    print("Tag or arm moved during capture.")
                    continue
                base_T_gripper = arm.gripper_link_pose()
                samples.append(Sample(base_T_gripper, again))
                # One-pose preview. C averages the whole set.
                estimate = base_T_gripper @ GRIPPER_T_TAG @ np.linalg.inv(again)
                saved = False
                print(f"Captured {len(samples)}. Press C once you have at least 6 varied poses.")
