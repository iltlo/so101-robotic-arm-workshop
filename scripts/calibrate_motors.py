"""Record the motor midpoint and joint ranges into config/motors.json.

Torque turns off. Support the arm. Wrist roll keeps a full turn and is not swept.
"""

from so101_control.backend import connect_live
from so101_utils.config import load_robot


def main() -> None:
    print("Torque will turn off. Support the arm; it can fall.", flush=True)
    follower = connect_live(load_robot(), require_calibration=False)
    try:
        input("Move every joint to the middle of its travel, then press Enter. ")
        follower.bus.set_midpoint()
        follower.bus.start_range_capture()
        print(
            "Move every joint except wrist roll through its whole travel, including the gripper.",
            flush=True,
        )
        input("Press Enter to save. ")
        saved = follower.bus.stop_range_capture()
    finally:
        follower.disconnect()
    print(f"Saved {follower.bus.calibration_path}", flush=True)
    for name, cal in saved.items():
        print(f"  {name}: {cal['range_min']} .. {cal['range_max']}", flush=True)


if __name__ == "__main__":
    main()
