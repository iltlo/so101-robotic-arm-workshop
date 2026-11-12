"""Show the AprilTag in the camera frame. The arm is not connected.

Tag family, id, and size come from config/robot.yaml. Ctrl-C stops.
"""

from so101_utils.camera import open_camera
from so101_utils.config import load_robot
from so101_utils.tags import tag_rpy_deg
from so101_utils.views import TagView


if __name__ == "__main__":
    tag = load_robot()["tag"]
    with open_camera() as cam:
        view = TagView(cam.intrinsics, tag)
        print("AprilTag in Rerun. Ctrl-C stops. No arm motion.")
        try:
            while True:
                pose = view.update(cam.read().rgb_bgr)
                if pose is not None:
                    x, y, z = pose[:3, 3]
                    roll, pitch, yaw = tag_rpy_deg(pose)
                    print(
                        f"\rtag in camera: xyz {x:.3f} {y:.3f} {z:.3f} m  "
                        f"rpy {roll:.1f} {pitch:.1f} {yaw:.1f} deg",
                        end="", flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped.")
