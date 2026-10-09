"""Show the AprilTag in the camera frame. The arm is not connected.

Tag family tag36h11, id 0, 40 mm black-square edge. Ctrl-C stops.
"""

from so101_utils.camera import open_camera
from so101_utils.tags import TagView


if __name__ == "__main__":
    with open_camera() as cam:
        view = TagView(cam.intrinsics, tag_id=0, tag_size_m=0.04)
        print("AprilTag in Rerun. Ctrl-C stops. No arm motion.")
        try:
            while True:
                pose = view.update(cam.read().rgb_bgr)
                if pose is not None:
                    x, y, z = pose[:3, 3]
                    print(f"\rtag in camera: {x:.3f} {y:.3f} {z:.3f} m", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
