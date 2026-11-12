"""Show the camera from config/camera.yaml. Q quits.

`open_camera()` uses the profile's backend: a RealSense, or the synthetic demo.
Each logged frame is the picture that call just returned.

    uv run python scripts/camera_view.py
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from so101_utils.camera import open_camera
from so101_utils.keys import keyboard


if __name__ == "__main__":
    print("Live camera in Rerun. Q quits.")
    with open_camera() as cam, keyboard(("q",)) as keys:
        rr.init("SO-101 camera", spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Spatial2DView(name="Camera", origin="camera/rgb"),
            auto_views=False,
        ))
        while True:
            image = cam.read().rgb_bgr
            rr.log("camera/rgb", rr.Image(np.ascontiguousarray(image[:, :, ::-1])))
            rr.get_global_data_recording().flush(timeout_sec=0)
            if keys.consume() == "q":
                break
