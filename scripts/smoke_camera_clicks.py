"""Smoke: print pixels from `rr.poll_image_click` on the Camera panel.

    uv run python scripts/smoke_camera_clicks.py
"""

import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from so101_utils.views import CAMERA_ENTITY


def main() -> None:
    width, height = 640, 480
    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    image_bgr[:, :] = (120, 80, 40)
    image_bgr[100:380, 180:460] = (60, 180, 220)

    rr.init("so101_smoke_camera_clicks", spawn=True)
    rr.log(
        CAMERA_ENTITY,
        rr.Pinhole(
            focal_length=[600.0, 600.0],
            principal_point=[width / 2, height / 2],
            resolution=[width, height],
        ),
        static=True,
    )
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial2DView(name="Camera", origin=CAMERA_ENTITY),
        auto_views=False,
    ))
    rr.log(CAMERA_ENTITY, rr.Image(image_bgr[:, :, ::-1]))
    rr.get_global_data_recording().flush()
    print("Click the Camera panel in the native viewer. Ctrl-C stops.", flush=True)
    try:
        while True:
            click = rr.poll_image_click(CAMERA_ENTITY, width=width, height=height)
            if click is not None:
                print(f"Got click {click}", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
