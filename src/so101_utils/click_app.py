"""Glue shared by the click-to-move application scripts."""

import numpy as np

from so101_utils.transforms import base_link_plane, intersect_pixel


def frame_viewer(arm, twin):
    """Return a `show(frame)` that logs a frame if there is one, then redraws the arm."""

    def show(frame) -> None:
        if frame is not None:
            twin.log_camera(frame.rgb_bgr)
        twin.show(arm)

    return show


def read_frame(cam, view, show):
    """Read one frame, publish it to the page, show it, and return it."""
    frame = cam.read()
    view.publish(frame.rgb_bgr)
    show(frame)
    return frame


def attach_camera(arm, cam, view, show) -> None:
    """Publish the camera during moves, and one frame before returning."""

    def paint() -> None:
        frame = cam.poll()
        if frame is not None:
            view.publish(frame.rgb_bgr)
        show(frame)

    arm.track(paint)
    read_frame(cam, view, show)


def click_target(click, intrinsics, base_T_camera, height_m) -> np.ndarray:
    """Where the click ray meets the base plane, lifted `height_m` along its normal."""
    plane = base_link_plane()
    point = intersect_pixel(click, intrinsics, base_T_camera, plane["point"], plane["normal"])
    return point + plane["normal"] * height_m
