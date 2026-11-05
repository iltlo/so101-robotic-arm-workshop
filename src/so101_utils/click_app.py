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


def select_box(click, boxes, resolution, intrinsics):
    """The smallest detection box under a click, its index, and its centre pixel.

    Boxes use the detection resolution, which can differ from the camera frame.
    """
    width = float(intrinsics.width)
    height = float(intrinsics.height)
    res_x, res_y = (
        (float(value) for value in resolution) if resolution else (width, height)
    )
    x = click[0] * res_x / width
    y = click[1] * res_y / height
    found = None
    for index, box in enumerate(boxes):
        inside = box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]
        if not inside:
            continue
        area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        if found is None or area < found[0]:
            found = (area, index, box)
    if found is None:
        raise ValueError("Click inside a detection box to pick that object")
    _, index, box = found
    centre = (
        int((box["x1"] + box["x2"]) / 2.0 * width / res_x),
        int((box["y1"] + box["y2"]) / 2.0 * height / res_y),
    )
    return index, box, centre
