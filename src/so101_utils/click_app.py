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


def read_frame(cam, view, show, prepare=None):
    """Read one frame and publish it. `prepare` can annotate or detect first."""
    frame = cam.read()
    view.publish(prepare(frame.rgb_bgr) if prepare is not None else frame.rgb_bgr)
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


def _detection_scale(resolution, intrinsics):
    """Camera-frame size and detection-frame size, as floats.

    A missing detection resolution means the boxes are already in camera pixels.
    """
    width = float(intrinsics.width)
    height = float(intrinsics.height)
    if resolution:
        res_x, res_y = (float(value) for value in resolution)
    else:
        res_x, res_y = width, height
    return width, height, res_x, res_y


def box_centre(box, resolution, intrinsics) -> tuple[int, int]:
    """Box centre mapped from detection pixels into the camera frame."""
    width, height, res_x, res_y = _detection_scale(resolution, intrinsics)
    return (
        int((box["x1"] + box["x2"]) / 2.0 * width / res_x),
        int((box["y1"] + box["y2"]) / 2.0 * height / res_y),
    )


def select_box(click, boxes, resolution, intrinsics):
    """The smallest detection box under a click, its index, and its centre pixel.

    Boxes use the detection resolution, which can differ from the camera frame.
    """
    width, height, res_x, res_y = _detection_scale(resolution, intrinsics)
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
    return index, box, box_centre(box, resolution, intrinsics)


def run_click_loop(cam, view, show, on_click, prepare=None) -> None:
    """Publish frames and hand each click to `on_click` until Ctrl-C.

    `prepare` sees each BGR frame before it is published, so a detection script
    can update its box list and draw boxes. ValueError and RuntimeError from
    `on_click` are printed and shown on the page. The callback decides whether a
    click that arrived during a move is kept.
    """
    try:
        while True:
            read_frame(cam, view, show, prepare)
            click = view.consume()
            if click is None:
                continue
            try:
                on_click(click)
            except (ValueError, RuntimeError) as exc:
                view.report(str(exc))
                print(exc)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        view.close()
