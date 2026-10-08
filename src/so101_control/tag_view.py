"""Reusable Rerun diagnostics for one tag in the RGB camera optical frame."""

import numpy as np

from .calibration import find_tag


def pose_arrows(rr, pose, label, *, length=0.055):
    """Render X/Y/Z as red/green/blue arrows in the coordinates of pose."""
    pose = np.asarray(pose, dtype=float)
    return rr.Arrows3D(
        origins=np.repeat(pose[:3, 3][None, :], 3, axis=0),
        vectors=pose[:3, :3].T * length,
        colors=[[255, 70, 70], [70, 240, 110], [65, 140, 255]],
        radii=[0.003, 0.003, 0.003],
        labels=[f"{label} X", f"{label} Y", f"{label} Z"],
    )


def format_camera_T_tag(pose, *, tag_id):
    """Describe the transform mapping tag-frame points into camera optical axes."""
    pose = np.asarray(pose, dtype=float)
    rows = "\n".join("[" + " ".join(f"{value:8.4f}" for value in row) + "]"
                     for row in pose)
    x, y, z = pose[:3, 3]
    return (f"Tag ID: {tag_id}\n"
            "Camera optical frame: +X right, +Y down, +Z forward\n"
            f"Tag origin in camera (metres): X={x:.4f}, Y={y:.4f}, Z={z:.4f}\n"
            "camera_T_tag (maps tag coordinates to camera coordinates; translation metres):\n"
            f"{rows}")


def draw_tag_overlay(image_bgr, corners, tag_id):
    """Outline the detected tag and mark its center on a mutable RGB frame."""
    import cv2

    outline = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_bgr, [outline], True, (0, 255, 0), 2, cv2.LINE_AA)
    center = np.rint(corners.mean(axis=0)).astype(int)
    cv2.circle(image_bgr, tuple(center), 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"tag36h11 ID {tag_id} detected", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 20), 2, cv2.LINE_AA)


class TagPoseView:
    """Display live RGB and camera_T_tag without using robot FK or a tag mount."""

    def __init__(self, intrinsics, *, tag_id=0, tag_size_m=0.04, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb
        from pupil_apriltags import Detector

        if tag_size_m <= 0:
            raise ValueError("Printed AprilTag black-square edge must be positive (metres)")
        self.rr = rr
        self.intrinsics = intrinsics
        self.tag_id, self.tag_size_m = tag_id, tag_size_m
        self.detector = Detector(families="tag36h11")
        rr.init("SO-101 camera-to-AprilTag diagnostic", spawn=spawn)
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)  # optical: right, down, forward
        rr.log("camera/origin", pose_arrows(rr, np.eye(4), "camera", length=0.04), static=True)
        rr.log("camera/rgb", rr.Pinhole(focal_length=[intrinsics.fx, intrinsics.fy],
                                        principal_point=[intrinsics.cx, intrinsics.cy],
                                        resolution=[intrinsics.width, intrinsics.height]), static=True)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(name="Camera optical frame (metres)", origin="/camera",
                              contents=["/camera/origin", "/camera/tag", "/camera/tag_axes"],
                              background=[12, 20, 28]),
            rrb.Vertical(
                rrb.Spatial2DView(name="Live RGB (tag outlined)", origin="/camera/rgb"),
                rrb.TextDocumentView(name="camera_T_tag (metres)", origin="/camera/pose_text"),
            ),
        ), auto_views=False))

    def update(self, image_bgr):
        """Return the fresh camera_T_tag, or None; never display a stale tag pose."""
        import cv2

        image = image_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(image, self.intrinsics, tag_id=self.tag_id,
                                              tag_size_m=self.tag_size_m, detector=self.detector)
        except ValueError as exc:
            self.rr.log("camera/tag", self.rr.Clear(recursive=True))
            self.rr.log("camera/tag_axes", self.rr.Clear(recursive=False))
            self.rr.log("camera/pose_text", self.rr.TextDocument(f"No current pose: {exc}"))
            cv2.putText(image, str(exc), (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (20, 30, 240), 2, cv2.LINE_AA)
            result = None
        else:
            draw_tag_overlay(image, corners, self.tag_id)
            self.rr.log("camera/tag", self.rr.Transform3D(
                translation=camera_T_tag[:3, 3], mat3x3=camera_T_tag[:3, :3]))
            self.rr.log("camera/tag_axes", pose_arrows(self.rr, camera_T_tag, "tag", length=0.03))
            self.rr.log("camera/pose_text", self.rr.TextDocument(
                format_camera_T_tag(camera_T_tag, tag_id=self.tag_id)))
            result = camera_T_tag
        self.rr.log("camera/rgb", self.rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        return result