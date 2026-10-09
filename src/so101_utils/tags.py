"""AprilTag pose in the camera frame, and the Rerun tag viewer."""

import numpy as np

from so101_utils.transforms import Intrinsics, normalized_ray, transform


def gripper_link_pose(arm) -> np.ndarray:
    """4x4 pose of the tag's mount link (gripper_link), in the base frame."""
    return arm._kin.fk_gripper_link(arm.get_joints())


def draw_tag_overlay(image_bgr, corners, tag_id: int) -> None:
    import cv2

    outline = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_bgr, [outline], True, (0, 255, 0), 2, cv2.LINE_AA)
    center = np.rint(corners.mean(axis=0)).astype(int)
    cv2.circle(image_bgr, tuple(center), 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"tag36h11 ID {tag_id}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 20), 2, cv2.LINE_AA)


def _pose_from_corners(corners, intr: Intrinsics, size_m: float) -> np.ndarray:
    """IPPE on the tag square. Corner order matches pupil_apriltags, not image BL/BR/TR/TL."""
    import cv2

    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Four ordered tag corners required")
    half = size_m / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0],
                       [-half, -half, 0]], dtype=float)
    success, rvec, tvec = cv2.solvePnP(
        points, corners, intr.matrix(),
        np.asarray(intr.distortion or [0, 0, 0, 0, 0], dtype=float),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Tag pose ambiguous or behind the camera")
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec[:, 0]
    return transform(pose)


def find_tag(image_bgr, intr: Intrinsics, *, tag_id=0, tag_size_m=0.04, detector=None):
    """Return camera_T_tag (metres) and the four RGB corners. No depth."""
    import cv2

    intr.matrix()
    if tag_size_m <= 0 or image_bgr.shape[:2] != (intr.height, intr.width):
        raise ValueError("Tag size or image size does not match the camera")
    if detector is None:
        from pupil_apriltags import Detector
        detector = Detector(families="tag36h11")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tags = [tag for tag in detector.detect(gray) if tag.tag_id == tag_id]
    if len(tags) != 1:
        raise ValueError(f"Expected one tag36h11 ID {tag_id}, found {len(tags)}")
    corners = np.asarray(tags[0].corners, dtype=float)
    # Native detector order is IPPE (-x,+y), (+x,+y), (+x,-y), (-x,-y).
    undistorted = np.asarray([normalized_ray(tuple(corner), intr) for corner in corners])
    undistorted = undistorted * [intr.fx, intr.fy] + [intr.cx, intr.cy]
    pose = _pose_from_corners(
        undistorted, Intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy),
        tag_size_m,
    )
    return pose, corners


def _format_pose(pose, *, tag_id: int) -> str:
    pose = np.asarray(pose, dtype=float)
    rows = "\n".join("[" + " ".join(f"{value:8.4f}" for value in row) + "]" for row in pose)
    x, y, z = pose[:3, 3]
    return (f"Tag ID: {tag_id}\n"
            "Camera frame: +X right, +Y down, +Z forward\n"
            f"Tag origin (metres): X={x:.4f}, Y={y:.4f}, Z={z:.4f}\n"
            f"camera_T_tag:\n{rows}")


def _arrows(rr, pose, label, *, length=0.05):
    pose = np.asarray(pose, dtype=float)
    return rr.Arrows3D(
        origins=np.repeat(pose[:3, 3][None, :], 3, axis=0),
        vectors=pose[:3, :3].T * length,
        colors=[[255, 70, 70], [70, 240, 110], [65, 140, 255]],
        radii=[0.003, 0.003, 0.003],
        labels=[f"{label} X", f"{label} Y", f"{label} Z"],
    )


class TagPoseView:
    """Rerun view of one tag in the camera frame. The arm is not involved."""

    def __init__(self, intrinsics: Intrinsics, *, tag_id=0, tag_size_m=0.04, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb
        from pupil_apriltags import Detector

        if tag_size_m <= 0:
            raise ValueError("Tag size must be positive, in metres")
        self.rr = rr
        self.intrinsics = intrinsics
        self.tag_id = tag_id
        self.tag_size_m = tag_size_m
        self.detector = Detector(families="tag36h11")
        rr.init("SO-101 AprilTag", spawn=spawn)
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)
        rr.log("camera/origin", _arrows(rr, np.eye(4), "camera", length=0.04), static=True)
        rr.log("camera/rgb", rr.Pinhole(
            focal_length=[intrinsics.fx, intrinsics.fy],
            principal_point=[intrinsics.cx, intrinsics.cy],
            resolution=[intrinsics.width, intrinsics.height],
        ), static=True)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(name="Camera frame (metres)", origin="/camera"),
            rrb.Vertical(
                rrb.Spatial2DView(name="RGB", origin="/camera/rgb"),
                rrb.TextDocumentView(name="camera_T_tag", origin="/camera/pose_text"),
            ),
        ), auto_views=False))

    def update(self, image_bgr):
        """Return camera_T_tag, or None when the tag is not visible."""
        import cv2

        image = image_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(
                image, self.intrinsics, tag_id=self.tag_id,
                tag_size_m=self.tag_size_m, detector=self.detector,
            )
        except ValueError as exc:
            self.rr.log("camera/tag", self.rr.Clear(recursive=True))
            self.rr.log("camera/tag_axes", self.rr.Clear(recursive=False))
            self.rr.log("camera/pose_text", self.rr.TextDocument(f"No tag: {exc}"))
            cv2.putText(image, str(exc), (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (20, 30, 240), 2, cv2.LINE_AA)
            result = None
        else:
            draw_tag_overlay(image, corners, self.tag_id)
            self.rr.log("camera/tag", self.rr.Transform3D(
                translation=camera_T_tag[:3, 3], mat3x3=camera_T_tag[:3, :3]))
            self.rr.log("camera/tag_axes", _arrows(self.rr, camera_T_tag, "tag", length=0.03))
            self.rr.log("camera/pose_text", self.rr.TextDocument(
                _format_pose(camera_T_tag, tag_id=self.tag_id)))
            result = camera_T_tag
        self.rr.log("camera/rgb", self.rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        return result


TagView = TagPoseView
