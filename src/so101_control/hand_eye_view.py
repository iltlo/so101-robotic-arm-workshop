"""Live RGB/tag/robot calibration diagnostics; no robot actuation."""

import numpy as np

from .calibration import find_tag
from .tag_view import draw_tag_overlay, pose_arrows
from .twin import Twin


_pose_arrows = pose_arrows


class HandEyeView:
    def __init__(self, urdf, intrinsics, gripper_T_tag):
        import rerun as rr
        import rerun.blueprint as rrb

        self.twin = Twin(urdf, spawn=True)
        self.rr = rr
        self.intrinsics = intrinsics
        self.gripper_T_tag = gripper_T_tag
        from pupil_apriltags import Detector

        self.detector = Detector(families="tag36h11")
        root = f"/{self.twin.tree.name}"
        self.root = root
        rr.log(f"{root}/tag_mount", rr.CoordinateFrame("tag_mount"), rr.TransformAxes3D(0.03), static=True)
        rr.log("camera/pose", rr.CoordinateFrame("station_camera"), rr.TransformAxes3D(0.05), static=True)
        rr.log("camera/tag_observed", rr.CoordinateFrame("camera_tag"), rr.TransformAxes3D(0.03), static=True)
        rr.log("camera/rgb", rr.CoordinateFrame("station_camera"), static=True)
        for name in ("ee", "tag", "camera"):
            rr.log(f"{root}/calibration_axes/{name}", rr.CoordinateFrame("base_link"), static=True)
        rr.log("camera/rgb", rr.Pinhole(focal_length=[intrinsics.fx, intrinsics.fy],
                                        principal_point=[intrinsics.cx, intrinsics.cy],
                                        resolution=[intrinsics.width, intrinsics.height]), static=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Robot and tag frames", origin=root,
                                  contents=[f"{root}/**", "/tf_static", "/transforms/**", "/camera/**"],
                                  overrides={f"{root}/collision_geometries": rrb.EntityBehavior(visible=False)}),
                rrb.Spatial2DView(name="RealSense RGB (tag36h11 ID 0)", origin="/camera/rgb"),
            ), auto_views=False))

    def update(self, frame, state, base_T_gripper, ee_pose, base_T_camera=None):
        import cv2

        self.twin.update(state, ee_pose=ee_pose)
        self.rr.log(f"{self.root}/calibration_axes/ee", _pose_arrows(self.rr, ee_pose, "EE"))
        base_T_tag_mount = base_T_gripper @ self.gripper_T_tag
        self.rr.log(f"{self.root}/calibration_axes/tag", _pose_arrows(self.rr, base_T_tag_mount, "tag"))
        self.rr.log("transforms/tag_mount", self.rr.Transform3D(
            translation=self.gripper_T_tag[:3, 3], mat3x3=self.gripper_T_tag[:3, :3],
            parent_frame="gripper_link", child_frame="tag_mount"))
        if base_T_camera is not None:
            self.rr.log(f"{self.root}/calibration_axes/camera", _pose_arrows(self.rr, base_T_camera, "camera"))
            self.rr.log("transforms/station_camera", self.rr.Transform3D(
                translation=base_T_camera[:3, 3], mat3x3=base_T_camera[:3, :3],
                parent_frame="base_link", child_frame="station_camera"))
        image = frame.rgb_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(image, self.intrinsics, detector=self.detector)
        except ValueError as exc:
            camera_T_tag = None
            cv2.putText(image, str(exc), (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (20, 30, 240), 2, cv2.LINE_AA)
        else:
            draw_tag_overlay(image, corners, 0)
            if base_T_camera is not None:
                bTtag = base_T_camera @ camera_T_tag
                self.rr.log("transforms/camera_tag", self.rr.Transform3D(
                    translation=bTtag[:3, 3], mat3x3=bTtag[:3, :3],
                    parent_frame="base_link", child_frame="camera_tag"))
        self.rr.log("camera/rgb", self.rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        return camera_T_tag

    def show_camera_estimate(self, base_T_camera, *, sample_count, confirmed=False):
        """Show provisional/final camera pose and each collected hypothesis."""
        pose = np.asarray(base_T_camera, dtype=float)
        self.rr.log(f"{self.root}/calibration_axes/camera",
                    _pose_arrows(self.rr, pose, "camera"))
        self.rr.log("transforms/station_camera", self.rr.Transform3D(
            translation=pose[:3, 3], mat3x3=pose[:3, :3],
            parent_frame="base_link", child_frame="station_camera"))
        self.rr.log("calibration/camera_status", self.rr.TextLog(
            f"{'ACCEPTED' if confirmed else 'PREVIEW ONLY'}: {sample_count} poses; "
            "verify multiple viewpoints before trusting camera extrinsics"))
        if not confirmed:
            self.rr.log(f"{self.root}/calibration_axes/candidates/{sample_count}",
                        self.rr.CoordinateFrame("base_link"), static=True)
            self.rr.log(f"{self.root}/calibration_axes/candidates/{sample_count}",
                        _pose_arrows(self.rr, pose, f"sample {sample_count}", length=0.025))