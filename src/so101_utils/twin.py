"""Rerun twin of the follower URDF. Driven by joint feedback, not a second model."""

from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from so101_control.kinematics import JOINTS
from so101_utils.keys import keyboard


class Twin:
    """One 3D view plus joint and gripper-pose plots."""

    def __init__(self, urdf: str | Path, *, spawn: bool = True, recording: str | None = None,
                 camera: bool = False):
        import rerun as rr
        import rerun.blueprint as rrb

        path = Path(urdf).resolve()
        if path.name != "so101_new_calib.urdf" or not path.is_file():
            raise ValueError("Twin requires the downloaded so101_new_calib.urdf")
        self.rr = rr
        self.tree = rr.urdf.UrdfTree.from_file_path(path)
        self.joints = {name: self.tree.get_joint_by_name(name) for name in JOINTS}
        if any(joint is None or joint.joint_type != "revolute" for joint in self.joints.values()):
            raise ValueError("Follower URDF missing an arm revolute joint")
        self.jaw = self.tree.get_joint_by_name("gripper")
        if self.jaw is None or self.jaw.joint_type != "revolute":
            raise ValueError("Follower URDF missing gripper jaw joint")
        self.tick = 0
        self.camera = camera
        self.ee_tool_offset = None
        rr.init("so101_control", spawn=spawn and recording is None)
        if recording is not None:
            rr.save(recording)

        self.tree.log_urdf_to_recording()
        rr.get_global_data_recording().flush()
        robot_root = f"/{self.tree.name}"
        rr.log(robot_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(f"{robot_root}/ee_pose", rr.CoordinateFrame("fk_ee_pose"),
               rr.TransformAxes3D(0.04), static=True)
        # Longer than ee_pose so the goal frame still shows once the arm arrives.
        rr.log(f"{robot_root}/hover_goal", rr.CoordinateFrame("hover_goal"),
               rr.TransformAxes3D(0.07), static=True)

        colors = ([255, 94, 111], [255, 188, 92], [91, 197, 233],
                  [169, 130, 255], [79, 207, 148])
        for name, color in zip(JOINTS, colors):
            rr.log(f"telemetry/joints/{name}", rr.SeriesLines(names=name, colors=color), static=True)
        # Metres and degrees stay on separate plots so the position traces stay visible.
        for group, names, group_colors in (
            ("xyz", ("x_m", "y_m", "z_m"), colors[:3]),
            ("rpy", ("roll_deg", "pitch_deg", "yaw_deg"), (*colors[3:], [255, 255, 255])),
        ):
            for name, color in zip(names, group_colors):
                rr.log(f"telemetry/ee/{group}/{name}", rr.SeriesLines(names=name, colors=color), static=True)

        arm_view = rrb.Spatial3DView(
            name="SO-101", origin=robot_root,
            contents=[f"{robot_root}/**", "/tf_static", "/transforms/**"],
            overrides={f"{robot_root}/collision_geometries": rrb.EntityBehavior(visible=False)},
        )
        joint_plot = rrb.TimeSeriesView(name="Joint angles (degrees)", origin="/telemetry/joints")
        pose_plots = rrb.Horizontal(
            rrb.TimeSeriesView(name="EE XYZ (m)", origin="/telemetry/ee/xyz"),
            rrb.TimeSeriesView(name="EE RPY (deg)", origin="/telemetry/ee/rpy"),
        )
        if camera:
            # The camera image is shown here. Rerun does not send clicks on that
            # image back to Python, so click-and-go still uses its browser page.
            side = rrb.Vertical(
                rrb.Spatial2DView(name="Camera", origin="/camera"),
                joint_plot,
            )
        else:
            side = rrb.Vertical(joint_plot, pose_plots)
        blueprint = rrb.Blueprint(
            rrb.Horizontal(arm_view, side),
            auto_views=False,
        )
        rr.send_blueprint(blueprint)
        if recording is not None:
            blueprint.save("so101_control", Path(recording).with_suffix(".rbl"))

    def mark_ee_tool(self, metres: float) -> None:
        """Draw the tool point: `metres` along gripper X. Negative is opposite the red arrow."""
        offset = float(metres)
        if not np.isfinite(offset) or offset == 0.0:
            raise ValueError("EE tool offset must be a non-zero distance")
        self.ee_tool_offset = offset
        root = f"/{self.tree.name}"
        self.rr.log(f"{root}/ee_tool", self.rr.CoordinateFrame("ee_tool"), static=True)
        self.rr.log(f"{root}/ee_tool", self.rr.Points3D(
            [[0.0, 0.0, 0.0]], radii=0.007, colors=[[255, 210, 40]],
            labels=[f"ee {offset * 1000:+.0f} mm"],
        ), static=True)
        self.rr.log(f"{root}/ee_tool_line", self.rr.CoordinateFrame("ee_tool_line"), static=True)
        self.rr.log("transforms/ee_tool_line", self.rr.Transform3D(
            translation=[0.0, 0.0, 0.0], parent_frame="base_link", child_frame="ee_tool_line",
        ), static=True)

    def show_goal_pose(self, pose) -> None:
        """Draw the tool pose `move_to` solved.

        Rotation is gripper_frame_link. Translation is the tool point, offset
        along that frame's X axis when `ee_tool_offset` is set.
        """
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Invalid goal pose")
        translation = matrix[:3, 3]
        if self.ee_tool_offset:
            translation = translation + matrix[:3, :3] @ np.array([self.ee_tool_offset, 0.0, 0.0])
        self.rr.log("transforms/hover_goal", self.rr.Transform3D(
            translation=translation, mat3x3=matrix[:3, :3],
            parent_frame="base_link", child_frame="hover_goal"))

    def update(self, state: dict[str, float], *, ee_pose=None) -> None:
        self.tick += 1
        self.rr.set_time("tick", sequence=self.tick)
        for name, joint in self.joints.items():
            angle = float(state[name])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid joint feedback {name}")
            self.rr.log(f"transforms/{name}", joint.compute_transform(float(np.deg2rad(angle)), clamp=False))
            self.rr.log(f"telemetry/joints/{name}", self.rr.Scalars(angle))

        # 0 is closed and 100 is open. The URDF has no servo-to-jaw map, so the
        # jaw angle is only a picture. Do not treat it as a grasp width.
        gripper = float(state["gripper"])
        if not np.isfinite(gripper):
            raise ValueError("Invalid gripper feedback")
        jaw_angle = float(np.interp(
            np.clip(gripper, 0.0, 100.0), [0.0, 100.0],
            [self.jaw.limit_lower, self.jaw.limit_upper],
        ))
        self.rr.log("transforms/gripper", self.jaw.compute_transform(jaw_angle, clamp=True))

        if ee_pose is not None:
            pose = np.asarray(ee_pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Invalid EE FK pose")
            xyz = pose[:3, 3]
            rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
            self.rr.log("transforms/fk_ee_pose", self.rr.Transform3D(
                translation=xyz, mat3x3=pose[:3, :3],
                parent_frame="base_link", child_frame="fk_ee_pose",
            ))
            if self.ee_tool_offset:
                tip = xyz + pose[:3, :3] @ np.array([self.ee_tool_offset, 0.0, 0.0])
                self.rr.log("transforms/ee_tool", self.rr.Transform3D(
                    translation=tip, parent_frame="base_link", child_frame="ee_tool"))
                self.rr.log(f"/{self.tree.name}/ee_tool_line", self.rr.LineStrips3D(
                    [np.stack([xyz, tip])], colors=[[255, 210, 40]], radii=0.0015,
                ))
            for group, names, values in (
                ("xyz", ("x_m", "y_m", "z_m"), xyz),
                ("rpy", ("roll_deg", "pitch_deg", "yaw_deg"), rpy),
            ):
                for name, value in zip(names, values):
                    self.rr.log(f"telemetry/ee/{group}/{name}", self.rr.Scalars(float(value)))

    def log_camera(self, image_bgr) -> None:
        """Show one RGB frame. `image_bgr` is OpenCV BGR."""
        import cv2

        if not self.camera:
            raise RuntimeError("This twin was opened without a camera view")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.rr.log("camera/rgb", self.rr.Image(rgb))

    def show(self, arm) -> None:
        state = arm.get_joints()
        self.update(state, ee_pose=arm._kin.fk(state))


def watch_arm(arm, urdf: str | Path, *, camera: bool = False) -> Twin:
    """Open Rerun and redraw the arm after every motion step."""
    twin = Twin(urdf, camera=camera)
    twin.ee_tool_offset = float(arm.ee_tool_offset)

    def refresh() -> None:
        twin.show(arm)
        twin.rr.get_global_data_recording().flush(timeout_sec=0)

    arm.track(refresh)

    def mark_target(pose) -> None:
        twin.show_goal_pose(pose)
        twin.rr.get_global_data_recording().flush(timeout_sec=0)

    arm.on_target(mark_target)
    return twin


def play_demo(arm, twin: Twin) -> None:
    """Log five synthetic poses, including the jaw opening and closing."""
    if not arm.mock:
        raise RuntimeError("Demo poses are for --mock only")
    sequence = ({}, {"shoulder_pan": 25.0}, {"elbow_flex": -20.0}, {"gripper": 100.0}, {"gripper": 0.0})
    for update in sequence:
        arm._backend.send_action({f"{name}.pos": float(value) for name, value in update.items()})
        twin.show(arm)
    twin.rr.get_global_data_recording().flush()


def follow(arm, twin: Twin, *, relax: bool = False) -> None:
    """Read the live arm until Ctrl-C. With relax=True, T toggles torque."""
    print("Live twin. Ctrl-C exits. This does not move the arm.")
    if relax:
        print("T turns torque off. Support the arm; it can fall.")
    listener = keyboard(("t",)) if relax else nullcontext()
    try:
        with listener as keys:
            while True:
                pressed = False
                if keys is not None:
                    while keys.consume() is not None:
                        pressed = True
                if pressed:
                    arm.relax() if arm.torque_on else arm.hold()
                twin.show(arm)
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Twin closed.")


class HandEyeView:
    """Twin, the RGB image, and the tag axes while collecting hand-eye samples."""

    def __init__(self, urdf, intrinsics, gripper_T_tag):
        import rerun as rr
        import rerun.blueprint as rrb
        from pupil_apriltags import Detector

        from so101_utils.tags import _arrows

        self.twin = Twin(urdf, spawn=True)
        self.rr = self.twin.rr
        self.intrinsics = intrinsics
        self.gripper_T_tag = np.asarray(gripper_T_tag, dtype=float)
        self.detector = Detector(families="tag36h11")
        self._arrows = _arrows
        root = f"/{self.twin.tree.name}"
        rr.log("camera/rgb", rr.Pinhole(
            focal_length=[intrinsics.fx, intrinsics.fy],
            principal_point=[intrinsics.cx, intrinsics.cy],
            resolution=[intrinsics.width, intrinsics.height]), static=True)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(
                name="Arm and camera", origin=root,
                contents=[f"{root}/**", "/transforms/**"],
                overrides={f"{root}/collision_geometries": rrb.EntityBehavior(visible=False)},
            ),
            rrb.Spatial2DView(name="RGB", origin="/camera/rgb"),
        ), auto_views=False))

    def update(self, frame, arm, base_T_camera=None):
        import cv2

        from so101_utils.tags import draw_tag_overlay, find_tag

        state = arm.get_joints()
        self.twin.update(state, ee_pose=arm._kin.fk(state))
        base_T_gripper = arm._kin.gripper_link()
        self.rr.log("transforms/tag_mount", self.rr.Transform3D(
            translation=self.gripper_T_tag[:3, 3], mat3x3=self.gripper_T_tag[:3, :3],
            parent_frame="gripper_link", child_frame="tag_mount"))
        image = frame.rgb_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(image, self.intrinsics, detector=self.detector)
        except ValueError as exc:
            camera_T_tag = None
            cv2.putText(image, str(exc), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 220), 2)
        else:
            draw_tag_overlay(image, corners, 0)
            if base_T_camera is not None:
                seen = base_T_camera @ camera_T_tag
                self.rr.log("transforms/camera_tag", self.rr.Transform3D(
                    translation=seen[:3, 3], mat3x3=seen[:3, :3],
                    parent_frame="base_link", child_frame="camera_tag"))
        if base_T_camera is not None:
            self.rr.log("transforms/station_camera", self.rr.Transform3D(
                translation=base_T_camera[:3, 3], mat3x3=base_T_camera[:3, :3],
                parent_frame="base_link", child_frame="station_camera"))
            self.rr.log(f"/{self.twin.tree.name}/camera_axes",
                        self._arrows(self.rr, base_T_camera, "camera"))
        self.rr.log(f"/{self.twin.tree.name}/tag_axes",
                    self._arrows(self.rr, base_T_gripper @ self.gripper_T_tag, "tag"))
        self.rr.log("camera/rgb", self.rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        return camera_T_tag
