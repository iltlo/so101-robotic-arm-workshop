"""The Rerun views: arm twin, hand-eye view, and tag view. Shared layout and helpers."""

from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from so101_utils.keys import keyboard
from so101_utils.model import (
    GRIPPER_CLOSED_PERCENT,
    GRIPPER_OPEN_PERCENT,
    JOINTS,
    tool_rpy,
)
from so101_utils.tags import Tag, draw_error, draw_tag_overlay, find_tag, tag_detector

_JOINT_COLORS = ((255, 94, 111), (255, 188, 92), (91, 197, 233),
                 (169, 130, 255), (79, 207, 148))
_EE_XYZ_COLORS = _JOINT_COLORS[:3]
_EE_RPY_COLORS = ((169, 130, 255), (79, 207, 148), (255, 255, 255))
_AXIS_COLORS = ((255, 70, 70), (70, 240, 110), (65, 140, 255))
_CAMERA = "camera/rgb"

_AXIS_RADIUS_M = 0.003
_GOAL_AXIS_M = 0.07
_FK_AXIS_M = 0.05
_TOOL_AXIS_M = 0.04
_LINK_AXIS_M = 0.035
_TAG_AXIS_M = 0.03
_CAMERA_AXIS_M = 0.04
_TOOL_POINT_RADIUS_M = 0.007
_TOOL_LINE_RADIUS_M = 0.0015
_LIVE_PERIOD_S = 0.1

# Five synthetic poses for scripts/twin.py --mock.
_DEMO_POSES = ({}, {"shoulder_pan": 25.0}, {"elbow_flex": -20.0},
               {"gripper": GRIPPER_OPEN_PERCENT}, {"gripper": GRIPPER_CLOSED_PERCENT})


def _frame_arrows(rr, pose, label, *, length=_FK_AXIS_M):
    """RGB axes at a 4x4 pose in the base frame."""
    pose = np.asarray(pose, dtype=float)
    return rr.Arrows3D(
        origins=np.repeat(pose[:3, 3][None, :], 3, axis=0),
        vectors=pose[:3, :3].T * length,
        colors=[list(color) for color in _AXIS_COLORS],
        radii=[_AXIS_RADIUS_M] * 3,
        labels=[f"{label} X", f"{label} Y", f"{label} Z"],
    )


def _pinhole_entity(rr, intrinsics):
    return rr.Pinhole(
        focal_length=[intrinsics.fx, intrinsics.fy],
        principal_point=[intrinsics.cx, intrinsics.cy],
        resolution=[intrinsics.width, intrinsics.height],
    )


def _log_rgb(rr, image_bgr) -> None:
    import cv2

    rr.log(_CAMERA, rr.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))


def _flush(rr) -> None:
    rr.get_global_data_recording().flush(timeout_sec=0)


def _arm_blueprint(rrb, root, *, intrinsics):
    """3D arm left; joint angles and EE poses right; camera under the arm."""
    arm_view = rrb.Spatial3DView(
        name="SO-101", origin=root,
        contents=[f"{root}/**", "/tf_static", "/transforms/**"],
        overrides={f"{root}/collision_geometries": rrb.EntityBehavior(visible=False)},
    )
    joint_plot = rrb.TimeSeriesView(name="Joint angles (degrees)", origin="/telemetry/joints")
    pose_plots = rrb.Horizontal(
        rrb.TimeSeriesView(name="EE XYZ (m)", origin="/telemetry/ee/xyz"),
        rrb.TimeSeriesView(name="EE RPY (deg)", origin="/telemetry/ee/rpy"),
    )
    telemetry = rrb.Vertical(joint_plot, pose_plots)
    if intrinsics is None:
        layout = rrb.Horizontal(arm_view, telemetry)
    else:
        camera = rrb.Spatial2DView(name="Camera", origin=_CAMERA)
        layout = rrb.Horizontal(rrb.Vertical(arm_view, camera), telemetry)
    return rrb.Blueprint(layout, auto_views=False)


class Twin:
    """One 3D view plus joint-angle and EE pose plots."""

    def __init__(self, urdf: str | Path, *, spawn: bool = True, recording: str | None = None,
                 intrinsics=None):
        import rerun as rr
        import rerun.blueprint as rrb

        path = Path(urdf).resolve()
        if path.name != "so101_new_calib.urdf" or not path.is_file():
            raise ValueError("Twin requires the downloaded so101_new_calib.urdf")
        self.rr = rr
        self.intrinsics = intrinsics
        self.tree = rr.urdf.UrdfTree.from_file_path(path)
        self.joints = {name: self.tree.get_joint_by_name(name) for name in JOINTS}
        if any(joint is None or joint.joint_type != "revolute" for joint in self.joints.values()):
            raise ValueError("Follower URDF missing an arm revolute joint")
        self.jaw = self.tree.get_joint_by_name("gripper")
        if self.jaw is None or self.jaw.joint_type != "revolute":
            raise ValueError("Follower URDF missing gripper jaw joint")
        self.tick = 0
        self.ee_tool_offset = None
        rr.init("so101_control", spawn=spawn and recording is None)
        if recording is not None:
            rr.save(recording)

        self.tree.log_urdf_to_recording()
        rr.get_global_data_recording().flush()
        robot_root = f"/{self.tree.name}"
        rr.log(robot_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # FK / goal geometry is already expressed in base_link metres. Pin those
        # entities to that frame so the view does not look for a missing TF path.
        for entity in ("fk_ee_axes", "ee_tool", "ee_tool_line", "ee_tool_axes", "hover_goal"):
            rr.log(f"{robot_root}/{entity}", rr.CoordinateFrame("base_link"), static=True)

        for name, color in zip(JOINTS, _JOINT_COLORS):
            rr.log(f"telemetry/joints/{name}", rr.SeriesLines(names=name, colors=color), static=True)
        for group, names, colors in (
            ("xyz", ("x_m", "y_m", "z_m"), _EE_XYZ_COLORS),
            ("rpy", ("roll_deg", "pitch_deg", "yaw_deg"), _EE_RPY_COLORS),
        ):
            for name, color in zip(names, colors):
                rr.log(f"telemetry/ee/{group}/{name}", rr.SeriesLines(names=name, colors=color), static=True)

        if intrinsics is not None:
            rr.log(_CAMERA, _pinhole_entity(rr, intrinsics), static=True)
        blueprint = _arm_blueprint(rrb, robot_root, intrinsics=intrinsics)
        rr.send_blueprint(blueprint)
        if recording is not None:
            blueprint.save("so101_control", Path(recording).with_suffix(".rbl"))

    def mark_ee_tool(self, metres: float) -> None:
        """Draw the tool point each update: `metres` along gripper +Z."""
        offset = float(metres)
        if not np.isfinite(offset) or offset == 0.0:
            raise ValueError("EE tool offset must be a non-zero distance")
        self.ee_tool_offset = offset

    def show_goal_pose(self, pose) -> None:
        """Draw the tool pose `move_to` solved.

        Rotation is gripper_frame_link (X out the fingers). Translation is the
        tool point, offset along that frame's +Z when `ee_tool_offset` is set.
        """
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Invalid goal pose")
        drawn = matrix.copy()
        if self.ee_tool_offset:
            drawn[:3, 3] = matrix[:3, 3] + matrix[:3, :3] @ np.array(
                [0.0, 0.0, self.ee_tool_offset])
        self.rr.log(f"/{self.tree.name}/hover_goal",
                    _frame_arrows(self.rr, drawn, "goal", length=_GOAL_AXIS_M))
        self.flush()

    def update(self, state: dict[str, float], *, ee_pose=None) -> None:
        """Log one frame. Callers flush once the frame is complete."""
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
            gripper, [GRIPPER_CLOSED_PERCENT, GRIPPER_OPEN_PERCENT],
            [self.jaw.limit_lower, self.jaw.limit_upper],
        ))
        self.rr.log("transforms/gripper", self.jaw.compute_transform(jaw_angle, clamp=True))

        if ee_pose is not None:
            pose = np.asarray(ee_pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Invalid EE FK pose")
            xyz = pose[:3, 3]
            rpy = tool_rpy(pose[:3, :3])
            # The Placo gripper_frame pose, not the URDF link axes: those drift
            # apart after a local tool-frame edit.
            root = f"/{self.tree.name}"
            self.rr.log(f"{root}/fk_ee_axes", _frame_arrows(self.rr, pose, "fk"))
            if self.ee_tool_offset:
                tip = xyz + pose[:3, :3] @ np.array([0.0, 0.0, self.ee_tool_offset])
                tip_pose = np.eye(4)
                tip_pose[:3, :3] = pose[:3, :3]
                tip_pose[:3, 3] = tip
                self.rr.log(f"{root}/ee_tool", self.rr.Points3D(
                    [tip], radii=_TOOL_POINT_RADIUS_M, colors=[[255, 210, 40]],
                    labels=[f"ee {self.ee_tool_offset * 1000:+.0f} mm"],
                ))
                self.rr.log(f"{root}/ee_tool_line", self.rr.LineStrips3D(
                    [np.stack([xyz, tip])], colors=[[255, 210, 40]], radii=_TOOL_LINE_RADIUS_M,
                ))
                self.rr.log(f"{root}/ee_tool_axes",
                            _frame_arrows(self.rr, tip_pose, "tool", length=_TOOL_AXIS_M))
            for group, names, values in (
                ("xyz", ("x_m", "y_m", "z_m"), xyz),
                ("rpy", ("roll_deg", "pitch_deg", "yaw_deg"), rpy),
            ):
                for name, value in zip(names, values):
                    self.rr.log(f"telemetry/ee/{group}/{name}", self.rr.Scalars(float(value)))

    def log_camera(self, image_bgr) -> None:
        """Show one RGB frame. `image_bgr` is OpenCV BGR."""
        if self.intrinsics is None:
            raise RuntimeError("This twin was opened without a camera view")
        _log_rgb(self.rr, image_bgr)

    def flush(self) -> None:
        """Push the latest log data to the viewer now."""
        _flush(self.rr)

    def show(self, arm) -> None:
        state = arm.get_joints()
        self.update(state, ee_pose=arm.gripper_frame_pose(state))
        self.flush()


def watch_arm(arm, urdf: str | Path, *, intrinsics=None) -> Twin:
    """Open Rerun and redraw the arm after every motion step and goal."""
    twin = Twin(urdf, intrinsics=intrinsics)
    if arm.ee_tool_offset:
        twin.mark_ee_tool(arm.ee_tool_offset)
    arm.track(lambda: twin.show(arm))
    arm.on_target(twin.show_goal_pose)
    return twin


def play_demo(arm, twin: Twin) -> None:
    """Log five synthetic poses, including the jaw opening and closing."""
    if not arm.mock:
        raise RuntimeError("Demo poses are for --mock only")
    sequence = _DEMO_POSES
    for update in sequence:
        arm._backend.send_action({f"{name}.pos": float(value) for name, value in update.items()})
        twin.show(arm)


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
                time.sleep(_LIVE_PERIOD_S)
    except KeyboardInterrupt:
        print("Twin closed.")


class HandEyeView:
    """Twin, tag image, and tag axes while collecting hand-eye samples.

    Samples use `gripper_link` (the tag's parent), not `gripper_frame_link`; the
    fingertips triad only checks that FK matches the mesh.
    """

    def __init__(self, urdf, intrinsics, tag: Tag, *, spawn=True):
        self.twin = Twin(urdf, intrinsics=intrinsics, spawn=spawn)
        self.rr = self.twin.rr
        self.intrinsics = intrinsics
        self.tag = tag
        self.detector = tag_detector(tag)
        root = f"/{self.twin.tree.name}"
        for path in ("camera_axes", "tag_axes", "gripper_link_axes"):
            self.rr.log(f"{root}/{path}", self.rr.CoordinateFrame("base_link"), static=True)

    def update(self, frame, arm, base_T_camera=None):
        state = arm.get_joints()
        tool = arm.gripper_frame_pose(state)
        self.twin.update(state, ee_pose=tool)
        base_T_gripper = arm.gripper_link_pose(state)
        root = f"/{self.twin.tree.name}"
        # Relabel the twin FK triad as "tool" so it is not confused with the tag.
        self.rr.log(f"{root}/fk_ee_axes", _frame_arrows(self.rr, tool, "tool"))
        self.rr.log(f"{root}/gripper_link_axes",
                    _frame_arrows(self.rr, base_T_gripper, "link", length=_LINK_AXIS_M))
        mount = self.tag.gripper_T_tag
        self.rr.log("transforms/tag_mount", self.rr.Transform3D(
            translation=mount[:3, 3], mat3x3=mount[:3, :3],
            parent_frame="gripper_link", child_frame="tag_mount"))
        image = frame.rgb_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(image, self.intrinsics, self.tag,
                                             detector=self.detector)
        except ValueError as exc:
            camera_T_tag = None
            draw_error(image, str(exc), (40, 40, 220))
        else:
            draw_tag_overlay(image, corners, self.tag)
            if base_T_camera is not None:
                seen = base_T_camera @ camera_T_tag
                self.rr.log("transforms/camera_tag", self.rr.Transform3D(
                    translation=seen[:3, 3], mat3x3=seen[:3, :3],
                    parent_frame="base_link", child_frame="camera_tag"))
        if base_T_camera is not None:
            self.rr.log("transforms/station_camera", self.rr.Transform3D(
                translation=base_T_camera[:3, 3], mat3x3=base_T_camera[:3, :3],
                parent_frame="base_link", child_frame="station_camera"))
            self.rr.log(f"{root}/camera_axes",
                        _frame_arrows(self.rr, base_T_camera, "camera", length=_CAMERA_AXIS_M))
        tag_pose = base_T_gripper @ mount
        self.rr.log(f"{root}/tag_axes", _frame_arrows(self.rr, tag_pose, "tag", length=_TAG_AXIS_M))
        self.twin.log_camera(image)
        self.twin.flush()
        return camera_T_tag


def _format_pose(pose, tag: Tag) -> str:
    pose = np.asarray(pose, dtype=float)
    rows = "\n".join("[" + " ".join(f"{value:8.4f}" for value in row) + "]" for row in pose)
    x, y, z = pose[:3, 3]
    return (f"Tag ID: {tag.id}\n"
            "Camera frame: +X right, +Y down, +Z forward\n"
            f"Tag origin (metres): X={x:.4f}, Y={y:.4f}, Z={z:.4f}\n"
            f"camera_T_tag:\n{rows}")


class TagView:
    """Rerun view of one tag in the camera frame. The arm is not involved."""

    def __init__(self, intrinsics, tag: Tag, *, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb

        if tag.size_m <= 0:
            raise ValueError("Tag size must be positive, in metres")
        self.rr = rr
        self.intrinsics = intrinsics
        self.tag = tag
        self.detector = tag_detector(tag)
        rr.init("SO-101 AprilTag", spawn=spawn)
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)
        rr.log("camera/origin",
                _frame_arrows(rr, np.eye(4), "camera", length=_CAMERA_AXIS_M), static=True)
        rr.log(_CAMERA, _pinhole_entity(rr, intrinsics), static=True)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(name="Camera frame (metres)", origin="/camera"),
            rrb.Vertical(
                rrb.Spatial2DView(name="RGB", origin=_CAMERA),
                rrb.TextDocumentView(name="camera_T_tag", origin="/camera/pose_text"),
            ),
        ), auto_views=False))

    def update(self, image_bgr):
        """Return camera_T_tag, or None when the tag is not visible."""
        image = image_bgr.copy()
        try:
            camera_T_tag, corners = find_tag(
                image, self.intrinsics, self.tag, detector=self.detector,
            )
        except ValueError as exc:
            self.rr.log("camera/tag", self.rr.Clear(recursive=True))
            self.rr.log("camera/tag_axes", self.rr.Clear(recursive=False))
            self.rr.log("camera/pose_text", self.rr.TextDocument(f"No tag: {exc}"))
            draw_error(image, str(exc))
            result = None
        else:
            draw_tag_overlay(image, corners, self.tag)
            self.rr.log("camera/tag", self.rr.Transform3D(
                translation=camera_T_tag[:3, 3], mat3x3=camera_T_tag[:3, :3]))
            self.rr.log("camera/tag_axes",
                        _frame_arrows(self.rr, camera_T_tag, "tag", length=_TAG_AXIS_M))
            self.rr.log("camera/pose_text", self.rr.TextDocument(
                _format_pose(camera_T_tag, self.tag)))
            result = camera_T_tag
        _log_rgb(self.rr, image)
        _flush(self.rr)
        return result
