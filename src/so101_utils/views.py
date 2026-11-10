"""The Rerun views: arm twin, hand-eye, tag, and chessboard. Shared layout and helpers."""

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
from so101_utils.urdf_view import UrdfScene

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
# The plots follow the newest sample at their right edge. Without a window the
# view is sized to the whole session and the curve appears to start midway.
_PLOT_WINDOW_TICKS = 400
_PLOT_WINDOW_S = 60.0

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

    # A raw 1280×720 frame is about 3 MB. JPEG keeps the live camera from
    # pushing the arm meshes and the plot history out of the viewer.
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80],
    )
    if not ok:
        rr.log(_CAMERA, rr.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        return
    rr.log(_CAMERA, rr.EncodedImage(contents=bytes(encoded), media_type="image/jpeg"))


def _flush(rr) -> None:
    rr.get_global_data_recording().flush(timeout_sec=0)


def _plot_ranges(rrb):
    """Right-anchored windows for the sequence tick and wall-clock timelines."""
    return [
        rrb.VisibleTimeRange(
            "tick",
            start=rrb.TimeRangeBoundary.cursor_relative(seq=-_PLOT_WINDOW_TICKS),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        ),
        rrb.VisibleTimeRange(
            "log_time",
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-_PLOT_WINDOW_S),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        ),
    ]


def _plot(rrb, name, origin):
    return rrb.TimeSeriesView(name=name, origin=origin, time_ranges=_plot_ranges(rrb))


def _arm_blueprint(rrb, root, *, intrinsics):
    """3D arm and joint angles left; camera and EE poses right."""
    arm_view = rrb.Spatial3DView(
        name="SO-101", origin=root,
        contents=[f"{root}/**"],
        background=rrb.Background(
            kind=rrb.BackgroundKind.SolidColor, color=(24, 24, 24, 255),
        ),
    )
    joint_plot = _plot(rrb, "Joint angles (degrees)", "/telemetry/joints")
    pose_plots = rrb.Horizontal(
        _plot(rrb, "EE XYZ (m)", "/telemetry/ee/xyz"),
        _plot(rrb, "EE RPY (deg)", "/telemetry/ee/rpy"),
    )
    if intrinsics is None:
        layout = rrb.Horizontal(arm_view, rrb.Vertical(joint_plot, pose_plots))
    else:
        camera = rrb.Spatial2DView(
            name="Camera", origin=_CAMERA,
            background=rrb.Background(
                kind=rrb.BackgroundKind.SolidColor, color=(24, 24, 24, 255),
            ),
        )
        layout = rrb.Horizontal(
            rrb.Vertical(arm_view, joint_plot),
            rrb.Vertical(camera, pose_plots),
        )
    return rrb.Blueprint(layout, auto_views=False, collapse_panels=True)


class Twin:
    """One 3D view plus joint-angle and EE pose plots."""

    def __init__(self, urdf: str | Path, *, spawn: bool = True, recording: str | None = None,
                 intrinsics=None, grpc_url: str | None = None):
        import rerun as rr
        import rerun.blueprint as rrb

        path = Path(urdf).resolve()
        if path.name != "so101_new_calib.urdf" or not path.is_file():
            raise ValueError("Twin requires the downloaded so101_new_calib.urdf")
        self.rr = rr
        self.intrinsics = intrinsics
        self.tree = UrdfScene.from_file_path(path)
        self.joints = {name: self.tree.get_joint_by_name(name) for name in JOINTS}
        if any(joint is None or joint.joint_type != "revolute" for joint in self.joints.values()):
            raise ValueError("Follower URDF missing an arm revolute joint")
        self.jaw = self.tree.get_joint_by_name("gripper")
        if self.jaw is None or self.jaw.joint_type != "revolute":
            raise ValueError("Follower URDF missing gripper jaw joint")
        self.tick = 0
        self.ee_tool_offset = None
        self.alive = True
        rr.init("so101_control", spawn=spawn and recording is None and grpc_url is None)
        if grpc_url:
            rr.connect_grpc(grpc_url)
        if recording is not None:
            rr.save(recording)

        self.tree.log_to_recording(rr)
        rr.get_global_data_recording().flush()
        # The scene root is the base frame. Arrow geometry is already in base
        # metres, so it sits next to base_link without a further transform.
        robot_root = f"/{self.tree.name}"
        rr.log(robot_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

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

    def close(self) -> None:
        """Stop sending to the viewer. Later draws become no-ops."""
        if not self.alive:
            return
        self.alive = False
        try:
            self.rr.disconnect()
        except Exception:
            pass

    def show_goal_pose(self, pose) -> None:
        """Draw the tool pose `move_to` solved.

        Rotation is gripper_frame_link (X out the fingers). Translation is the
        tool point, offset along that frame's +Z when `ee_tool_offset` is set.
        """
        if not self.alive:
            return
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

    def update(self, state: dict[str, float], *, ee_pose=None, flexion_axis=None) -> None:
        """Log one frame. Callers flush once the frame is complete."""
        if not self.alive:
            return
        self.tick += 1
        self.rr.set_time("tick", sequence=self.tick)
        for name, joint in self.joints.items():
            angle = float(state[name])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid joint feedback {name}")
            self.rr.log(joint.entity, joint.compute_transform(float(np.deg2rad(angle)), clamp=False))
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
        self.rr.log(self.jaw.entity, self.jaw.compute_transform(jaw_angle, clamp=True))

        if ee_pose is not None:
            pose = np.asarray(ee_pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Invalid EE FK pose")
            xyz = pose[:3, 3]
            rpy = tool_rpy(pose[:3, :3], flexion_axis)
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
        if not self.alive:
            return
        if self.intrinsics is None:
            raise RuntimeError("This twin was opened without a camera view")
        _log_rgb(self.rr, image_bgr)

    def flush(self) -> None:
        """Push the latest log data to the viewer now."""
        if self.alive:
            _flush(self.rr)

    def show(self, arm) -> None:
        state = arm.get_joints()
        self.update(state, ee_pose=arm.gripper_frame_pose(state), flexion_axis=arm.nod_axis(state))
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


def log_hand_eye(twin, image_bgr, arm, tag, base_T_camera=None, detector=None):
    """Draw the arm, the tag, and the camera on an existing twin.

    Returns camera_T_tag, or None when the tag is not a single clear detection.
    The JPEG click view is separate and is not modified here.
    """
    if not getattr(twin, "alive", True):
        return None
    state = arm.get_joints()
    tool = arm.gripper_frame_pose(state)
    twin.update(state, ee_pose=tool)
    base_T_gripper = arm.gripper_link_pose(state)
    root = f"/{twin.tree.name}"
    rr = twin.rr
    rr.log(f"{root}/fk_ee_axes", _frame_arrows(rr, tool, "tool"))
    rr.log(f"{root}/gripper_link_axes",
           _frame_arrows(rr, base_T_gripper, "link", length=_LINK_AXIS_M))
    mount = tag.gripper_T_tag
    rr.log(f"{twin.tree.link_entity('gripper_link')}/tag_mount", rr.Transform3D(
        translation=mount[:3, 3], mat3x3=mount[:3, :3]))
    image = np.ascontiguousarray(image_bgr).copy()
    try:
        camera_T_tag, corners = find_tag(image, twin.intrinsics, tag, detector=detector)
    except ValueError as exc:
        camera_T_tag = None
        draw_error(image, str(exc), (40, 40, 220))
    else:
        draw_tag_overlay(image, corners, tag)
        if base_T_camera is not None:
            seen = base_T_camera @ camera_T_tag
            base = twin.tree.link_entity("base_link")
            rr.log(f"{base}/camera_tag", rr.Transform3D(
                translation=seen[:3, 3], mat3x3=seen[:3, :3]))
    if base_T_camera is not None:
        base = twin.tree.link_entity("base_link")
        rr.log(f"{base}/station_camera", rr.Transform3D(
            translation=base_T_camera[:3, 3], mat3x3=base_T_camera[:3, :3]))
        rr.log(f"{root}/camera_axes",
               _frame_arrows(rr, base_T_camera, "camera", length=_CAMERA_AXIS_M))
    tag_pose = base_T_gripper @ mount
    rr.log(f"{root}/tag_axes", _frame_arrows(rr, tag_pose, "tag", length=_TAG_AXIS_M))
    twin.log_camera(image)
    twin.flush()
    return camera_T_tag


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

    def update(self, frame, arm, base_T_camera=None):
        return log_hand_eye(
            self.twin, frame.rgb_bgr, arm, self.tag, base_T_camera, detector=self.detector,
        )


def _format_pose(pose, tag: Tag) -> str:
    pose = np.asarray(pose, dtype=float)
    rows = "\n".join("[" + " ".join(f"{value:8.4f}" for value in row) + "]" for row in pose)
    x, y, z = pose[:3, 3]
    return (f"Tag ID: {tag.id}\n"
            "Camera frame: +X right, +Y down, +Z forward\n"
            f"Tag origin (metres): X={x:.4f}, Y={y:.4f}, Z={z:.4f}\n"
            f"camera_T_tag:\n{rows}")


class ChessboardView:
    """One RGB image for chessboard calibration. The arm is not involved."""

    def __init__(self, *, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init("SO-101 intrinsics", spawn=spawn)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Spatial2DView(name="Chessboard", origin=_CAMERA),
            auto_views=False,
        ))

    def update(self, image_bgr) -> None:
        _log_rgb(self.rr, image_bgr)
        _flush(self.rr)


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
