"""Rerun twin of the follower URDF. Driven by joint feedback, not a second model."""

from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from so101_control.kinematics import JOINTS


class Twin:
    """One 3D view plus joint and gripper-pose plots."""

    def __init__(self, urdf: str | Path, *, spawn: bool = True, recording: str | None = None):
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
        rr.init("so101_control", spawn=spawn and recording is None)
        if recording is not None:
            rr.save(recording)

        self.tree.log_urdf_to_recording()
        rr.get_global_data_recording().flush()
        robot_root = f"/{self.tree.name}"
        rr.log(robot_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(f"{robot_root}/ee_pose", rr.CoordinateFrame("fk_ee_pose"),
               rr.TransformAxes3D(0.04), static=True)
        rr.log(f"{robot_root}/hover_goal", rr.CoordinateFrame("hover_goal"), static=True)
        rr.log(f"{robot_root}/hover_goal", rr.Points3D(
            [[0, 0, 0]], radii=0.012, colors=[[255, 70, 70]],
            labels=["target"],
        ), static=True)

        colors = ([255, 94, 111], [255, 188, 92], [91, 197, 233],
                  [169, 130, 255], [79, 207, 148])
        for name, color in zip(JOINTS, colors):
            rr.log(f"telemetry/joints/{name}", rr.SeriesLines(names=name, colors=color), static=True)
        for name, color in zip(("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"),
                               (*colors, [255, 255, 255])):
            rr.log(f"telemetry/ee/{name}", rr.SeriesLines(names=name, colors=color), static=True)

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="SO-101", origin=robot_root,
                    contents=[f"{robot_root}/**", "/tf_static", "/transforms/**"],
                    overrides={f"{robot_root}/collision_geometries": rrb.EntityBehavior(visible=False)},
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Joint angles (degrees)", origin="/telemetry/joints"),
                    rrb.TimeSeriesView(name="EE XYZ (m) / RPY (deg)", origin="/telemetry/ee"),
                ),
            ),
            auto_views=False,
        )
        rr.send_blueprint(blueprint)
        if recording is not None:
            blueprint.save("so101_control", Path(recording).with_suffix(".rbl"))

    def show_hover_goal(self, position) -> None:
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid hover goal")
        self.rr.log("transforms/hover_goal", self.rr.Transform3D(
            translation=point, parent_frame="base_link", child_frame="hover_goal"))

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
            for name, value in zip(("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"), (*xyz, *rpy)):
                self.rr.log(f"telemetry/ee/{name}", self.rr.Scalars(float(value)))

    def show(self, arm) -> None:
        state = arm.get_joints()
        self.update(state, ee_pose=arm._kin.fk(state))


def play_demo(arm, twin: Twin) -> None:
    """Log five synthetic poses, including the jaw opening and closing."""
    if not arm.mock:
        raise RuntimeError("Demo poses are for --mock only")
    sequence = ({}, {"shoulder_pan": 25.0}, {"elbow_flex": -20.0}, {"gripper": 100.0}, {"gripper": 0.0})
    for update in sequence:
        for name, value in update.items():
            arm._backend.positions[f"{name}.pos"] = float(value)
        twin.show(arm)
    twin.rr.get_global_data_recording().flush()


@contextmanager
def _torque_key(enabled: bool):
    """Yield a consume() callable. Only the T key is watched."""
    if not enabled:
        yield lambda: False
        return
    from threading import Lock
    from pynput import keyboard

    lock = Lock()
    pending = False
    held = False

    def on_press(key):
        nonlocal pending, held
        if getattr(key, "char", None) and key.char.lower() == "t" and not held:
            held = True
            with lock:
                pending = True

    def on_release(key):
        nonlocal held
        if getattr(key, "char", None) and key.char.lower() == "t":
            held = False

    def consume() -> bool:
        nonlocal pending
        with lock:
            fired, pending = pending, False
            return fired

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        if not listener.is_alive():
            raise RuntimeError("Keyboard listener did not start (an X11 session is required for T)")
        yield consume
    finally:
        listener.stop()


def follow(arm, twin: Twin, *, relax: bool = False) -> None:
    """Read the live arm until Ctrl-C. With relax=True, T toggles torque."""
    print("Live twin. Ctrl-C exits. This does not move the arm.")
    if relax:
        print("T turns torque off. Support the arm; it can fall.")
    try:
        with _torque_key(relax) as pressed:
            while True:
                if pressed():
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

        self.twin.show(arm)
        base_T_gripper = arm.gripper_link_pose()
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
