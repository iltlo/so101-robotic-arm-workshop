"""Read-only articulated SO-101 Rerun viewer, driven by measured joint feedback.

Uses Rerun's own URDF parser for both geometry and named-frame transforms;
there is no guessed entity path or second visualization-only URDF.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import JOINTS


class Twin:
    """Plot joint feedback and FK while animating the single follower URDF."""

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

        # The same UrdfTree imports the meshes, visual origins and static links.
        # Dynamic joints below replace the named parent/child frame relationship.
        self.tree.log_urdf_to_recording()
        rr.get_global_data_recording().flush()
        robot_root = f"/{self.tree.name}"
        rr.log(robot_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
         # The URDF importer uses named frames, not implicit entity-path frames.
         # Draw a native coordinate-frame gizmo in its own named FK frame.
        rr.log(f"{robot_root}/ee_pose", rr.CoordinateFrame("fk_ee_pose"),
             rr.TransformAxes3D(0.04), static=True)
        rr.log(f"{robot_root}/hover_goal", rr.CoordinateFrame("hover_goal"), static=True)
        rr.log(f"{robot_root}/hover_goal", rr.Points3D([[0, 0, 0]], radii=0.012,
                                                     colors=[[255, 70, 70]],
                                                     labels=["candidate TCP (preview only)"]), static=True)

        colors = ([255, 94, 111], [255, 188, 92], [91, 197, 233],
                  [169, 130, 255], [79, 207, 148])
        for name, color in zip(JOINTS, colors):
            rr.log(f"telemetry/joints/{name}", rr.SeriesLines(names=name, colors=color), static=True)
        for name, color in zip(("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"),
                               (*colors, [255, 255, 255])):
            rr.log(f"telemetry/ee/{name}", rr.SeriesLines(names=name, colors=color), static=True)

        # Explicit blueprint prevents Rerun's auto-layout from making a grid for
        # every scalar. The FK pose lives under the robot, not telemetry.
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="SO-101", origin=robot_root,
                                  contents=[f"{robot_root}/**", "/tf_static", "/transforms/**"],
                                  overrides={f"{robot_root}/collision_geometries":
                                             rrb.EntityBehavior(visible=False)}),
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

    def update(self, state: dict[str, float], *, ee_pose=None):
        self.tick += 1
        self.rr.set_time("tick", sequence=self.tick)
        for name, joint in self.joints.items():
            angle = float(state[name])
            if not np.isfinite(angle):
                raise ValueError(f"Invalid joint feedback {name}")
            # compute_transform includes URDF origin AND named frame IDs.
            # One entity per frame relationship: do not let updates for five
            # joints on one timeline tick compete for the same entity's pose.
            self.rr.log(f"transforms/{name}", joint.compute_transform(float(np.deg2rad(angle)), clamp=False))
            self.rr.log(f"telemetry/joints/{name}", self.rr.Scalars(angle))

        # LeRobot reports 0=closed, 100=open. The URDF does not define a
        # servo-to-jaw conversion: use its joint limits only as a visual proxy.
        # This value must NOT be used to estimate contact, grasp width or FK.
        gripper = float(state["gripper"])
        if not np.isfinite(gripper):
            raise ValueError("Invalid gripper feedback")
        jaw_angle = float(np.interp(np.clip(gripper, 0.0, 100.0),
                                    [0.0, 100.0], [self.jaw.limit_lower, self.jaw.limit_upper]))
        self.rr.log("transforms/gripper", self.jaw.compute_transform(jaw_angle, clamp=True))

        if ee_pose is not None:
            pose = np.asarray(ee_pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Invalid EE FK pose")
            xyz = pose[:3, 3]
            rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
            # base_T_gripper_frame_link from FK: connect the visualization's
            # named child frame to the URDF importer's named base_link frame.
            self.rr.log("transforms/fk_ee_pose", self.rr.Transform3D(
                translation=xyz, mat3x3=pose[:3, :3],
                parent_frame="base_link", child_frame="fk_ee_pose",
            ))
            for name, value in zip(("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"), (*xyz, *rpy)):
                self.rr.log(f"telemetry/ee/{name}", self.rr.Scalars(float(value)))