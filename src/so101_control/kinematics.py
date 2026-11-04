"""URDF forward kinematics (PyBullet) and pose IK. Joints in degrees, TCP in metres."""

import numpy as np
from scipy.spatial.transform import Rotation

from so101_utils.model import JOINTS, gripper_rotation_from_rpy
from so101_utils.transforms import transform

# Feedback can rest slightly past a calibrated stop. Commanded goals stay inside.
LIMIT_MARGIN_DEG = 2.0
# A short translation is solved inside this many degrees of the start. Outside it,
# the five-joint arm can spin a wrist through a long path that does not change the tool point.
NEAR_STEP_DEG = 8.0
# An orientation goal that does not fit in that window is solved again on the full limits.
NEAR_ORIENTATION_DEG = 1.0
# Gripper +X.z must be below this for a downward solve to count.
DOWNWARD_Z = -0.95
# A solved pose counts once its tool point is this close, and orientation this close.
DEFAULT_TOLERANCE_M = 0.015
DEFAULT_TOLERANCE_DEG = 8.0
# Phosphobot damps every movable joint and then selects IK_SDLS.
_JOINT_DAMPING = 0.001
_IK_ITERATIONS = 8
_IK_MAX_NUM_ITERATIONS = 180
_IK_RESIDUAL = 1e-6
# The jaw is not one of JOINTS. Pin it so IK cannot move the fingers' opening.
_JAW = "gripper"


def _frame_with_fingers(fingers: np.ndarray, twist_hint: np.ndarray) -> np.ndarray:
    side = twist_hint[:, 1] - fingers * float(np.dot(twist_hint[:, 1], fingers))
    if np.linalg.norm(side) < 1e-8:
        side = twist_hint[:, 2] - fingers * float(np.dot(twist_hint[:, 2], fingers))
    side = side / np.linalg.norm(side)
    return np.column_stack((fingers, side, np.cross(fingers, side)))


def _fingers_down(seed_rotation: np.ndarray) -> np.ndarray:
    """Gripper rotation with +X straight down, keeping the seed's twist about that axis."""
    return _frame_with_fingers(np.array([0.0, 0.0, -1.0]), seed_rotation)


def _downward_orientations(seed_rotation: np.ndarray):
    """Straight down, then leans that still count as fingers down (z < -0.95)."""
    base = _fingers_down(seed_rotation)
    yield base
    for tilt_deg in (5.0, 10.0, 15.0):
        tilt = np.deg2rad(tilt_deg)
        for azimuth_deg in range(0, 360, 30):
            azimuth = np.deg2rad(azimuth_deg)
            fingers = np.array([
                np.sin(tilt) * np.cos(azimuth),
                np.sin(tilt) * np.sin(azimuth),
                -np.cos(tilt),
            ])
            yield _frame_with_fingers(fingers, base)


class Kinematics:
    def __init__(self, urdf: str, ee_tool_offset: float = 0.0):
        import pybullet as p

        offset = float(ee_tool_offset)
        if not np.isfinite(offset):
            raise ValueError("ee_tool_offset must be finite")
        # Metres along gripper_frame_link +Z. Positive follows the blue axis.
        self.ee_tool_offset = offset
        self._bullet = p
        self._client = p.connect(p.DIRECT)
        try:
            # Meshes are not part of the frame. Leave the fixed gripper_frame_joint
            # unmerged so its rpy stays the gripper frame in the URDF.
            flags = (p.URDF_MAINTAIN_LINK_ORDER | p.URDF_IGNORE_VISUAL_SHAPES
                     | p.URDF_IGNORE_COLLISION_SHAPES)
            self._body = p.loadURDF(
                urdf, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0],
                useFixedBase=True, flags=flags, physicsClientId=self._client)
            self._joint_index, self._joint_name, self._revolute = {}, {}, []
            self._ee_link = self._gripper_link = None
            for index in range(p.getNumJoints(self._body, physicsClientId=self._client)):
                info = p.getJointInfo(self._body, index, physicsClientId=self._client)
                name = info[1].decode()
                child = info[12].decode()
                self._joint_name[index] = name
                self._joint_index[name] = index
                if info[2] == p.JOINT_REVOLUTE:
                    self._revolute.append(index)
                if child == "gripper_frame_link":
                    self._ee_link = index
                elif child == "gripper_link":
                    self._gripper_link = index
            missing = [name for name in (*JOINTS, _JAW) if name not in self._joint_index]
            if missing or self._ee_link is None or self._gripper_link is None:
                raise ValueError(f"URDF is missing kinematics frames: {missing or 'gripper links'}")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return
        self._client = None
        if self._bullet.isConnected(client):
            self._bullet.disconnect(client)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _set_joints(self, state: dict[str, float]) -> None:
        for name in JOINTS:
            self._bullet.resetJointState(
                self._body, self._joint_index[name], float(np.deg2rad(state[name])),
                physicsClientId=self._client)
        for index in self._revolute:
            if self._joint_name[index] not in JOINTS:
                self._bullet.resetJointState(self._body, index, 0.0, physicsClientId=self._client)

    def _link_pose(self, link: int) -> np.ndarray:
        """World link frame, not the centre of mass. Quaternion is xyzw."""
        state = self._bullet.getLinkState(
            self._body, link, computeForwardKinematics=1, physicsClientId=self._client)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(state[5]).as_matrix()
        pose[:3, 3] = state[4]
        return transform(pose)

    def fk(self, state: dict[str, float]) -> np.ndarray:
        """4x4 pose of gripper_frame_link in the base frame."""
        self._set_joints({name: float(state[name]) for name in JOINTS})
        return self._link_pose(self._ee_link)

    def tool_point(self, pose: np.ndarray) -> np.ndarray:
        """Point `ee_tool_offset` along gripper_frame +Z."""
        pose = np.asarray(pose, dtype=float)
        return pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, self.ee_tool_offset])

    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of gripper_link, where the AprilTag is mounted. Not the TCP."""
        self.fk(state)
        return self.gripper_link()

    def gripper_link(self) -> np.ndarray:
        """gripper_link from the model last updated by fk()."""
        return self._link_pose(self._gripper_link)

    def _ik_bounds(self, lower_deg, upper_deg, rest_deg):
        by_name = {name: (lower_deg[i], upper_deg[i], rest_deg[i]) for i, name in enumerate(JOINTS)}
        lower, upper, ranges, rest = [], [], [], []
        for index in self._revolute:
            name = self._joint_name[index]
            if name in by_name:
                low, high, seed = (float(np.deg2rad(value)) for value in by_name[name])
            else:
                low = high = seed = 0.0
            lower.append(low)
            upper.append(high)
            ranges.append(high - low)
            rest.append(seed)
        return lower, upper, ranges, rest

    def _arm_degrees(self, positions) -> np.ndarray:
        positions = list(positions)
        if len(positions) == len(self._revolute):
            by_index = dict(zip(self._revolute, positions))
        elif len(positions) == len(self._joint_index):
            by_index = dict(enumerate(positions))
        else:
            raise RuntimeError(
                f"PyBullet IK returned {len(positions)} joints, expected {len(self._revolute)}")
        return np.rad2deg([by_index[self._joint_index[name]] for name in JOINTS])

    def _inverse(self, seed_deg, lower_deg, upper_deg, tool_target, rotation,
                 tolerance_m: float, *, lock_orientation: bool) -> np.ndarray:
        """PyBullet IK for the link origin that puts the tool point on `tool_target`.

        A locked orientation is the gripper quaternion. Without it, the offset
        direction is taken from the pose each iteration just reached.
        """
        tool_target = np.asarray(tool_target, dtype=float)
        guess = np.array(rotation, dtype=float)
        damping = [_JOINT_DAMPING] * len(self._revolute)
        seed = np.asarray(seed_deg, dtype=float)
        current = seed.copy()
        best = seed.copy()
        best_error = np.inf
        for _ in range(_IK_ITERATIONS):
            self._set_joints(dict(zip(JOINTS, current)))
            link_target = tool_target - guess @ np.array([0.0, 0.0, self.ee_tool_offset])
            # Rest follows the joints this walk has reached, after starting at the seed,
            # so a long move is not pulled back to the start forever.
            lower, upper, ranges, rest = self._ik_bounds(lower_deg, upper_deg, current)
            solve = dict(
                lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                maxNumIterations=_IK_MAX_NUM_ITERATIONS, residualThreshold=_IK_RESIDUAL,
                physicsClientId=self._client)
            # Same branch as phosphobot: damping selects the SDLS solver.
            if damping is not None:
                solve["jointDamping"] = damping
                solve["solver"] = self._bullet.IK_SDLS
            target = link_target.tolist()
            if lock_orientation:
                positions = self._bullet.calculateInverseKinematics(
                    self._body, self._ee_link, target, Rotation.from_matrix(guess).as_quat().tolist(),
                    **solve)
            else:
                positions = self._bullet.calculateInverseKinematics(
                    self._body, self._ee_link, target, **solve)
            current = np.clip(self._arm_degrees(positions), lower_deg, upper_deg)
            self._set_joints(dict(zip(JOINTS, current)))
            pose = self._link_pose(self._ee_link)
            if not lock_orientation:
                guess = pose[:3, :3]
            error = float(np.linalg.norm(self.tool_point(pose) - tool_target))
            if error < best_error:
                best_error = error
                best = current.copy()
            else:
                break
            if error <= tolerance_m:
                break
        return best

    def _attempt(self, seed, lower, upper, goal, rotation, tolerance_m, *, allow_position_only: bool):
        """Orientation first. A position goal with no rpy may then drop it to reach."""
        solved = self._inverse(
            seed, lower, upper, goal, rotation, tolerance_m, lock_orientation=True)
        if allow_position_only:
            posed = dict(zip(JOINTS, solved))
            if np.linalg.norm(self.tool_point(self.fk(posed)) - goal) > tolerance_m:
                alt = self._inverse(
                    seed, lower, upper, goal, rotation, tolerance_m, lock_orientation=False)
                if np.linalg.norm(self.tool_point(self.fk(dict(zip(JOINTS, alt)))) - goal) \
                        < np.linalg.norm(self.tool_point(self.fk(posed)) - goal):
                    solved = alt
        return solved

    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = DEFAULT_TOLERANCE_M,
                    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
                    downward: bool = False, rpy=None) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_tool_offset` along gripper_frame_link +Z. `rpy` is
        base-frame XYZ Euler degrees, the angles a pose reports; omit it to keep
        the start orientation. `downward` points the fingers (+X) straight down
        and keeps the twist about that axis.

        The first attempt stays within `NEAR_STEP_DEG` of `state` so a 5 mm step
        does not spin a joint. A point outside that window, or an orientation off
        by more than `NEAR_ORIENTATION_DEG`, is solved again on the full limits.
        """
        goal = np.asarray(target, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Invalid TCP target")
        if downward and rpy is not None:
            raise ValueError("downward replaces an rpy target")
        goal_rotation = None
        if rpy is not None:
            angles = np.asarray(rpy, dtype=float)
            if angles.shape != (3,) or not np.isfinite(angles).all():
                raise ValueError("Invalid RPY target")
            goal_rotation = gripper_rotation_from_rpy(angles)
        if set(limits) != set(JOINTS):
            raise ValueError("Limits required for all arm joints")
        lower = np.array([limits[name][0] for name in JOINTS], dtype=float)
        upper = np.array([limits[name][1] for name in JOINTS], dtype=float)
        initial = np.array([state[name] for name in JOINTS], dtype=float)
        if not np.isfinite(initial).all() or not np.all(lower - LIMIT_MARGIN_DEG <= initial) \
                or not np.all(initial <= upper + LIMIT_MARGIN_DEG):
            raise ValueError("Starting pose outside joint limits")
        initial = np.clip(initial, lower, upper)

        keep_rotation = self.fk(dict(zip(JOINTS, initial)))[:3, :3]
        if downward:
            rotation = _fingers_down(keep_rotation)
        elif goal_rotation is not None:
            rotation = goal_rotation
        else:
            rotation = keep_rotation
        # An rpy or downward goal owns the orientation. A bare point only prefers it,
        # so a far target can still be reached.
        allow_position_only = goal_rotation is None and not downward

        def position_error(angles) -> float:
            return float(np.linalg.norm(self.tool_point(self.fk(dict(zip(JOINTS, angles)))) - goal))

        def orientation_error_deg(angles) -> float:
            pose = self.fk(dict(zip(JOINTS, angles)))
            delta = Rotation.from_matrix(goal_rotation.T @ pose[:3, :3]).magnitude()
            return float(np.rad2deg(delta))

        def near_enough(angles) -> bool:
            if position_error(angles) > tolerance_m:
                return False
            if goal_rotation is not None and orientation_error_deg(angles) > tolerance_deg:
                return False
            if downward and self.fk(dict(zip(JOINTS, angles)))[2, 0] >= DOWNWARD_Z:
                return False
            return True

        # Straight down first. A lean still has +X.z < -0.95 when that pose cannot
        # put the tool on the point; the five-joint arm has no exact-down solution there.
        orientations = list(_downward_orientations(keep_rotation)) if downward else [rotation]

        def search(seed_angles, low, high):
            best = None
            best_error = np.inf
            for orient in orientations:
                candidate = self._attempt(
                    seed_angles, low, high, goal, orient, tolerance_m,
                    allow_position_only=allow_position_only)
                error = position_error(candidate)
                if error < best_error:
                    best = candidate
                    best_error = error
                if error <= tolerance_m and near_enough(candidate):
                    return candidate
            # The full-limit path accepts a point the fingers cannot aim down,
            # matching the previous solver: only the near window demands +X.z.
            if downward and best_error > tolerance_m:
                alt = self._inverse(
                    seed_angles, low, high, goal, orientations[0], tolerance_m,
                    lock_orientation=False)
                if position_error(alt) < best_error:
                    best = alt
            return best

        near_lower = np.maximum(lower, initial - NEAR_STEP_DEG)
        near_upper = np.minimum(upper, initial + NEAR_STEP_DEG)
        solved = None
        if np.all(near_lower < near_upper):
            near = self._attempt(
                initial, near_lower, near_upper, goal, rotation, tolerance_m,
                allow_position_only=allow_position_only)
            orientation_fits = goal_rotation is None or orientation_error_deg(near) <= NEAR_ORIENTATION_DEG
            if orientation_fits and near_enough(near):
                solved = near
        if solved is None:
            result = search(initial, lower, upper)
            # A folded or mid-move seed can sit in a local minimum that the home
            # joints escape, so retry from home before giving up.
            if position_error(result) > tolerance_m:
                home = np.clip(np.zeros_like(initial), lower, upper)
                alt = search(home, lower, upper)
                if position_error(alt) < position_error(result):
                    result = alt
            if position_error(result) > tolerance_m:
                if downward:
                    raise ValueError(
                        "IK cannot reach that point with the fingers pointing down")
                raise ValueError("IK did not reach the point within joint limits")
            if goal_rotation is not None and orientation_error_deg(result) > tolerance_deg:
                raise ValueError("IK did not match the orientation within joint limits")
            solved = result
        return {name: float(value) for name, value in zip(JOINTS, solved)}
