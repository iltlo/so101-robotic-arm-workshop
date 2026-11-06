"""URDF forward kinematics and pose IK. Joints in degrees, TCP in metres.

The chain is the URDF itself: each joint origin is a rigid transform and each
revolute joint is a rotation about its axis. Inverse kinematics is damped
least squares on that chain, with a null-space pull back to the seed so a
short step keeps the arm configuration the recorded solver produced.
"""

from dataclasses import dataclass
from functools import wraps
from threading import RLock
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from so101_utils.model import JOINTS, gripper_rotation_from_rpy, tool_rpy
from so101_utils.transforms import transform

# Feedback can rest slightly past a calibrated stop. Commanded goals stay inside.
LIMIT_MARGIN_DEG = 2.0
# A short translation is solved inside this many degrees of the start. Outside it,
# the five-joint arm can spin a wrist through a long path that does not change the tool point.
NEAR_STEP_DEG = 8.0
# A pitch step swings the tool point about the wrist, so holding that point needs
# a larger wrist step than the translation window allows.
NEAR_ORIENTATION_STEP_DEG = 20.0
# An orientation goal that does not fit in that window is solved again on the full limits.
NEAR_ORIENTATION_DEG = 1.0
# Gripper +X.z must be below this for a downward solve to count.
DOWNWARD_Z = -0.95
# A solved pose counts once its tool point is this close, and orientation this close.
DEFAULT_TOLERANCE_M = 0.015
DEFAULT_TOLERANCE_DEG = 8.0
# A turn that must hold the tool point is accepted only inside this band. A wider
# one lets the solver trade millimetres of the point for orientation, which reads
# as the gripper lurching when a pitch key is pressed.
TURN_TOLERANCE_M = 0.002
# A turn uses damped least squares on the tool point and tilt. World-Z rotation
# is free: the five joints cannot hold every orientation, and the pan axis
# leaves exactly that direction open.
TURN_DAMPING = 0.05
TURN_ROTATION_WEIGHT = 0.6
TURN_MAX_STEP_DEG = 15.0
TURN_ITERATIONS = 80
TURN_SETTLE_M = 0.0005
TURN_SETTLE_DEG = 0.2
_TURN_WEIGHTS = np.diag([1.0, 1.0, 1.0, TURN_ROTATION_WEIGHT,
                         TURN_ROTATION_WEIGHT, TURN_ROTATION_WEIGHT])
# Metres and radians share one residual. A smaller damping tracks the recorded
# joint answers; a larger one stalls short of them.
_IK_DAMPING = 1e-3
# Same order as the recorded solver's rest-pose term. Position-only goals
# leave two degrees free; this keeps wrist roll from drifting on a nudge
# without dragging a reachable pose off the seed.
_NULL_GAIN = 0.001
_IK_INNER_ITERATIONS = 180
_IK_MAX_STEP_RAD = 0.4
# One metre of translation and one radian of rotation, matching the scale the
# recorded solves traded off.
_ORIENTATION_WEIGHT = 1.0
# The jaw is not one of JOINTS. Pin it so IK cannot move the fingers' opening.
_JAW = "gripper"
# A bare point pins roll and pitch, not yaw: the pan carries the tool sideways and
# turns yaw together, so the yaw target follows the heading of the point.
_YAW_ROUNDS = 2
_YAW_TOLERANCE_DEG = 0.05
# Below this radius the heading of the point is undefined and pan buys nothing.
_YAW_RADIUS_M = 0.05


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


class IKError(ValueError):
    """The solver could not reach the goal inside the bounds it was given."""


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    movable: bool


def _vec3(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3)
    values = np.array([float(part) for part in text.split()], dtype=float)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 numbers, got {text!r}")
    return values


def _origin_matrix(element) -> np.ndarray:
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = Rotation.from_euler("xyz", _vec3(element.get("rpy"))).as_matrix()
    matrix[:3, 3] = _vec3(element.get("xyz"))
    return matrix


def _rotvec(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle of a rotation matrix. The near-identity case is the skew vector."""
    cosine = (float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2]) - 1.0) * 0.5
    if cosine > 0.999999:
        return np.array([
            (rotation[2, 1] - rotation[1, 2]) * 0.5,
            (rotation[0, 2] - rotation[2, 0]) * 0.5,
            (rotation[1, 0] - rotation[0, 1]) * 0.5,
        ])
    cosine = min(1.0, max(-1.0, cosine))
    angle = math.acos(cosine)
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        return Rotation.from_matrix(rotation).as_rotvec()
    scale = angle / (2.0 * sine)
    return np.array([
        (rotation[2, 1] - rotation[1, 2]) * scale,
        (rotation[0, 2] - rotation[2, 0]) * scale,
        (rotation[1, 0] - rotation[0, 1]) * scale,
    ])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis. Scalar trig avoids a SciPy call per joint."""
    x, y, z = (float(value) for value in axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return np.array([
        [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
        [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
        [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
    ])


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _load_chain(urdf: str) -> tuple[str, tuple[_Joint, ...]]:
    """Joints in parent-before-child order. Link frames ignore visual meshes."""
    root = ET.parse(urdf).getroot()
    joints = []
    for element in root.findall("joint"):
        kind = element.get("type")
        if kind not in {"revolute", "continuous", "fixed", "prismatic"}:
            continue
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {element.get('name')} is missing a link")
        axis_element = element.find("axis")
        axis = _vec3(None if axis_element is None else axis_element.get("xyz"))
        if kind == "prismatic":
            raise ValueError(f"Prismatic joint {element.get('name')} is not supported")
        movable = kind in {"revolute", "continuous"}
        if movable:
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                raise ValueError(f"Joint {element.get('name')} has no axis")
            axis = axis / norm
        joints.append(_Joint(
            name=element.get("name") or "",
            parent=parent.get("link"),
            child=child.get("link"),
            origin=_origin_matrix(element.find("origin")),
            axis=axis,
            movable=movable,
        ))
    children = {joint.child for joint in joints}
    roots = {joint.parent for joint in joints} - children
    if len(roots) != 1:
        raise ValueError(f"URDF must have one base link, found {sorted(roots)}")
    base = roots.pop()
    ordered: list[_Joint] = []
    pending = joints
    known = {base}
    while pending:
        ready = [joint for joint in pending if joint.parent in known]
        if not ready:
            raise ValueError("URDF joints do not form a tree")
        ordered.extend(ready)
        known.update(joint.child for joint in ready)
        pending = [joint for joint in pending if joint not in ready]
    return base, tuple(ordered)


def _dls(jacobian: np.ndarray, error: np.ndarray, q: np.ndarray, q_rest: np.ndarray) -> np.ndarray:
    """Damped step plus a pull toward `q_rest` in the unused directions."""
    width = jacobian.shape[1]
    gram = jacobian.T @ jacobian + (_IK_DAMPING ** 2) * np.eye(width)
    task = np.linalg.solve(gram, jacobian.T @ error)
    projector = np.eye(width) - np.linalg.solve(gram, jacobian.T @ jacobian)
    step = task + projector @ (_NULL_GAIN * (q_rest - q))
    norm = float(np.linalg.norm(step))
    if norm > _IK_MAX_STEP_RAD:
        step *= _IK_MAX_STEP_RAD / norm
    return step


def _locked(method):
    """Serialize calls that move the shared joint state.

    Status reads and keyboard motion run on different threads. A solve is many
    chain evaluations and must not interleave with a read of the joints.
    """
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return run


class Kinematics:
    def __init__(self, urdf: str, ee_tool_offset: float = 0.0):
        offset = float(ee_tool_offset)
        if not np.isfinite(offset):
            raise ValueError("ee_tool_offset must be finite")
        # Metres along gripper_frame_link +Z. Positive follows the blue axis.
        self.ee_tool_offset = offset
        self._lock = RLock()
        _base, chain = _load_chain(urdf)
        by_name = {joint.name: joint for joint in chain}
        missing = [name for name in (*JOINTS, _JAW) if name not in by_name]
        frames = {joint.child for joint in chain}
        if missing or "gripper_frame_link" not in frames or "gripper_link" not in frames:
            raise ValueError(f"URDF is missing kinematics frames: {missing or 'gripper links'}")
        arm = []
        for name in JOINTS:
            joint = by_name[name]
            if not joint.movable:
                raise ValueError(f"{name} is not a revolute joint")
            arm.append((joint.origin[:3, :3].copy(), joint.origin[:3, 3].copy(), joint.axis.copy()))
        self._arm = tuple(arm)
        fixed = by_name.get("gripper_frame_joint")
        if fixed is None or fixed.parent != "gripper_link" or fixed.movable:
            raise ValueError("URDF is missing the fixed gripper frame")
        self._fixed_rotation = fixed.origin[:3, :3].copy()
        self._fixed_translation = fixed.origin[:3, 3].copy()
        self._ee = np.eye(4)
        self._link = np.eye(4)

    def close(self) -> None:
        return None

    def _remember(self, degrees: np.ndarray):
        """Store gripper_link and gripper_frame_link for `degrees`."""
        radians = np.deg2rad(np.asarray(degrees, dtype=float))
        rotation = np.eye(3)
        translation = np.zeros(3)
        axes = np.empty((len(JOINTS), 3))
        pivots = np.empty((len(JOINTS), 3))
        for index, (origin_rotation, origin_translation, axis) in enumerate(self._arm):
            placed_rotation = rotation @ origin_rotation
            placed_translation = rotation @ origin_translation + translation
            world_axis = placed_rotation @ axis
            axes[index] = world_axis
            pivots[index] = placed_translation
            rotation = placed_rotation @ _rodrigues(axis, float(radians[index]))
            translation = placed_translation
        self._link = _pose(rotation, translation)
        ee_rotation = rotation @ self._fixed_rotation
        ee_translation = rotation @ self._fixed_translation + translation
        self._ee = _pose(ee_rotation, ee_translation)
        return axes, pivots, ee_rotation, ee_translation

    @_locked
    def fk(self, state: dict[str, float]) -> np.ndarray:
        """4x4 pose of gripper_frame_link in the base frame."""
        self._remember([float(state[name]) for name in JOINTS])
        return transform(self._ee)

    def tool_point(self, pose: np.ndarray) -> np.ndarray:
        """Point `ee_tool_offset` along gripper_frame +Z."""
        pose = np.asarray(pose, dtype=float)
        return pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, self.ee_tool_offset])

    @_locked
    def fk_gripper_link(self, state: dict[str, float]) -> np.ndarray:
        """Pose of gripper_link, where the AprilTag is mounted. Not the TCP."""
        self.fk(state)
        return self.gripper_link()

    @_locked
    def gripper_link(self) -> np.ndarray:
        """gripper_link from the model last updated by fk()."""
        return transform(self._link)

    def _jacobian(self, degrees: np.ndarray):
        """Tool-point Jacobian at `degrees`, in joint order. Also the link pose."""
        axes, pivots, ee_rotation, ee_translation = self._remember(degrees)
        point = ee_translation + ee_rotation @ np.array([0.0, 0.0, self.ee_tool_offset])
        linear = np.empty((3, len(JOINTS)))
        for column in range(len(JOINTS)):
            axis = axes[column]
            offset = point - pivots[column]
            linear[0, column] = axis[1] * offset[2] - axis[2] * offset[1]
            linear[1, column] = axis[2] * offset[0] - axis[0] * offset[2]
            linear[2, column] = axis[0] * offset[1] - axis[1] * offset[0]
        return linear, axes.T, point, self._ee

    def _inverse(self, seed_deg, lower_deg, upper_deg, tool_target, rotation,
                 tolerance_m: float, *, lock_orientation: bool) -> np.ndarray:
        """Joint degrees that put the tool point on `tool_target`.

        A locked orientation is a second task on the gripper frame. An unlocked
        one leaves that rotation free, which is how a distant point stays reachable
        on a five-joint arm.
        """
        target = np.asarray(tool_target, dtype=float)
        goal = np.asarray(rotation, dtype=float)
        q = np.deg2rad(np.asarray(seed_deg, dtype=float))
        lower = np.deg2rad(np.asarray(lower_deg, dtype=float))
        upper = np.deg2rad(np.asarray(upper_deg, dtype=float))
        q = np.clip(q, lower, upper)
        rest = q.copy()
        best = q.copy()
        best_cost = np.inf

        tool_offset = np.array([0.0, 0.0, self.ee_tool_offset])
        orient_limit = math.radians(0.2)
        # A step that is already on the point does not need the last fraction of
        # a millimetre. Chasing it spends the wrist on a weak singular direction.
        position_limit = tolerance_m if tolerance_m < 0.005 else 0.002

        def cost_at(radians):
            _axes, _pivots, rotation, translation = self._remember(np.rad2deg(radians))
            point = translation + rotation @ tool_offset
            position_error = float(np.linalg.norm(target - point))
            if not lock_orientation:
                return position_error, position_error, 0.0
            orient = float(np.linalg.norm(_rotvec(goal @ rotation.T)))
            return position_error + orient, position_error, orient

        def residual_at(radians):
            linear, angular, point, pose = self._jacobian(np.rad2deg(radians))
            position = target - point
            if lock_orientation:
                angular_error = _rotvec(goal @ pose[:3, :3].T)
                residual = np.concatenate((position, _ORIENTATION_WEIGHT * angular_error))
                jacobian = np.vstack((linear, _ORIENTATION_WEIGHT * angular))
            else:
                residual = position
                jacobian = linear
            return jacobian, residual

        for _ in range(_IK_INNER_ITERATIONS):
            cost, position_error, orient = cost_at(q)
            if cost < best_cost:
                best_cost = cost
                best = q.copy()
            if position_error <= position_limit and orient <= orient_limit:
                best = q.copy()
                break
            jacobian, residual = residual_at(q)
            step = _dls(jacobian, residual, q, rest)
            accepted = None
            for shrink in range(8):
                trial = np.clip(q + step * (0.5 ** shrink), lower, upper)
                if np.allclose(trial, q, atol=1e-8):
                    break
                trial_cost, _position, _orient = cost_at(trial)
                if trial_cost < cost:
                    accepted = trial
                    break
            if accepted is None:
                break
            q = accepted
        return np.rad2deg(best)

    def _attempt(self, seed, lower, upper, goal, rotation, tolerance_m, *,
                 allow_position_only: bool):
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

    def _pan_following(self, seed, lower, upper, goal, rotation, tolerance_m, *,
                       allow_position_only: bool):
        """Pin roll and pitch, and let the yaw target follow the shoulder pan.

        A five-joint arm cannot hold every orientation while stepping sideways.
        The pan moves the tool and turns yaw together, so the target yaw follows
        the heading of the point instead of being held at the seed.
        """
        roll, pitch, yaw = (float(value) for value in tool_rpy(rotation))
        wanted = np.arctan2(goal[1], goal[0])

        def heading_offset(point) -> float:
            offset = float(np.rad2deg(wanted - np.arctan2(point[1], point[0])))
            return (offset + 180.0) % 360.0 - 180.0

        point = self.tool_point(self.fk(dict(zip(JOINTS, seed))))
        solved = None
        for _ in range(_YAW_ROUNDS):
            yaw += heading_offset(point)
            target = gripper_rotation_from_rpy([roll, pitch, yaw])
            solved = self._attempt(
                seed, lower, upper, goal, target, tolerance_m,
                allow_position_only=allow_position_only)
            point = self.tool_point(self.fk(dict(zip(JOINTS, solved))))
            if abs(heading_offset(point)) <= _YAW_TOLERANCE_DEG:
                break
        return solved

    def _tcp_jacobian(self, angles):
        """Tool-point linear and angular Jacobians, and the pose they came from."""
        linear, angular, point, pose = self._jacobian(np.asarray(angles, dtype=float))
        return linear, angular, point, pose[:3, :3]

    def _turn_ik(self, seed, goal, rotation, lower, upper, tolerance_m):
        """Hold the tool point while the tool turns, or None if it cannot.

        Damped least squares on position plus the tilt the arm can meet. Steps
        stay inside `lower`/`upper`, which the caller narrows to the near window,
        so a blocked turn cannot snap to another arm branch.
        """
        target = np.asarray(goal, dtype=float)
        angles = np.clip(np.asarray(seed, dtype=float), lower, upper)
        best, best_error = angles.copy(), np.inf
        settle_m = min(float(tolerance_m), TURN_SETTLE_M)
        for _ in range(TURN_ITERATIONS):
            linear, angular, point, pose = self._tcp_jacobian(angles)
            offset = target - point
            error = Rotation.from_matrix(rotation @ pose.T).as_rotvec()
            error[2] = 0.0
            weight = float(np.linalg.norm(offset)) + TURN_ROTATION_WEIGHT * float(
                np.linalg.norm(error))
            if weight < best_error:
                best, best_error = angles.copy(), weight
            if (np.linalg.norm(offset) <= settle_m
                    and np.rad2deg(np.linalg.norm(error)) <= TURN_SETTLE_DEG):
                return angles
            jacobian = np.vstack((linear, angular))
            normal = jacobian.T @ _TURN_WEIGHTS @ jacobian
            step = np.linalg.solve(
                normal + TURN_DAMPING ** 2 * np.eye(len(JOINTS)),
                jacobian.T @ _TURN_WEIGHTS @ np.concatenate((offset, error)))
            limit = np.deg2rad(TURN_MAX_STEP_DEG)
            moved = np.clip(angles + np.rad2deg(np.clip(step, -limit, limit)),
                            lower, upper)
            if np.allclose(moved, angles, atol=1e-4):
                break
            angles = moved
        linear, angular, point, pose = self._tcp_jacobian(best)
        error = Rotation.from_matrix(rotation @ pose.T).as_rotvec()
        error[2] = 0.0
        if (np.linalg.norm(target - point) > tolerance_m
                or np.rad2deg(np.linalg.norm(error)) > TURN_SETTLE_DEG):
            return None
        return best

    @_locked
    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = DEFAULT_TOLERANCE_M,
                    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
                    downward: bool = False, rpy=None,
                    follow_yaw: bool = False, local_only: bool = False) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_tool_offset` along gripper_frame_link +Z. `rpy` is
        base-frame XYZ Euler degrees, the angles a pose reports; omit it to keep
        roll and pitch, with yaw following the pan. `downward` points the fingers
        (+X) straight down and keeps the twist about that axis.

        `follow_yaw` gives an rpy goal the same freedom a bare point has: roll and
        pitch are pinned, and world-Z rotation follows the pan, which a five-joint
        arm can do without giving up the point. Such a goal is solved with damped
        least squares, so the tool point does not move while the tool turns.
        `local_only` refuses the full-limit search, so a blocked step raises
        instead of jumping to another arm branch.

        The first attempt stays within `NEAR_STEP_DEG` of `state` so a 5 mm step
        does not spin a joint; an rpy goal gets a wider window because holding
        the point while turning needs more wrist travel. A point outside that
        window, or an orientation off by more than `NEAR_ORIENTATION_DEG`, is
        solved again on the full limits unless `local_only` is set.
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
        # A bare point keeps roll and pitch but lets yaw track the pan, which is
        # what makes a sideways step reachable on this arm. A turn with
        # `follow_yaw` needs the same freedom while it holds the point.
        follow = (goal_rotation is None or follow_yaw) and not downward
        yaw_free = follow and float(np.hypot(goal[0], goal[1])) >= _YAW_RADIUS_M

        def solve(seed_angles, low, high, orient):
            if yaw_free:
                return self._pan_following(
                    seed_angles, low, high, goal, orient, tolerance_m,
                    allow_position_only=allow_position_only)
            return self._attempt(
                seed_angles, low, high, goal, orient, tolerance_m,
                allow_position_only=allow_position_only)

        def search(seed_angles, low, high):
            best = None
            best_error = np.inf
            for orient in orientations:
                candidate = solve(seed_angles, low, high, orient)
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

        near_step = NEAR_ORIENTATION_STEP_DEG if goal_rotation is not None else NEAR_STEP_DEG
        near_lower = np.maximum(lower, initial - near_step)
        near_upper = np.minimum(upper, initial + near_step)
        solved = None
        if np.all(near_lower < near_upper):
            if follow_yaw:
                solved = self._turn_ik(
                    initial, goal, rotation, near_lower, near_upper, tolerance_m)
            else:
                near = solve(initial, near_lower, near_upper, rotation)
                orientation_fits = (goal_rotation is None
                                    or orientation_error_deg(near) <= NEAR_ORIENTATION_DEG)
                if orientation_fits and near_enough(near):
                    solved = near
        if solved is None and local_only:
            raise IKError("Cannot hold the tool point through that turn")
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
