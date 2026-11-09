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
# The cone below the base. A downward goal constrains the finger direction to
# this cone, not to one exact orientation: near the workspace edge no exact
# pose fits, but a lean inside the cone still points the fingers down.
_DOWNWARD_TILT = math.acos(-DOWNWARD_Z) - math.radians(0.2)
# Pull the fingers back inside the cone strongly enough that the solver does not
# settle just outside it, where the point is within tolerance but not downward.
_DOWNWARD_TILT_WEIGHT = 2.0
_DOWN = np.array([0.0, 0.0, -1.0])
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
_TURN_SQRT_WEIGHTS = np.sqrt([1.0, 1.0, 1.0, TURN_ROTATION_WEIGHT,
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
# Even a loose caller tolerance wants a precise pose: the next nudge starts from
# millimetres of residual, not from the full allowance.
_PRECISE_GOAL_M = 0.002
# Orientation settles well inside a pose tolerance before a solve is accepted.
_POSE_SETTLE_DEG = 0.2
# A bare point pins roll and pitch, not yaw: the pan carries the tool sideways and
# turns yaw together, so the yaw target follows the heading of the point.
_YAW_ROUNDS = 2
_YAW_TOLERANCE_DEG = 0.05
# Below this radius the heading of the point is undefined and pan buys nothing.
_YAW_RADIUS_M = 0.05
# A full-limit solve can stall in a local minimum far from the seed. Retry from
# a fixed low-discrepancy set of seeds, so the same goal always gives the same
# answer and a reachable pose is not refused because of where the arm started.
_RESTART_COUNT = 32
_RESTART_BASES = (2, 3, 5, 7, 11)


def _halton(count: int, bases) -> np.ndarray:
    """Low-discrepancy points in the unit cube, one column per base."""
    values = np.empty((count, len(bases)))
    for column, base in enumerate(bases):
        for index in range(count):
            number, fraction, value = index + 1, 1.0, 0.0
            while number:
                fraction /= base
                value += fraction * (number % base)
                number //= base
            values[index, column] = value
    return values


_RESTART_FRACTIONS = _halton(_RESTART_COUNT, _RESTART_BASES)
# A folded arm can reach points behind or below the base with the joints near
# their stops. Uniform restarts rarely land in those narrow basins, so the last
# fallback seeds the corners of the joint range instead.
_FOLDED_PAN_SEEDS = 6
_FOLDED_EDGE_DEG = 5.0


def _folded_seeds(lower: np.ndarray, upper: np.ndarray):
    """Seeds with the lift and elbow near their stops, where folded poses live."""
    pans = np.linspace(lower[0] + _FOLDED_EDGE_DEG, upper[0] - _FOLDED_EDGE_DEG,
                       _FOLDED_PAN_SEEDS)
    for pan in pans:
        for lift in (lower[1] + _FOLDED_EDGE_DEG, upper[1] - _FOLDED_EDGE_DEG):
            for elbow in (lower[2] + _FOLDED_EDGE_DEG, upper[2] - _FOLDED_EDGE_DEG):
                for flex in (0.0, lower[3] + _FOLDED_EDGE_DEG,
                             upper[3] - _FOLDED_EDGE_DEG):
                    for roll in (0.0, -120.0, 120.0):
                        yield np.clip(np.array([pan, lift, elbow, flex, roll]),
                                      lower, upper)


def _frame_with_fingers(fingers: np.ndarray, twist_hint: np.ndarray) -> np.ndarray:
    side = twist_hint[:, 1] - fingers * float(np.dot(twist_hint[:, 1], fingers))
    if np.linalg.norm(side) < 1e-8:
        side = twist_hint[:, 2] - fingers * float(np.dot(twist_hint[:, 2], fingers))
    side = side / np.linalg.norm(side)
    return np.column_stack((fingers, side, np.cross(fingers, side)))


def _fingers_down(seed_rotation: np.ndarray) -> np.ndarray:
    """Gripper rotation with +X straight down, keeping the seed's twist about that axis."""
    return _frame_with_fingers(np.array([0.0, 0.0, -1.0]), seed_rotation)


def _down_tilt(rotation: np.ndarray) -> tuple[float, np.ndarray]:
    """Angle between gripper +X and straight down, and the direction that tilts it."""
    fingers = rotation[:, 0]
    axis = np.cross(fingers, _DOWN)
    sine = float(np.linalg.norm(axis))
    angle = math.atan2(sine, float(np.dot(fingers, _DOWN)))
    if sine > 1e-12:
        return angle, axis / sine
    # Fingers straight up have no tilt axis; any horizontal one leads down.
    return angle, np.array([1.0, 0.0, 0.0])


class IKError(ValueError):
    """The solver could not reach the goal inside the bounds it was given."""


@dataclass(frozen=True, eq=False)
class _Solution:
    """Joint degrees and the pose they produce, so callers never solve twice."""
    angles: np.ndarray
    point: np.ndarray
    rotation: np.ndarray
    position_error: float


@dataclass(frozen=True, eq=False)
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
    """Joints in parent-before-child order. Link frames ignore visual meshes.

    Prismatic joints are part of the chain but stay at zero displacement: the
    arm API has no command for them, and the SO-101 jaw may be modelled either
    way. Unsupported joint types would silently break the chain, so they fail.
    """
    root = ET.parse(urdf).getroot()
    joints = []
    names = set()
    for element in root.findall("joint"):
        name = element.get("name") or ""
        if not name:
            raise ValueError("URDF joint is missing a name")
        if name in names:
            raise ValueError(f"Duplicate joint {name}")
        names.add(name)
        kind = element.get("type")
        if kind not in {"revolute", "continuous", "fixed", "prismatic"}:
            raise ValueError(f"Joint {name} has unsupported type {kind!r}")
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {name} is missing a link")
        movable = kind in {"revolute", "continuous"}
        axis_element = element.find("axis")
        # URDF defaults a movable joint with no <axis> to the X axis.
        axis_text = None
        if axis_element is not None:
            axis_text = axis_element.get("xyz")
        elif movable:
            axis_text = "1 0 0"
        axis = _vec3(axis_text)
        if movable:
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                raise ValueError(f"Joint {name} has no axis")
            axis = axis / norm
        joints.append(_Joint(
            name=name,
            parent=parent.get("link"),
            child=child.get("link"),
            origin=_origin_matrix(element.find("origin")),
            axis=axis,
            movable=movable,
        ))
    children = {joint.child for joint in joints}
    if len(children) != len(joints):
        raise ValueError("URDF links have more than one parent joint")
    roots = {joint.parent for joint in joints} - children
    if len(roots) != 1:
        raise ValueError(f"URDF must have one base link, found {sorted(roots)}")
    base = roots.pop()
    ordered: list[_Joint] = []
    pending = list(joints)
    known = {base}
    while pending:
        ready = [joint for joint in pending if joint.parent in known]
        if not ready:
            raise ValueError("URDF joints do not form a tree")
        ordered.extend(ready)
        known.update(joint.child for joint in ready)
        ready_names = {joint.name for joint in ready}
        pending = [joint for joint in pending if joint.name not in ready_names]
    return base, tuple(ordered)


def _damped_step(jacobian: np.ndarray, error: np.ndarray, damping: float):
    """One damped-least-squares step and the coupling it leaves between joints."""
    width = jacobian.shape[1]
    gram = jacobian.T @ jacobian + (damping ** 2) * np.eye(width)
    inverse = np.linalg.inv(gram)
    return inverse @ (jacobian.T @ error), inverse @ (jacobian.T @ jacobian)


def _dls(jacobian: np.ndarray, error: np.ndarray, q: np.ndarray, q_rest: np.ndarray) -> np.ndarray:
    """Damped step plus a pull toward `q_rest` in the unused directions."""
    step, coupling = _damped_step(jacobian, error, _IK_DAMPING)
    projector = np.eye(len(q)) - coupling
    step = step + projector @ (_NULL_GAIN * (q_rest - q))
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
        base, chain = _load_chain(urdf)
        by_name = {joint.name: joint for joint in chain}
        missing = [name for name in JOINTS if name not in by_name]
        for name in JOINTS:
            if name in by_name and not by_name[name].movable:
                raise ValueError(f"{name} is not a revolute joint")
        if missing:
            raise ValueError(f"URDF is missing kinematics frames: {missing}")
        # Walk the actual tool chain instead of assuming the five arm joints are
        # adjacent. Fixed spacers and a prismatic jaw must not change the frame.
        child_to_joint = {joint.child: joint for joint in chain}
        if "gripper_frame_link" not in child_to_joint:
            raise ValueError("URDF is missing kinematics frames: gripper_frame_link")
        tool_chain = []
        link = "gripper_frame_link"
        while link != base:
            joint = child_to_joint.get(link)
            if joint is None:
                raise ValueError(f"Link {link} is not connected to the URDF base")
            tool_chain.append(joint)
            link = joint.parent
        # A world link may sit above base_link. The arm API is in the robot base
        # frame, so stop at base_link when it is on the path to the tool.
        for index, joint in enumerate(tool_chain):
            if joint.parent == "base_link":
                tool_chain = tool_chain[:index + 1]
                break
        tool_chain.reverse()
        tool_names = {joint.name for joint in tool_chain}
        missing = [name for name in JOINTS if name not in tool_names]
        if missing:
            raise ValueError(f"URDF is missing kinematics frames: {missing}")
        if "gripper_link" not in {joint.child for joint in tool_chain}:
            raise ValueError("URDF is missing kinematics frames: gripper_link")
        if tool_chain[-1].child != "gripper_frame_link":
            raise ValueError("URDF gripper frame is not the end of the arm chain")
        self._chain = tuple(tool_chain)
        # Unfolded length from the base to the tool frame. No rotation can put the
        # tool point farther than this, so a farther goal is outside reach and a
        # search for it would only burn restarts.
        self._path_length = float(sum(np.linalg.norm(joint.origin[:3, 3])
                                      for joint in tool_chain))
        self._columns = {name: index for index, name in enumerate(JOINTS)}
        self._chain_slots = tuple(self._columns.get(joint.name, -1) for joint in tool_chain)
        self._ee = np.eye(4)
        self._link = np.eye(4)

    def close(self) -> None:
        return None

    def _remember_radians(self, radians: np.ndarray):
        """Store gripper_link and gripper_frame_link for `radians`, in joint order."""
        values = np.asarray(radians, dtype=float)
        rotation = np.eye(3)
        translation = np.zeros(3)
        axes = np.zeros((len(JOINTS), 3))
        pivots = np.zeros((len(JOINTS), 3))
        for joint, column in zip(self._chain, self._chain_slots):
            placed_rotation = rotation @ joint.origin[:3, :3]
            placed_translation = rotation @ joint.origin[:3, 3] + translation
            if joint.movable:
                world_axis = placed_rotation @ joint.axis
                if column >= 0:
                    axes[column] = world_axis
                    pivots[column] = placed_translation
                angle = values[column] if column >= 0 else 0.0
                rotation = placed_rotation @ _rodrigues(joint.axis, angle)
            else:
                rotation = placed_rotation
            translation = placed_translation
            if joint.child == "gripper_link":
                self._link = _pose(rotation, translation)
        self._ee = _pose(rotation, translation)
        return axes, pivots, rotation, translation

    def _remember(self, degrees: np.ndarray):
        """Store gripper_link and gripper_frame_link for `degrees`."""
        return self._remember_radians(np.deg2rad(np.asarray(degrees, dtype=float)))

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

    def _jacobian_radians(self, radians: np.ndarray):
        """Tool-point Jacobian at `radians`, in joint order. Also the frame pose."""
        axes, pivots, ee_rotation, ee_translation = self._remember_radians(radians)
        point = ee_translation + ee_rotation @ np.array([0.0, 0.0, self.ee_tool_offset])
        linear = np.cross(axes, point - pivots).T
        return linear, axes.T, point, self._ee

    def _jacobian(self, degrees: np.ndarray):
        """Tool-point Jacobian at `degrees`, in joint order. Also the frame pose."""
        return self._jacobian_radians(np.deg2rad(np.asarray(degrees, dtype=float)))

    def _descend(self, q, lower, upper, rest, evaluate) -> _Solution:
        """Damped-least-squares descent shared by every joint-space task.

        `evaluate(radians)` returns the task Jacobian, its residual, the merit
        used to accept a step, whether the task is met, and the pose it measured.
        The best pose seen is returned when no step improves it.
        """
        best = None
        best_cost = np.inf
        for _ in range(_IK_INNER_ITERATIONS):
            jacobian, residual, cost, done, point, rotation = evaluate(q)
            position_error = float(np.linalg.norm(residual[:3]))
            if done:
                return _Solution(np.rad2deg(q), point, rotation, position_error)
            if cost < best_cost:
                best_cost = cost
                best = (q.copy(), point, rotation, position_error)
            step = _dls(jacobian, residual, q, rest)
            accepted = None
            for shrink in range(8):
                trial = np.clip(q + step * (0.5 ** shrink), lower, upper)
                if np.allclose(trial, q, atol=1e-8):
                    break
                if evaluate(trial)[2] < cost:
                    accepted = trial
                    break
            if accepted is None:
                break
            q = accepted
        assert best is not None
        q, point, rotation, position_error = best
        return _Solution(np.rad2deg(q), point, rotation, position_error)

    def _inverse(self, seed_deg, lower_deg, upper_deg, tool_target, rotation,
                 tolerance_m: float, *, lock_orientation: bool) -> _Solution:
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
        orient_limit = math.radians(_POSE_SETTLE_DEG)
        position_limit = min(float(tolerance_m), _PRECISE_GOAL_M)

        def evaluate(radians):
            linear, angular, point, pose = self._jacobian_radians(radians)
            position = target - point
            position_error = float(np.linalg.norm(position))
            frame_rotation = pose[:3, :3]
            if not lock_orientation:
                return linear, position, position_error, position_error <= position_limit, \
                    point, frame_rotation
            angular_error = _rotvec(goal @ frame_rotation.T)
            orient = float(np.linalg.norm(angular_error))
            jacobian = np.vstack((linear, _ORIENTATION_WEIGHT * angular))
            residual = np.concatenate((position, _ORIENTATION_WEIGHT * angular_error))
            done = (position_error <= position_limit and orient <= orient_limit)
            return jacobian, residual, position_error + orient, done, point, frame_rotation

        return self._descend(q, lower, upper, rest, evaluate)

    def _attempt(self, seed, lower, upper, goal, rotation, tolerance_m, *,
                 allow_position_only: bool) -> _Solution:
        """Orientation first. A position goal with no rpy may then drop it to reach."""
        solved = self._inverse(
            seed, lower, upper, goal, rotation, tolerance_m, lock_orientation=True)
        if allow_position_only and solved.position_error > tolerance_m:
            alt = self._inverse(
                seed, lower, upper, goal, rotation, tolerance_m, lock_orientation=False)
            if alt.position_error < solved.position_error:
                solved = alt
        return solved

    def _pan_following(self, seed, lower, upper, goal, rotation, tolerance_m, *,
                       allow_position_only: bool) -> _Solution:
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
            point = solved.point
            if abs(heading_offset(point)) <= _YAW_TOLERANCE_DEG:
                break
        return solved

    def _tcp_jacobian(self, angles):
        """Tool-point linear and angular Jacobians, and the pose they came from."""
        linear, angular, point, pose = self._jacobian(np.asarray(angles, dtype=float))
        return linear, angular, point, pose[:3, :3]

    def _turn_ik(self, seed, goal, rotation, lower, upper, tolerance_m, *,
                 free_yaw: bool):
        """Hold the tool point while the tool turns, or None if it cannot.

        Damped least squares on position plus the tilt the arm can meet. Steps
        stay inside `lower`/`upper`, which the caller narrows to the near window,
        so a blocked turn cannot snap to another arm branch. World-Z rotation is
        free only when the pan carries the tool: on the base axis a wrist twist
        can hold the yaw, so a yaw goal is honoured there.
        """
        target = np.asarray(goal, dtype=float)
        angles = np.clip(np.asarray(seed, dtype=float), lower, upper)
        best, best_error = angles.copy(), np.inf
        settle_m = min(float(tolerance_m), TURN_SETTLE_M)

        def tilt_error(pose) -> np.ndarray:
            error = _rotvec(rotation @ pose.T)
            if free_yaw:
                error[2] = 0.0
            return error

        for _ in range(TURN_ITERATIONS):
            linear, angular, point, pose = self._tcp_jacobian(angles)
            offset = target - point
            error = tilt_error(pose)
            weight = float(np.linalg.norm(offset)) + TURN_ROTATION_WEIGHT * float(
                np.linalg.norm(error))
            if weight < best_error:
                best, best_error = angles.copy(), weight
            if (np.linalg.norm(offset) <= settle_m
                    and np.rad2deg(np.linalg.norm(error)) <= TURN_SETTLE_DEG):
                return _Solution(angles, point, pose, float(np.linalg.norm(offset)))
            jacobian = np.vstack((linear, angular))
            weighted = _TURN_SQRT_WEIGHTS[:, None] * jacobian
            weighted_error = _TURN_SQRT_WEIGHTS * np.concatenate((offset, error))
            step, _coupling = _damped_step(weighted, weighted_error, TURN_DAMPING)
            limit = np.deg2rad(TURN_MAX_STEP_DEG)
            moved = np.clip(angles + np.rad2deg(np.clip(step, -limit, limit)),
                            lower, upper)
            if np.allclose(moved, angles, atol=1e-4):
                break
            angles = moved
        linear, angular, point, pose = self._tcp_jacobian(best)
        error = tilt_error(pose)
        position_error = float(np.linalg.norm(target - point))
        if (position_error > tolerance_m
                or np.rad2deg(np.linalg.norm(error)) > TURN_SETTLE_DEG):
            return None
        return _Solution(best, point, pose, position_error)

    def _downward_twist_free(self, seed_deg, target, lower_deg, upper_deg,
                             tolerance_m: float) -> _Solution:
        """Joint degrees that put the tool point on `target` inside the down cone.

        The orientation-locked solve can miss a point that needs a lean inside
        the cone but no sampled twist. This solve leaves the twist about the
        finger axis free and pulls on the tilt only when the fingers leave the
        cone, so the point stays reachable at the workspace edge.
        """
        target = np.asarray(target, dtype=float)
        q = np.deg2rad(np.asarray(seed_deg, dtype=float))
        lower = np.deg2rad(np.asarray(lower_deg, dtype=float))
        upper = np.deg2rad(np.asarray(upper_deg, dtype=float))
        q = np.clip(q, lower, upper)
        rest = q.copy()
        position_limit = min(float(tolerance_m), _PRECISE_GOAL_M)

        def evaluate(radians):
            linear, angular, point, pose = self._jacobian_radians(radians)
            frame_rotation = pose[:3, :3]
            offset = target - point
            angle, direction = _down_tilt(frame_rotation)
            inside = angle <= _DOWNWARD_TILT
            tilt = np.zeros(2)
            if not inside:
                # Only the two horizontal tilt rows are constrained. Rotation
                # about the finger axis stays free, which is what reaches the edge.
                tilt = (direction[:2] * angle
                        * (1.0 - _DOWNWARD_TILT / angle)
                        * _DOWNWARD_TILT_WEIGHT)
            jacobian = np.vstack((linear, angular[:2]))
            residual = np.concatenate((offset, tilt))
            position_error = float(np.linalg.norm(offset))
            cost = position_error + float(np.linalg.norm(tilt))
            done = position_error <= position_limit and inside
            return jacobian, residual, cost, done, point, frame_rotation

        return self._descend(q, lower, upper, rest, evaluate)

    @_locked
    def position_ik(self, state: dict[str, float], target, limits: dict[str, tuple[float, float]],
                    *, tolerance_m: float = DEFAULT_TOLERANCE_M,
                    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
                    downward: bool = False, rpy=None,
                    follow_yaw: bool = False, local_only: bool = False) -> dict[str, float]:
        """Solve the five arm joints so the tool point reaches target xyz.

        The tool point is `ee_tool_offset` along gripper_frame_link +Z. `rpy` is
        base-frame XYZ Euler degrees, the angles a pose reports; omit it to keep
        roll and pitch, with yaw following the pan. `downward` constrains the
        finger axis (+X) to the cone below the base and leaves the twist about
        that axis free. A fixed rpy pitch of 90 is not a substitute: it pins the
        twist as well, and no exact pose fits near the workspace edge.

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
        reach = self._path_length + abs(self.ee_tool_offset)
        if float(np.linalg.norm(goal)) > reach + 1e-9:
            raise ValueError(f"IK target is outside the arm's reach ({reach:.3f} m)")
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
        # A bare point keeps roll and pitch but lets yaw track the pan, which is
        # what makes a sideways step reachable on this arm. A turn with
        # `follow_yaw` needs the same freedom while it holds the point.
        follow = (goal_rotation is None or follow_yaw) and not downward
        yaw_free = follow and float(np.hypot(goal[0], goal[1])) >= _YAW_RADIUS_M

        def orientation_error_deg(solution: _Solution) -> float:
            if yaw_free:
                # World-Z rotation is the direction a five-joint arm cannot hold
                # while the pan carries the tool; match the tilt the caller asked for.
                error = _rotvec(goal_rotation @ solution.rotation.T)
                error[2] = 0.0
                return float(np.rad2deg(np.linalg.norm(error)))
            return float(np.rad2deg(np.linalg.norm(
                _rotvec(goal_rotation.T @ solution.rotation))))

        def goal_met(solution: _Solution) -> bool:
            if solution.position_error > tolerance_m:
                return False
            if goal_rotation is not None and orientation_error_deg(solution) > tolerance_deg:
                return False
            if downward and solution.rotation[2, 0] >= DOWNWARD_Z:
                return False
            return True

        def solve(seed_angles, low, high):
            if yaw_free:
                return self._pan_following(
                    seed_angles, low, high, goal, rotation, tolerance_m,
                    allow_position_only=allow_position_only)
            return self._attempt(
                seed_angles, low, high, goal, rotation, tolerance_m,
                allow_position_only=allow_position_only)

        near_step = NEAR_ORIENTATION_STEP_DEG if goal_rotation is not None else NEAR_STEP_DEG
        near_lower = np.maximum(lower, initial - near_step)
        near_upper = np.minimum(upper, initial + near_step)
        solved = None
        if np.all(near_lower < near_upper):
            if follow_yaw and follow:
                solved = self._turn_ik(
                    initial, goal, rotation, near_lower, near_upper, tolerance_m,
                    free_yaw=yaw_free)
            else:
                near = solve(initial, near_lower, near_upper)
                orientation_fits = (goal_rotation is None
                                    or orientation_error_deg(near) <= NEAR_ORIENTATION_DEG)
                if orientation_fits and goal_met(near):
                    solved = near
        if solved is None and local_only:
            raise IKError("Cannot hold the tool point through that turn")
        if solved is None:
            # An orientation goal that starts far from an exact wrist pose can
            # stall the locked solve. Reach the point first, then turn the wrist.
            position_first = goal_rotation is not None and not follow_yaw and not downward

            def attempt_seed(seed_angles):
                candidate = solve(seed_angles, lower, upper)
                if position_first and not goal_met(candidate):
                    guess = self._inverse(
                        seed_angles, lower, upper, goal, rotation, tolerance_m,
                        lock_orientation=False)
                    alt = solve(guess.angles, lower, upper)
                    if goal_met(alt) or alt.position_error < candidate.position_error:
                        candidate = alt
                return candidate

            result = attempt_seed(initial)
            home = np.clip(np.zeros_like(initial), lower, upper)
            # A folded or mid-move seed can sit in a local minimum that the home
            # joints escape, so retry from home before giving up.
            if not goal_met(result):
                alt = attempt_seed(home)
                if goal_met(alt) or alt.position_error < result.position_error:
                    result = alt
            # An exact straight-down pose cannot fit every edge point. The cone
            # solve leaves the twist free, so it reaches those points from the
            # fixed seeds without a sweep over sampled orientations.
            if not goal_met(result) and downward:
                def downward_seeds():
                    yield initial
                    if not np.array_equal(home, initial):
                        yield home
                    for fraction in _RESTART_FRACTIONS:
                        yield lower + (upper - lower) * fraction

                for seed_angles in downward_seeds():
                    alt = self._downward_twist_free(
                        seed_angles, goal, lower, upper, tolerance_m)
                    if goal_met(alt):
                        result = alt
                        break
                    if alt.position_error < result.position_error:
                        result = alt
            # A far seed can also land in a narrow basin. The fixed restarts make
            # a reachable goal independent of where the arm happened to start.
            if not goal_met(result) and not downward:
                for fraction in _RESTART_FRACTIONS:
                    alt = attempt_seed(lower + (upper - lower) * fraction)
                    if goal_met(alt):
                        result = alt
                        break
                    if alt.position_error < result.position_error:
                        result = alt
            # Behind and below the base, a reachable pose needs a folded arm with
            # the joints near their stops. Sweep those corners before failing.
            if not goal_met(result) and position_first:
                for seed_angles in _folded_seeds(lower, upper):
                    alt = solve(seed_angles, lower, upper)
                    if goal_met(alt):
                        result = alt
                        break
                    if alt.position_error < result.position_error:
                        result = alt
            if not goal_met(result):
                if downward:
                    raise ValueError(
                        "IK cannot reach that point with the fingers pointing down")
                if result.position_error > tolerance_m:
                    raise ValueError("IK did not reach the point within joint limits")
                raise ValueError("IK did not match the orientation within joint limits")
            solved = result
        return {name: float(value) for name, value in zip(JOINTS, solved.angles)}
