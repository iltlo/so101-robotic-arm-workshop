"""Standard Denavit–Hartenberg chain for a serial revolute arm.

A row is (θ, d, a, α), in that order. θ is the joint variable plus a fixed
offset. The transform from the previous frame to the next one is

    Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

Nothing in this module knows a joint name, a URDF, or which arm it is. An arm
adapter supplies the rows, a fixed base frame, and the fixed tool frame.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DhRow:
    """One standard DH joint. Lengths are metres and angles are radians."""

    theta: float
    d: float
    a: float
    alpha: float

    def as_dict(self) -> dict[str, float]:
        return {
            "theta_rad": self.theta,
            "d_m": self.d,
            "a_m": self.a,
            "alpha_rad": self.alpha,
        }


def dh_transform(row: DhRow, joint: float) -> np.ndarray:
    """4×4 frame change for `row` when the revolute variable is `joint` radians."""
    theta = row.theta + float(joint)
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(row.alpha), math.sin(row.alpha)
    a, d = row.a, row.d
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("DH axis has no direction")
    return np.asarray(vector, dtype=float) / length


def _perpendicular(axis: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to `axis`."""
    axis = _unit(axis)
    other = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, other))


def _signed_angle(start: np.ndarray, end: np.ndarray, axis: np.ndarray) -> float:
    """Right-handed angle from `start` to `end` about `axis`."""
    axis = _unit(axis)
    start = start - np.dot(start, axis) * axis
    end = end - np.dot(end, axis) * axis
    start_length = float(np.linalg.norm(start))
    end_length = float(np.linalg.norm(end))
    if start_length < 1e-12 or end_length < 1e-12:
        return 0.0
    start = start / start_length
    end = end / end_length
    return math.atan2(float(np.dot(np.cross(start, end), axis)), float(np.dot(start, end)))


def _frame(origin: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    z_axis = _unit(z_axis)
    x_axis = np.asarray(x_axis, dtype=float) - np.dot(x_axis, z_axis) * z_axis
    x_axis = _unit(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = origin
    return pose


def _common_normal(origin: np.ndarray, first: np.ndarray, point: np.ndarray, second: np.ndarray):
    """Feet and signed distance of the common normal from one axis line to the next.

    `first` and `second` are unit directions. The returned direction points along
    first × second when the axes are not parallel, and the distance is signed
    along that direction.
    """
    first = _unit(first)
    second = _unit(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-8:
        offset = point - origin
        along = float(np.dot(offset, first))
        perpendicular = offset - along * first
        distance = float(np.linalg.norm(perpendicular))
        if distance < 1e-10:
            return origin.copy(), origin + along * first, _perpendicular(first), 0.0
        direction = perpendicular / distance
        return origin.copy(), origin + perpendicular, direction, distance
    direction = cross / sine
    system = np.column_stack((first, -second, direction))
    slide_first, _slide_second, distance = np.linalg.solve(system, point - origin)
    foot_first = origin + slide_first * first
    foot_second = foot_first + distance * direction
    return foot_first, foot_second, direction, float(distance)


class DhChain:
    """Forward kinematics and the geometric Jacobian of one DH table.

    `base` is the fixed frame of the first joint axis, expressed in the robot
    base. `tool` and `link` are fixed frames from the end of the table to the
    tool and to an intermediate link. Both are identity until `mount` sees the
    arm's real frames.
    """

    def __init__(self, rows: tuple[DhRow, ...], base: np.ndarray):
        if not rows:
            raise ValueError("A DH chain needs at least one joint")
        self.rows = tuple(rows)
        self.base = np.asarray(base, dtype=float)
        self.tool = np.eye(4)
        self.link = np.eye(4)

    @classmethod
    def from_axes(cls, points, directions) -> "DhChain":
        """Rows whose axes at zero joint angles are `directions` through `points`.

        Points and directions are in the robot base frame, root joint first.
        The last row is a pure rotation about the last axis; the tool frame
        carries whatever rigid offset remains.
        """
        count = len(directions)
        if count == 0 or len(points) != count:
            raise ValueError("DH axes and points must be the same non-empty list")
        directions = [_unit(direction) for direction in directions]
        points = [np.asarray(point, dtype=float) for point in points]
        if count == 1:
            origin = points[0]
            x_axis = _perpendicular(directions[0])
        else:
            origin, _next, x_axis, _distance = _common_normal(
                points[0], directions[0], points[1], directions[1])
        base = _frame(origin, x_axis, directions[0])
        previous_origin = origin
        previous_z = directions[0]
        previous_x = base[:3, 0].copy()
        rows = []
        for index in range(count - 1):
            foot_previous, foot_next, normal, _length = _common_normal(
                previous_origin, previous_z, points[index + 1], directions[index + 1])
            normal = _unit(normal - np.dot(normal, previous_z) * previous_z)
            rows.append(DhRow(
                theta=_signed_angle(previous_x, normal, previous_z),
                d=float(np.dot(foot_previous - previous_origin, previous_z)),
                a=float(np.dot(foot_next - foot_previous, normal)),
                alpha=_signed_angle(previous_z, directions[index + 1], normal),
            ))
            previous_origin = foot_next
            previous_z = directions[index + 1]
            previous_x = normal - np.dot(normal, previous_z) * previous_z
            previous_x = _unit(previous_x)
        rows.append(DhRow(theta=0.0, d=0.0, a=0.0, alpha=0.0))
        return cls(tuple(rows), base)

    def mount(self, tool: np.ndarray, link: np.ndarray) -> None:
        """Fix the tool and link frames from their poses at zero joint angles."""
        end = self.end(np.zeros(len(self.rows)))
        self.tool = np.linalg.inv(end) @ np.asarray(tool, dtype=float)
        self.link = np.linalg.inv(end) @ np.asarray(link, dtype=float)

    def end(self, joints: np.ndarray) -> np.ndarray:
        """Frame after the last row, before the tool mount."""
        pose = self.base.copy()
        for row, angle in zip(self.rows, np.asarray(joints, dtype=float)):
            pose = pose @ dh_transform(row, angle)
        return pose

    def frames(self, joints: np.ndarray):
        """Joint axes, pivots, link pose, and tool pose at `joints` radians.

        Axes and pivots follow the row order. An axis is the z column of the
        frame the joint rotates, which is the standard DH joint axis.
        """
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (len(self.rows),):
            raise ValueError(f"Expected {len(self.rows)} joint angles")
        pose = self.base.copy()
        axes = np.zeros((len(self.rows), 3))
        pivots = np.zeros((len(self.rows), 3))
        for index, (row, angle) in enumerate(zip(self.rows, joints)):
            axes[index] = pose[:3, 2]
            pivots[index] = pose[:3, 3]
            pose = pose @ dh_transform(row, angle)
        return axes, pivots, pose @ self.link, pose @ self.tool

    def table(self) -> tuple[dict[str, float], ...]:
        return tuple(row.as_dict() for row in self.rows)
