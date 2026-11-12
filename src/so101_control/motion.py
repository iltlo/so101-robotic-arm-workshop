"""Speed-limited joint steps. This is not collision avoidance."""

import math
import time

from so101_utils.model import JOINTS

from .kinematics import LIMIT_MARGIN_DEG

DEFAULT_SPEED_DEG_S = 10.0
DEFAULT_PERIOD_S = 0.05
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_TOLERANCE_DEG = 2.0

# The gearbox has backlash and the arm carries its own weight, so a joint can
# stop a few degrees short of its goal and stay there. Accept a still joint
# inside this band; without it every move waits out the timeout.
SETTLE_ACCEPT_DEG = 6.0
SETTLE_STILL_DEG = 0.3
SETTLE_STABLE_S = 0.4

# Feedback further than this from the trajectory means the bus is out of step.
FEEDBACK_LAG_DEG = 8.0
FEEDBACK_LAG_SPEED_FACTOR = 2.0
# A driver that rounds a goal by less than this is not a trajectory failure.
GOAL_EPSILON_DEG = 1e-3


class MotionError(RuntimeError):
    pass


def joint_state(robot) -> dict[str, float]:
    obs = robot.get_observation()
    state = {}
    for joint in (*JOINTS, "gripper"):
        key = f"{joint}.pos"
        value = obs.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise MotionError(f"Bad feedback for {key}")
        state[joint] = float(value)
    return state


def move_joints(robot, targets: dict[str, float], limits: dict[str, tuple[float, float]],
                *, speed_deg_s: float = DEFAULT_SPEED_DEG_S, period_s: float = DEFAULT_PERIOD_S,
                timeout_s: float = DEFAULT_TIMEOUT_S, tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
                wait=True, on_step=None, cancel=None,
                start: dict[str, float] | None = None,
                commanded: dict[str, float] | None = None,
                single_goal: bool = False) -> dict[str, float] | None:
    """Step toward the targets. Returns None when `cancel()` asks to stop.

    The path is split into steps of `speed_deg_s * period_s`. `start` is the
    last joint command. Without it, the split begins at the encoders.
    `single_goal` sends a move of up to two steps in one write: a keyboard nudge
    should not wait out a ramp it can barely see.
    """
    if not targets or not set(targets) <= set(JOINTS):
        raise ValueError("Unknown or empty joint target")
    if speed_deg_s <= 0 or period_s <= 0 or timeout_s <= 0 or tolerance_deg <= 0:
        raise ValueError("Motion settings must be positive")
    measured = joint_state(robot)
    origin = dict(measured)
    if start is not None:
        for name in targets:
            if name not in start or not math.isfinite(start[name]):
                raise ValueError(f"Missing command for {name}")
            origin[name] = float(start[name])
    for joint, value in targets.items():
        if joint not in limits or not math.isfinite(value):
            raise MotionError(f"No limit or invalid target for {joint}")
        low, high = limits[joint]
        if not low <= value <= high:
            raise MotionError(f"{joint} goal {value:.2f}° outside limits [{low:.2f}, {high:.2f}]")
        if not low - LIMIT_MARGIN_DEG <= measured[joint] <= high + LIMIT_MARGIN_DEG:
            raise MotionError(f"{joint} at {measured[joint]:.2f}° outside limits [{low:.2f}, {high:.2f}]")
        # Plan from inside the stop so the first goal is not clipped.
        origin[joint] = min(max(origin[joint], low), high)

    def stopped() -> bool:
        return cancel is not None and cancel()

    begun = time.monotonic()
    distance = max(abs(value - origin[name]) for name, value in targets.items())
    budget = speed_deg_s * period_s
    steps = max(1, math.ceil(distance / budget))
    if single_goal and distance <= 2.0 * budget:
        steps = 1
    for index in range(1, steps + 1):
        if stopped():
            return None
        if time.monotonic() - begun > timeout_s:
            raise MotionError("Motion timed out; no more goals sent")
        point = {name: origin[name] + (value - origin[name]) * index / steps
                 for name, value in targets.items()}
        desired = {f"{name}.pos": angle for name, angle in point.items()}
        measured = joint_state(robot)
        if any(abs(measured[name] - desired[f"{name}.pos"])
               > max(FEEDBACK_LAG_DEG, speed_deg_s * FEEDBACK_LAG_SPEED_FACTOR)
               for name in targets):
            raise MotionError("Feedback differs from the trajectory; no more goals sent")
        actual = robot.send_action(desired)
        if any(abs(actual.get(key, float("nan")) - goal) > GOAL_EPSILON_DEG
               for key, goal in desired.items()):
            raise MotionError("Driver clipped a goal; no more goals sent")
        if commanded is not None:
            commanded.update(point)
        if on_step is not None:
            on_step()
        if index < steps:
            time.sleep(period_s)
    if wait:
        anchor = None
        stable_since = 0.0
        while True:
            if stopped():
                return None
            now = time.monotonic()
            if now - begun > timeout_s:
                raise MotionError("Failed to settle before timeout")
            measured = joint_state(robot)
            error = max(abs(measured[name] - value) for name, value in targets.items())
            if error <= tolerance_deg:
                if on_step is not None:
                    on_step()
                return measured
            if error <= SETTLE_ACCEPT_DEG:
                if anchor is None or any(
                        abs(measured[name] - anchor[name]) > SETTLE_STILL_DEG
                        for name in targets):
                    anchor = measured
                    stable_since = now
                elif now - stable_since >= SETTLE_STABLE_S:
                    print(f"Settled {error:.1f} deg short of the goal; continuing.")
                    return measured
            else:
                anchor = None
            if on_step is not None:
                on_step()
            time.sleep(period_s)
    return joint_state(robot)
