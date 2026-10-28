"""Speed-limited joint steps. This is not collision avoidance."""

import math
import time

from .kinematics import JOINTS, LIMIT_MARGIN_DEG


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
                *, speed_deg_s: float = 10.0, period_s: float = 0.05,
                timeout_s: float = 20.0, tolerance_deg: float = 2.0,
                wait=True, on_step=None, cancel=None) -> dict[str, float] | None:
    """Step toward the targets. Returns None when `cancel()` asks to stop."""
    if not targets or not set(targets) <= set(JOINTS):
        raise ValueError("Unknown or empty joint target")
    if speed_deg_s <= 0 or period_s <= 0 or timeout_s <= 0 or tolerance_deg <= 0:
        raise ValueError("Motion settings must be positive")
    start = joint_state(robot)
    for joint, value in targets.items():
        if joint not in limits or not math.isfinite(value):
            raise MotionError(f"No limit or invalid target for {joint}")
        low, high = limits[joint]
        if not low <= value <= high:
            raise MotionError(f"{joint} goal {value:.2f}° outside limits [{low:.2f}, {high:.2f}]")
        if not low - LIMIT_MARGIN_DEG <= start[joint] <= high + LIMIT_MARGIN_DEG:
            raise MotionError(f"{joint} at {start[joint]:.2f}° outside limits [{low:.2f}, {high:.2f}]")
        # Plan from inside the stop so the first goal is not clipped.
        start[joint] = min(max(start[joint], low), high)

    def stopped() -> bool:
        return cancel is not None and cancel()

    begun = time.monotonic()
    distance = max(abs(value - start[name]) for name, value in targets.items())
    steps = max(1, math.ceil(distance / (speed_deg_s * period_s)))
    for index in range(1, steps + 1):
        if stopped():
            return None
        if time.monotonic() - begun > timeout_s:
            raise MotionError("Motion timed out; no more goals sent")
        desired = {f"{name}.pos": start[name] + (value - start[name]) * index / steps
                   for name, value in targets.items()}
        measured = joint_state(robot)
        if any(abs(measured[name] - desired[f"{name}.pos"]) > max(8.0, speed_deg_s * 2)
               for name in targets):
            raise MotionError("Feedback differs from the trajectory; no more goals sent")
        actual = robot.send_action(desired)
        if any(abs(actual.get(key, float("nan")) - goal) > 1e-3 for key, goal in desired.items()):
            raise MotionError("Driver clipped a goal; no more goals sent")
        if on_step is not None:
            on_step()
        if index < steps:
            time.sleep(period_s)
    if wait:
        while True:
            if stopped():
                return None
            if time.monotonic() - begun > timeout_s:
                raise MotionError("Failed to settle before timeout")
            measured = joint_state(robot)
            if all(abs(measured[name] - value) <= tolerance_deg for name, value in targets.items()):
                if on_step is not None:
                    on_step()
                return measured
            if on_step is not None:
                on_step()
            time.sleep(period_s)
    return joint_state(robot)
