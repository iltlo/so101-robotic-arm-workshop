"""Incremental, feedback-checked joint movement. Not collision avoidance."""

import math
import time
from .config import JOINTS


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
                *, speed_deg_s: float = 8.0, period_s: float = 0.1,
                timeout_s: float = 20.0, tolerance_deg: float = 2.0,
                stop=None, wait=True) -> dict[str, float]:
    if not targets or not set(targets) <= set(JOINTS):
        raise ValueError("Unknown or empty joint target")
    if speed_deg_s <= 0 or period_s <= 0 or timeout_s <= 0 or tolerance_deg <= 0:
        raise ValueError("Motion settings must be positive")
    start = joint_state(robot)
    for joint, value in targets.items():
        if joint not in limits or not math.isfinite(value):
            raise MotionError(f"No verified limit or invalid target for {joint}")
        low, high = limits[joint]
        if not low <= start[joint] <= high or not low <= value <= high:
            raise MotionError(f"{joint} outside verified limits")
    begun = time.monotonic()
    distance = max(abs(value - start[name]) for name, value in targets.items())
    steps = max(1, math.ceil(distance / (speed_deg_s * period_s)))
    for index in range(1, steps + 1):
        if (stop and stop()) or time.monotonic() - begun > timeout_s:
            raise MotionError("Motion aborted or timed out; no more goals sent")
        desired = {f"{name}.pos": start[name] + (value - start[name]) * index / steps
                   for name, value in targets.items()}
        measured = joint_state(robot)
        if any(abs(measured[name] - desired[f"{name}.pos"]) > max(8.0, speed_deg_s * 2)
               for name in targets):
            raise MotionError("Feedback differs from trajectory; no more goals sent")
        actual = robot.send_action(desired)
        if any(abs(actual.get(key, float("nan")) - goal) > 1e-3 for key, goal in desired.items()):
            raise MotionError("Driver clipped a goal; no more goals sent")
        if index < steps:
            time.sleep(period_s)
    if wait:
        while True:
            if (stop and stop()) or time.monotonic() - begun > timeout_s:
                raise MotionError("Failed to settle before timeout")
            measured = joint_state(robot)
            if all(abs(measured[name] - value) <= tolerance_deg for name, value in targets.items()):
                return measured
            time.sleep(period_s)
    return joint_state(robot)


def jog_command(robot, limits: dict[str, tuple[float, float]], command: str) -> dict[str, float]:
    """One bounded arm-joint step; gripper 0..100 is not in degrees."""
    name, amount = command.split()
    delta = float(amount)
    if name not in JOINTS or not math.isfinite(delta) or abs(delta) > 2 or delta == 0:
        raise ValueError("Use <arm_joint> <delta_degrees> with 0 < |delta| <= 2; no gripper jog")
    state = joint_state(robot)
    return move_joints(robot, {name: state[name] + delta}, limits)