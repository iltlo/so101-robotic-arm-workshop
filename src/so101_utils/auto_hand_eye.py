"""Fixed-pose automatic hand-eye tour shared by the CLI and the workshop page.

Roll stays at -170 degrees, so the wrist clears its stop while pan and pitch
give the samples independent rotation axes. Each recorded pose is mirrored in
the base XZ plane, which yields ten poses with minimal arm travel between them.
"""

# Wait after a move before the arm and the camera are sampled.
DWELL_S = 1.5

# x y z roll pitch yaw, metres and degrees.
RECORDED_POSES = (
    (0.24,  0.03, 0.13, -170.0,  3.0,   8.0),
    (0.23,  0.03, 0.20, -170.0,  2.0,   8.0),
    (0.18, -0.15, 0.20, -170.0,  7.0, -48.0),
    (0.21, -0.16, 0.11, -170.0, -6.0, -44.0),
    (0.31, -0.14, 0.09, -170.0, -2.0, -28.0),
)


def calibration_poses() -> list[tuple[float, float, float, float, float, float]]:
    """The recorded poses plus their base-XZ mirror (flip y and yaw)."""
    poses = list(RECORDED_POSES)
    for x, y, z, roll, pitch, yaw in RECORDED_POSES:
        poses.append((x, -y, z, roll, pitch, -yaw))
    return poses
