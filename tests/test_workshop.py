import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from so101_control.calibration import (Sample, solve_eye_to_hand, solve_eye_to_hand_known_tag,
                                       measured_gripper_T_tag, table_plane, _average, _tag_pose_from_corners,
                                       find_tag)
from so101_control.config import ALL_JOINTS
from so101_control.geometry import Intrinsics, base_link_plane, hover_target, intersect_pixel, normalized_ray
from so101_control.motion import MotionError, move_joints, jog_command
from so101_control.robot import OfflineRobot, connect_live


def pose(rvec, xyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    result[:3, 3] = xyz
    return result


def test_fixed_camera_unknown_mount_synthetic():
    rng = np.random.default_rng(2)
    bTc = pose([0.1, 0.3, -0.4], [0.2, -0.12, 0.45])
    gTtag = pose([0.2, 0.15, -0.1], [0.025, -0.01, 0.06])
    samples = []
    for _ in range(16):
        bTg = pose(rng.uniform(-0.8, 0.8, 3), rng.uniform(-0.2, 0.2, 3))
        cTtag = np.linalg.inv(bTc) @ bTg @ gTtag
        samples.append(Sample(bTg, cTtag))
    solved = solve_eye_to_hand(samples, max_error_m=1e-4, max_error_deg=1e-3)
    assert np.allclose(solved["base_T_camera"], bTc, atol=1e-5)
    assert np.allclose(solved["gripper_T_tag"], gTtag, atol=1e-5)


def test_camera_rotation_average_uses_progressive_slerp():
    rotations = [Rotation.from_euler("z", angle, degrees=True) for angle in (0, 50, 100)]
    samples = [pose(rotation.as_rotvec(), [0.2, 0, 0]) for rotation in rotations]
    expected = Rotation.from_euler("z", 50, degrees=True).as_matrix()
    for order in (samples, samples[::-1]):
        assert np.allclose(_average(order)[:3, :3], expected, atol=1e-6)
        assert np.allclose(_average(order)[:3, 3], [0.2, 0, 0])


def test_camera_rotation_average_slerp_wraparound():
    samples = [pose(Rotation.from_euler("z", angle, degrees=True).as_rotvec(), [0, 0, 0])
               for angle in (170, -170)]
    assert np.allclose(_average(samples)[:3, :3],
                       Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-6)


def test_camera_rotation_average_uses_slerp_for_noncommuting_rotations():
    rotations = [Rotation.from_euler(axis, angle, degrees=True)
                 for axis, angle in (("x", 20), ("y", 60), ("z", 90))]
    samples = [pose(rotation.as_rotvec(), [0, 0, 0]) for rotation in rotations]
    midpoint = Slerp([0, 1], Rotation.from_quat([rotations[0].as_quat(),
                                                 rotations[1].as_quat()]))([0.5])[0]
    expected = Slerp([0, 1], Rotation.from_quat([midpoint.as_quat(),
                                                 rotations[2].as_quat()]))([1 / 3])[0]
    assert np.allclose(_average(samples)[:3, :3], expected.as_matrix(), atol=1e-6)


def test_measured_gripper_tag_and_known_mount_camera_solve():
    mount = measured_gripper_T_tag()
    assert np.allclose(mount[:3, 3], [-0.0372, 0, -0.0284])
    original = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    local_z_half_turn = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    assert np.allclose(mount[:3, :3], original @ local_z_half_turn)
    assert np.allclose(mount[:3, :3] @ [0, 0, 1], original @ [0, 0, 1])
    base_T_camera = pose([0.2, 0.1, -0.1], [0.18, -0.09, 0.45])
    rng = np.random.default_rng(4)
    samples = []
    for _ in range(10):
        base_T_gripper = pose(rng.uniform(-0.7, 0.7, 3), rng.uniform(-0.2, 0.2, 3))
        camera_T_tag = np.linalg.inv(base_T_camera) @ base_T_gripper @ mount
        samples.append(Sample(base_T_gripper, camera_T_tag))
    solution = solve_eye_to_hand_known_tag(samples, mount, max_error_m=1e-5, max_error_deg=1e-4)
    assert np.allclose(solution["base_T_camera"], base_T_camera, atol=1e-6)


@pytest.mark.parametrize("offset_m, accepted", [(0.017, True), (0.021, False)])
def test_known_mount_accepts_seventeen_mm_but_rejects_over_twenty(offset_m, accepted):
    mount = measured_gripper_T_tag()
    base_T_camera = pose([0.2, 0.1, -0.1], [0.18, -0.09, 0.45])
    rng = np.random.default_rng(17)
    samples = []
    for index in range(6):
        gripper = pose(rng.uniform(-0.7, 0.7, 3), rng.uniform(-0.2, 0.2, 3))
        tag = np.linalg.inv(base_T_camera) @ gripper @ mount
        tag[:3, 3] += (1 if index % 2 else -1) * offset_m * np.array([1, 0, 0])
        samples.append(Sample(gripper, tag))
    if accepted:
        solution = solve_eye_to_hand_known_tag(samples, mount)
        assert 0.015 < solution["max_residual_m"] < 0.020
        assert solution["sample_count"] == 6
    else:
        with pytest.raises(ValueError, match="Calibration inconsistent"):
            solve_eye_to_hand_known_tag(samples, mount)


def test_insufficient_and_degenerate_samples():
    sample = Sample(np.eye(4), np.eye(4))
    with pytest.raises(ValueError):
        solve_eye_to_hand([sample] * 6)


def test_rgb_ray_to_table_and_hover():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    bTc = pose([np.pi, 0, 0], [0.1, 0, 0.4])
    plane = {"point": [0, 0, 0], "normal": [0, 0, 1]}
    point = intersect_pixel((320, 240), intr, bTc, **{"point_on_plane": plane["point"],
                                                       "plane_normal": plane["normal"]})
    assert np.allclose(point, [0.1, 0, 0])
    target = hover_target((320, 240), intr, bTc, plane, 0.05,
                          {"x": [0, 0.2], "y": [-0.1, 0.1], "z": [0.02, 0.1]}, np.eye(4))
    assert np.allclose(target, [0.1, 0, 0.05])
    with pytest.raises(ValueError):
        intersect_pixel((-1, 240), intr, bTc, [0, 0, 0], [0, 0, 1])


def test_base_link_plane_hover_needs_no_table_tag():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    base_T_camera = pose([np.pi, 0, 0], [0.1, 0, 0.4])
    plane = base_link_plane()
    assert np.allclose(plane["point"], [0, 0, 0])
    assert np.allclose(plane["normal"], [0, 0, 1])
    assert np.allclose(hover_target((320, 240), intr, base_T_camera, plane, 0.05,
                                    {"x": [0, 0.2], "y": [-0.1, 0.1], "z": [0.02, 0.1]},
                                    np.eye(4)), [0.1, 0, 0.05])


def test_inverse_brown_ray_matches_realsense_deprojection():
    rs = pytest.importorskip("pyrealsense2")
    intr = Intrinsics(640, 480, 500, 500, 320, 240,
                      (0.12, -0.04, 0.001, 0.002, 0.0), "inverse_brown_conrady")
    native = rs.intrinsics()
    native.width, native.height = intr.width, intr.height
    native.fx, native.fy, native.ppx, native.ppy = intr.fx, intr.fy, intr.cx, intr.cy
    native.coeffs = list(intr.distortion)
    native.model = rs.distortion.inverse_brown_conrady
    ray = normalized_ray((500, 280), intr)
    expected = rs.rs2_deproject_pixel_to_point(native, [500, 280], 1.0)
    assert np.allclose(ray, expected[:2], atol=1e-6)


def test_ippe_square_recovers_tag_pose():
    import cv2
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    original = pose([0.2, -0.15, 0.1], [0.02, -0.01, 0.25])
    points = np.array([[-.02, .02, 0], [.02, .02, 0], [.02, -.02, 0], [-.02, -.02, 0]])
    corners, _ = cv2.projectPoints(points, cv2.Rodrigues(original[:3, :3])[0],
                                   original[:3, 3], intr.matrix(), None)
    recovered = _tag_pose_from_corners(corners[:, 0], intr, .04)
    assert np.allclose(recovered, original, atol=1e-6)


@pytest.mark.parametrize("tilted", [False, True])
def test_find_tag_axes_match_dobot_detector_pose(tilted):
    """Our PnP tag axes must match pupil_apriltags.pose_R used by Dobot."""
    import cv2
    from pupil_apriltags import Detector

    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 320)
    gray = np.full((480, 640), 255, dtype=np.uint8)
    gray[80:400, 160:480] = marker
    if tilted:
        source = np.float32([[160, 80], [479, 80], [479, 399], [160, 399]])
        destination = np.float32([[150, 130], [435, 60], [490, 340], [180, 410]])
        gray = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(source, destination),
                                   (640, 480), borderValue=255)
    detector = Detector(families="tag36h11")
    reference = detector.detect(gray, estimate_tag_pose=True,
                                camera_params=(500, 500, 320, 240), tag_size=0.04)[0]
    ours, corners = find_tag(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), intr, detector=detector)
    assert np.allclose(corners, reference.corners)
    delta = Rotation.from_matrix(reference.pose_R.T @ ours[:3, :3]).magnitude()
    assert np.rad2deg(delta) < (15 if tilted else 5)
    assert np.linalg.norm(ours[:3, 3] - reference.pose_t.ravel()) < (0.004 if tilted else 0.001)


def test_motion_limits_and_fake():
    robot = OfflineRobot()
    with pytest.raises(PermissionError):
        connect_live({"port": "/dev/null"}, enable_live=False)
    limits = {joint: (-10, 10) for joint in ALL_JOINTS if joint != "gripper"}
    assert move_joints(robot, {"shoulder_pan": 1}, limits, speed_deg_s=100,
                       period_s=0.1)["shoulder_pan"] == 1
    with pytest.raises(MotionError):
        move_joints(robot, {"elbow_flex": 11}, limits)
    assert robot.get_observation()["elbow_flex.pos"] == 0


def test_interactive_jog_command_is_bounded():
    robot = OfflineRobot()
    limits = {joint: (-10, 10) for joint in ALL_JOINTS if joint != "gripper"}
    assert jog_command(robot, limits, "shoulder_pan 1")["shoulder_pan"] == 1
    with pytest.raises(ValueError):
        jog_command(robot, limits, "gripper 1")
    with pytest.raises(ValueError):
        jog_command(robot, limits, "shoulder_pan 3")
    with pytest.raises(ValueError):
        jog_command(robot, limits, "shoulder_pan nan")


def test_table_sign_and_urdf_pose():
    import rerun as rr
    plane = table_plane(np.eye(4), np.eye(4))
    assert np.allclose(plane["normal"], [0, 0, 1])
    from pathlib import Path
    path = Path(__file__).parents[1] / "assets/SO101/so101_new_calib.urdf"
    if not path.exists():
        pytest.skip("download official model assets to test twin")
    model = rr.urdf.UrdfTree.from_file_path(path)
    assert model.name == "so101_new_calib"
    shoulder = model.get_joint_by_name("shoulder_pan")
    assert shoulder.parent_link == "base_link"
    assert shoulder.child_link == "shoulder_link"
    assert "parent_frame=['base_link']" in repr(shoulder.compute_transform(0.1))