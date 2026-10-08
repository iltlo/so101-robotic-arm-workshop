"""Click-to-hover should require accepted extrinsics, but not a tag-on-table measurement."""

import importlib.util
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest


def test_preview_uses_accepted_camera_calibration_without_plane(monkeypatch):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"gripper_T_tcp": np.eye(4).tolist(), "hover_clearance_m": 0.05,
              "workspace_m": {"x": [0, 0.2], "y": [-0.1, 0.1], "z": [0, 0.1]}}
    calibration = {"base_T_camera": np.eye(4).tolist(),
                   "max_residual_m": 0.001, "max_residual_deg": 1}
    monkeypatch.setattr(module, "load_station", lambda _: config)

    def load(_path, *, require_plane=False):
        assert not require_plane
        return calibration

    monkeypatch.setattr(module, "load_calibration", load)

    def stop_before_hardware(_config):
        raise RuntimeError("camera must not open in this test")

    monkeypatch.setattr(module, "rgb_camera", stop_before_hardware)
    with pytest.raises(RuntimeError, match="camera must not open"):
        module.main([])


def test_preview_with_unmeasured_tcp_workspace_and_clearance_opens_camera(monkeypatch):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_unmeasured", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_station", lambda _: {
        "gripper_T_tcp": None, "hover_clearance_m": None, "workspace_m": None})
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": np.eye(4).tolist(), "max_residual_m": 0.005,
        "max_residual_deg": 1.8})

    def camera(_config):
        raise RuntimeError("camera opening, no robot connection")

    monkeypatch.setattr(module, "rgb_camera", camera)
    with pytest.raises(RuntimeError, match="camera opening, no robot connection"):
        module.main([])


def test_preview_projects_browser_click_without_opening_robot_or_opencv_gui(monkeypatch, capsys):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_browser", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_station", lambda _: {"gripper_T_tcp": None})
    base_T_camera = np.eye(4)
    base_T_camera[:3, :3] = np.diag([1, -1, -1])
    base_T_camera[:3, 3] = [0.1, 0, 0.4]
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": base_T_camera, "max_residual_m": 0.005,
        "max_residual_deg": 1.8})
    monkeypatch.setattr(module, "matches_camera", lambda *args: None)
    from so101_control.geometry import Intrinsics

    class Camera:
        intrinsics = Intrinsics(640, 480, 500, 500, 320, 240)

        def read(self):
            return type("Frame", (), {"rgb_bgr": np.zeros((480, 640, 3), dtype=np.uint8),
                                       "timestamp": 0})()

    @contextmanager
    def camera(_):
        yield Camera()

    @contextmanager
    def viewer(*args):
        class View:
            frame_seq = 1

            def publish(self, image):
                pass

            def consume(self):
                return [(320, 240, 1)]

            def report(self, text):
                pass

        yield View(), "http://127.0.0.1:12345/"

    monkeypatch.setattr(module, "rgb_camera", camera)
    monkeypatch.setattr(module, "click_view", viewer)
    @contextmanager
    def offline_robot(_config, live):
        assert live is False
        yield object()

    monkeypatch.setattr(module, "station_robot", offline_robot)
    monkeypatch.setattr(module, "require_fresh", lambda _: None)
    def halt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.time, "sleep", halt)
    module.main([])
    output = capsys.readouterr().out
    assert "base_link z=0 intersection (m)" in output
    assert "[0.1 0.  0. ]" in output


def test_live_hover_refuses_missing_tcp_before_opening_camera(monkeypatch):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_missing_tcp", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_station", lambda _: {
        "gripper_T_tcp": None, "fk_tcp_verified": True, "base_plane_verified": True})
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": np.eye(4).tolist(), "max_residual_m": 0.005,
        "max_residual_deg": 1.8})
    with pytest.raises(ValueError, match="Hand-eye determines base_T_camera, not the physical tool center"):
        module.main(["--enable-live"])


def test_live_hover_requires_verified_physical_base_plane_before_hardware(monkeypatch):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_live", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"gripper_T_tcp": np.eye(4).tolist(), "fk_tcp_verified": True,
              "base_plane_verified": False}
    calibration = {"base_T_camera": np.eye(4).tolist(),
                   "max_residual_m": 0.001, "max_residual_deg": 1}
    monkeypatch.setattr(module, "load_station", lambda _: config)
    monkeypatch.setattr(module, "load_calibration", lambda _: calibration)
    with pytest.raises(ValueError, match="base_plane_verified"):
        module.main(["--enable-live"])


def test_live_click_displays_goal_without_workspace_but_never_commands_motion(monkeypatch, capsys):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_goal_preview", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"gripper_T_tcp": np.eye(4).tolist(), "fk_tcp_verified": True,
              "base_plane_verified": True, "hover_clearance_m": 0.04,
              "workspace_m": None, "urdf": "unused", "joint_limits_deg": {
                  name: [-90, 90] for name in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                "wrist_flex", "wrist_roll")}}
    extrinsics = np.diag([1., -1., -1., 1.])
    extrinsics[:3, 3] = [0.1, 0, 0.4]
    monkeypatch.setattr(module, "load_station", lambda _: config)
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": extrinsics, "max_residual_m": 0.005, "max_residual_deg": 1})
    monkeypatch.setattr(module, "matches_camera", lambda *args: None)
    monkeypatch.setattr(module, "require_fresh", lambda _: None)
    from so101_control.geometry import Intrinsics

    class Camera:
        intrinsics = Intrinsics(640, 480, 500, 500, 320, 240)

        def read(self):
            return type("Frame", (), {"rgb_bgr": np.zeros((480, 640, 3), dtype=np.uint8)})()

    @contextmanager
    def camera(_):
        yield Camera()

    @contextmanager
    def viewer(*args):
        class View:
            frame_seq = 1

            def publish(self, image):
                pass

            def consume(self):
                return [(320, 240, 1)]

            def report(self, text):
                pass

        yield View(), "http://127.0.0.1:12345/"

    class Robot:
        def get_observation(self):
            return {f"{name}.pos": 0.0 for name in (*config["joint_limits_deg"], "gripper")}

        def send_action(self, action):
            pytest.fail("preview must not command motion")

    @contextmanager
    def robot(_config, live):
        assert live
        yield Robot()

    goals = []

    class FakeTwin:
        def __init__(self, urdf):
            assert urdf == "unused"

        def update(self, state, *, ee_pose):
            pass

        def show_hover_goal(self, point):
            goals.append(np.asarray(point))

    class FakeKinematics:
        def __init__(self, urdf):
            pass

        def fk(self, state, tcp):
            pose = np.eye(4)
            pose[2, 3] = 0.2
            return pose

    monkeypatch.setattr(module, "rgb_camera", camera)
    monkeypatch.setattr(module, "click_view", viewer)
    monkeypatch.setattr(module, "station_robot", robot)
    monkeypatch.setattr(module, "Twin", FakeTwin, raising=False)
    monkeypatch.setattr(module, "Kinematics", FakeKinematics)
    monkeypatch.setattr(module.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unverified goal must not prompt for motion"))
    module.main(["--enable-live"])
    assert len(goals) == 1
    np.testing.assert_allclose(goals[0], [0.1, 0, 0.04], atol=1e-8)
    assert "Rejected click: Verified workspace bounds required" in capsys.readouterr().out



def test_simulation_animates_ik_without_connecting_or_commanding_robot(monkeypatch, capsys):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_simulation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from so101_control.geometry import Intrinsics

    config = {"gripper_T_tcp": np.eye(4).tolist(), "hover_clearance_m": 0.04,
              "workspace_m": None, "urdf": "unused", "joint_limits_deg": {
                  name: [-90, 90] for name in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                "wrist_flex", "wrist_roll")}}
    extrinsics = np.diag([1., -1., -1., 1.])
    extrinsics[:3, 3] = [0.1, 0, 0.4]
    monkeypatch.setattr(module, "load_station", lambda _: config)
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": extrinsics, "max_residual_m": 0.005, "max_residual_deg": 1})
    monkeypatch.setattr(module, "matches_camera", lambda *args: None)
    monkeypatch.setattr(module, "require_fresh", lambda _: None)

    class Camera:
        intrinsics = Intrinsics(640, 480, 500, 500, 320, 240)

        def read(self):
            return type("Frame", (), {"rgb_bgr": np.zeros((480, 640, 3), dtype=np.uint8)})()

    @contextmanager
    def camera(_):
        yield Camera()

    @contextmanager
    def viewer(*args):
        class View:
            frame_seq = 1

            def publish(self, image):
                pass

            def consume(self):
                return [(320, 240, 1)]

            def report(self, text):
                pass

        yield View(), "http://127.0.0.1:12345/"

    poses = []
    goals = []

    class FakeTwin:
        def __init__(self, urdf):
            assert urdf == "unused"

        def update(self, state, *, ee_pose):
            poses.append((state.copy(), ee_pose.copy()))

        def show_hover_goal(self, point):
            goals.append(np.asarray(point))

    class FakeKinematics:
        def __init__(self, urdf):
            assert urdf == "unused"

        def fk(self, state, tcp):
            pose = np.eye(4)
            pose[:3, 3] = [state["shoulder_pan"] / 100, 0, 0.2]
            return pose

        def simulate_tcp_ik(self, state, target, tcp, limits):
            np.testing.assert_allclose(target, [0.1, 0, 0.04], atol=1e-8)
            assert state["shoulder_pan"] == 0
            return {"shoulder_pan": 10.0}

    monkeypatch.setattr(module, "rgb_camera", camera)
    monkeypatch.setattr(module, "click_view", viewer)
    monkeypatch.setattr(module, "station_robot", lambda *args: pytest.fail("simulation connected robot"))
    monkeypatch.setattr(module, "Twin", FakeTwin)
    monkeypatch.setattr(module, "Kinematics", FakeKinematics)
    monkeypatch.setattr(module.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("simulation prompted for motion"))
    module.main(["--simulate"])
    assert len(goals) == 1
    assert [state["shoulder_pan"] for state, _ in poses] == [0.0, 10.0]
    assert "SIMULATION ONLY" in capsys.readouterr().out


def test_simulated_tcp_ik_reaches_fk_target_with_real_urdf():
    from so101_control.config import JOINTS, load_station, transform, validated_limits
    from so101_control.kinematics import Kinematics

    config = load_station(Path(__file__).parents[1] / "config" / "station.yaml")
    if not Path(config["urdf"]).exists():
        pytest.skip("Pinned URDF assets required")
    kin = Kinematics(config["urdf"])
    tcp = transform(config["gripper_T_tcp"])
    state = {**dict.fromkeys(JOINTS, 0.0), "gripper": 0.0}
    goal = np.array([0.246, 0.0, 0.02])
    solution = kin.simulate_tcp_ik(state, goal, tcp, validated_limits(config))
    np.testing.assert_allclose(kin.fk(solution, tcp)[:3, 3], goal, atol=0.015)


def test_live_ik_matches_simulated_tcp_solution_with_real_urdf():
    from so101_control.config import JOINTS, load_station, transform, validated_limits
    from so101_control.kinematics import Kinematics

    config = load_station(Path(__file__).parents[1] / "config" / "station.yaml")
    if not Path(config["urdf"]).exists():
        pytest.skip("Pinned URDF assets required")
    kin = Kinematics(config["urdf"])
    tcp = transform(config["gripper_T_tcp"])
    state = {**dict.fromkeys(JOINTS, 0.0), "gripper": 0.0}
    limits = validated_limits(config)
    target = np.array([0.246, 0.0, 0.02])
    pose = kin.fk(state, tcp)
    pose[:3, 3] = target

    simulated = kin.simulate_tcp_ik(state, target, tcp, limits)
    live = kin.hover_ik(state, pose, tcp, limits)
    for name in JOINTS:
        assert live[name] == pytest.approx(simulated[name])
    assert np.linalg.norm(kin.fk(live, tcp)[:3, 3] - target) <= 0.015


def test_live_ik_rejects_unreachable_target_without_motion():
    from so101_control.config import JOINTS, load_station, transform, validated_limits
    from so101_control.kinematics import Kinematics

    config = load_station(Path(__file__).parents[1] / "config" / "station.yaml")
    if not Path(config["urdf"]).exists():
        pytest.skip("Pinned URDF assets required")
    kin = Kinematics(config["urdf"])
    tcp = transform(config["gripper_T_tcp"])
    state = {**dict.fromkeys(JOINTS, 0.0), "gripper": 0.0}
    pose = kin.fk(state, tcp)
    pose[:3, 3] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="did not reach requested TCP position"):
        kin.hover_ik(state, pose, tcp, validated_limits(config))


def test_simulation_and_live_modes_are_mutually_exclusive():
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_mode_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="2"):
        module.main(["--simulate", "--enable-live"])


def test_preview_rejects_saved_calibration_over_hover_accuracy_limit(monkeypatch):
    script = Path(__file__).parents[1] / "vision" / "click_to_hover.py"
    spec = importlib.util.spec_from_file_location("click_to_hover_residual", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_station", lambda _: {"gripper_T_tcp": np.eye(4).tolist()})
    monkeypatch.setattr(module, "load_calibration", lambda _: {
        "base_T_camera": np.eye(4).tolist(), "max_residual_m": 0.017,
        "max_residual_deg": 4.1})
    with pytest.raises(ValueError, match="Hand-eye residual exceeds acceptance threshold"):
        module.main([])