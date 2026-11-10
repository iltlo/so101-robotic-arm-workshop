"""The viewer must use this environment's rerun and a free port pair."""

import socket
import threading
import time
from types import SimpleNamespace

from so101_utils.views import Twin
from so101_workshop import viewer
from so101_workshop.agent import Agent


def test_rerun_binary_prefers_the_interpreter_environment(tmp_path, monkeypatch):
    fake = tmp_path / "rerun"
    fake.write_text("")
    monkeypatch.setattr(viewer.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(viewer.shutil, "which", lambda name: "/snap/bin/rerun")
    assert viewer.rerun_binary() == str(fake)


def test_viewer_skips_ports_another_instance_holds(monkeypatch):
    held = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        first, second = sorted(sock.getsockname()[1] for sock in held)
        monkeypatch.setattr(viewer, "GRPC_PORT", first)
        monkeypatch.setattr(viewer, "WEB_PORT", second)
        chosen = viewer._pick_ports()
        assert chosen[0] != first
        assert chosen[1] != second
    finally:
        for sock in held:
            sock.close()


def test_a_stopped_viewer_is_reported_and_can_restart():
    class Stopped:
        url = "http://127.0.0.1:9091/"

        def running(self) -> bool:
            return False

    agent = Agent()
    agent.viewer = Stopped()
    status = agent.viewer_status()
    assert status["enabled"] is False
    assert "stopped" in status["detail"]


def test_starting_the_viewer_returns_before_the_flush_finishes(monkeypatch):
    gate = threading.Event()

    def slow_open(urdf, intrinsics):
        gate.wait(2)
        raise RuntimeError("viewer stub")

    monkeypatch.setattr(viewer, "open_viewer", slow_open)
    monkeypatch.setattr(Agent, "_ensure_urdf", lambda self: None)
    agent = Agent()
    started = time.monotonic()
    status = agent.start_viewer()
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert status["starting"] is True
    assert "Starting" in status["detail"]
    gate.set()
    agent._viewer_thread.join(timeout=2)
    finished = agent.viewer_status()
    assert finished["starting"] is False
    assert finished["detail"] == "viewer stub"


def test_viewer_command_bounds_memory():
    command = viewer._viewer_command("rerun", 9881, 9091, "http://127.0.0.1:9091")
    assert command[command.index("--memory-limit") + 1] == viewer.VIEWER_MEMORY_LIMIT
    assert command[command.index("--server-memory-limit") + 1] == viewer.SERVER_MEMORY_LIMIT
    # 0.26 rejects these. Passing them makes the Intel Mac viewer exit immediately.
    assert "--headless" not in command
    assert "--cors-allow-origin" not in command


def test_viewer_is_dead_when_its_grpc_port_closes():
    class Alive:
        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return 0

    class TwinStub:
        alive = True

        def __init__(self):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    twin = TwinStub()
    instance = viewer.WorkshopViewer(twin, "http://127.0.0.1:9091/", Alive(), port)
    assert instance.running() is False
    instance.close()
    assert twin.closed is True
    assert instance.running() is False


def test_the_viewer_keeps_ticking_while_a_job_owns_the_motion_worker(monkeypatch):
    agent = Agent()
    ticks = []
    monkeypatch.setattr(agent, "_refresh_viewer", lambda: ticks.append(time.monotonic()))
    agent.start_worker()
    gate = threading.Event()
    job = threading.Thread(target=lambda: agent.submit(gate.wait, timeout=5))
    job.start()
    try:
        time.sleep(0.15)
        first = len(ticks)
        time.sleep(0.25)
        assert first >= 1
        assert len(ticks) > first
    finally:
        gate.set()
        job.join(2)
        agent.close()


def test_the_agent_detaches_a_dead_viewer_before_the_next_flush():
    class DeadViewer:
        twin = SimpleNamespace(alive=True)

        def __init__(self):
            self.closed = False

        def running(self) -> bool:
            return False

        def close(self) -> None:
            self.closed = True

    agent = Agent()
    dead = DeadViewer()
    agent.viewer = dead
    agent._refresh_viewer()
    assert agent.viewer is None
    assert dead.closed is True
    assert "stopped" in agent.viewer_error


def test_a_closed_twin_stops_logging():
    class RR:
        disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    twin = Twin.__new__(Twin)
    twin.alive = True
    twin.rr = RR()
    twin.close()
    assert twin.alive is False
    assert twin.rr.disconnected is True
    twin.update({})
    twin.flush()


def test_stale_viewer_record_is_killed_and_cleared(monkeypatch, tmp_path):
    record = tmp_path / "viewer.json"
    record.write_text('{"pid": 4242, "grpc_port": 9881, "web_port": 9091}')
    killed, waited = [], []
    monkeypatch.setattr(viewer, "_viewer_state_path", lambda: record)
    monkeypatch.setattr(viewer, "_is_rerun_process", lambda pid: pid == 4242)
    monkeypatch.setattr(viewer, "_terminate_pid", killed.append)
    monkeypatch.setattr(viewer, "_wait_for_port", waited.append)
    viewer.stop_stale_viewer()
    assert killed == [4242]
    assert waited == [9881]
    assert not record.exists()


def test_stale_viewer_record_is_dropped_when_the_pid_is_not_rerun(monkeypatch, tmp_path):
    record = tmp_path / "viewer.json"
    record.write_text('{"pid": 4242, "grpc_port": 9881, "web_port": 9091}')
    killed = []
    monkeypatch.setattr(viewer, "_viewer_state_path", lambda: record)
    monkeypatch.setattr(viewer, "_is_rerun_process", lambda pid: False)
    monkeypatch.setattr(viewer, "_terminate_pid", killed.append)
    viewer.stop_stale_viewer()
    assert killed == []
    assert not record.exists()


def test_the_stale_viewer_check_reads_the_command_line_on_macos(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="/venv/bin/python /venv/bin/rerun --headless")

    monkeypatch.setattr(viewer.sys, "platform", "darwin")
    monkeypatch.setattr(viewer.subprocess, "run", fake_run)
    assert viewer._is_rerun_process(4242) is True
    assert calls[0][-1] == "args="


def test_each_viewer_start_gets_a_new_generation(monkeypatch):
    class Stub:
        url = "http://127.0.0.1:9091/"
        twin = SimpleNamespace(alive=True)

        def close(self) -> None:
            pass

        def running(self) -> bool:
            return True

    monkeypatch.setattr(Agent, "_ensure_urdf", lambda self: None)
    monkeypatch.setattr(viewer, "open_viewer", lambda urdf, intrinsics: Stub())
    agent = Agent()
    assert agent.viewer_status()["generation"] == ""
    agent._open_viewer()
    first = agent.viewer_status()["generation"]
    agent._open_viewer()
    second = agent.viewer_status()["generation"]
    assert first and second and first != second

