"""Tests for ZellijManager."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from bridge.zellij import (
    ZellijManager,
    ZellijError,
    ZellijSessionMissing,
    ZellijSpawnError,
)


@dataclass
class FakeProc:
    """Minimal fake subprocess for testing."""

    returncode: int
    stdout_data: bytes
    stderr_data: bytes

    async def wait(self) -> int:
        """Return the exit code."""
        return self.returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return stdout and stderr."""
        return (self.stdout_data, self.stderr_data)


@pytest.fixture
def patch_exec(monkeypatch):
    """Monkeypatch asyncio.create_subprocess_exec to return queued fake procs.

    Returns a list that can be populated with FakeProc instances.
    Each call to _create_subprocess_exec pops from the front of the list.
    """
    queue: list[FakeProc] = []
    call_log: list[tuple] = []

    async def _create_subprocess_exec(*argv: str, **kwargs: Any) -> FakeProc:
        """Record the call and pop the next proc from queue."""
        call_log.append((argv, kwargs))
        if not queue:
            raise AssertionError(f"Unexpected zellij call: {argv}")
        return queue.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create_subprocess_exec)
    # Make queue accessible to tests
    _create_subprocess_exec._queue = queue  # type: ignore
    _create_subprocess_exec._call_log = call_log  # type: ignore
    return _create_subprocess_exec


@pytest.fixture
def load_fixture():
    """Load JSON fixture from tests/fixtures/zellij_list_panes.json."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "zellij_list_panes.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.mark.asyncio
class TestZellijManager:
    """Tests for ZellijManager."""

    async def test_ensure_session_alive_success(self, patch_exec):
        """ensure_session_alive runs zellij attach --create-background bridge."""
        proc = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        await mgr.ensure_session_alive()

        # Check argv
        assert len(patch_exec._call_log) == 1
        argv, kwargs = patch_exec._call_log[0]
        assert argv == ("zellij", "attach", "--create-background", "bridge")

    async def test_ensure_session_alive_idempotent(self, patch_exec):
        """ensure_session_alive is idempotent on success."""
        proc = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc)
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        await mgr.ensure_session_alive()
        await mgr.ensure_session_alive()

        # Both calls should succeed
        assert len(patch_exec._call_log) == 2

    async def test_ensure_session_alive_failure(self, patch_exec):
        """ensure_session_alive raises ZellijSessionMissing on non-zero exit."""
        proc = FakeProc(
            returncode=1,
            stdout_data=b"",
            stderr_data=b"permission denied",
        )
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        with pytest.raises(ZellijSessionMissing, match="permission denied"):
            await mgr.ensure_session_alive()

    async def test_list_panes_success(self, patch_exec, load_fixture):
        """list_panes parses JSON and returns flattened pane list."""
        json_output = json.dumps(load_fixture).encode()
        proc = FakeProc(returncode=0, stdout_data=json_output, stderr_data=b"")
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        panes = await mgr.list_panes()

        # Should return a list of ZellijPaneInfo
        assert isinstance(panes, list)
        assert len(panes) > 0
        for pane in panes:
            assert isinstance(pane, dict)
            assert "id" in pane
            assert "pwd" in pane
            assert "terminal_command" in pane
            assert "exited" in pane

    async def test_list_panes_malformed_json(self, patch_exec):
        """list_panes raises ZellijError on malformed JSON."""
        proc = FakeProc(
            returncode=0,
            stdout_data=b"not valid json",
            stderr_data=b"",
        )
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        with pytest.raises(ZellijError, match="malformed"):
            await mgr.list_panes()

    async def test_write_to_pane_single_line(self, patch_exec):
        """write_to_pane with no newline issues only write-chars."""
        proc1 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc1)

        mgr = ZellijManager()
        await mgr.write_to_pane("pane-1", "hello")

        # Should have one call: write-chars
        assert len(patch_exec._call_log) == 1
        argv, _ = patch_exec._call_log[0]
        assert argv == (
            "zellij",
            "--session",
            "bridge",
            "action",
            "write-chars",
            "--pane-id",
            "pane-1",
            "hello",
        )

    async def test_write_to_pane_with_newline(self, patch_exec):
        """write_to_pane with newline issues write-chars then send-keys."""
        proc1 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        proc2 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc1)
        patch_exec._queue.append(proc2)

        mgr = ZellijManager()
        await mgr.write_to_pane("pane-1", "hello\n")

        # Should have two calls: write-chars, send-keys
        assert len(patch_exec._call_log) == 2
        argv1, _ = patch_exec._call_log[0]
        argv2, _ = patch_exec._call_log[1]
        assert argv1 == (
            "zellij",
            "--session",
            "bridge",
            "action",
            "write-chars",
            "--pane-id",
            "pane-1",
            "hello",
        )
        assert argv2 == (
            "zellij",
            "--session",
            "bridge",
            "action",
            "send-keys",
            "--pane-id",
            "pane-1",
            "--",
            "Enter",
        )

    async def test_write_to_pane_multiple_lines(self, patch_exec):
        """write_to_pane with multiple lines issues correct sequence."""
        procs = [FakeProc(returncode=0, stdout_data=b"", stderr_data=b"") for _ in range(4)]
        for p in procs:
            patch_exec._queue.append(p)

        mgr = ZellijManager()
        await mgr.write_to_pane("pane-1", "hello\nworld\n")

        # Should have 4 calls: write-chars, send-keys, write-chars, send-keys
        assert len(patch_exec._call_log) == 4
        assert patch_exec._call_log[0][0][4] == "write-chars"
        assert patch_exec._call_log[0][0][7] == "hello"
        assert patch_exec._call_log[1][0][4] == "send-keys"
        assert patch_exec._call_log[2][0][4] == "write-chars"
        assert patch_exec._call_log[2][0][7] == "world"
        assert patch_exec._call_log[3][0][4] == "send-keys"

    async def test_close_pane_success(self, patch_exec):
        """close_pane issues close-pane action."""
        proc = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        await mgr.close_pane("pane-1")

        assert len(patch_exec._call_log) == 1
        argv, _ = patch_exec._call_log[0]
        assert argv == (
            "zellij",
            "--session",
            "bridge",
            "action",
            "close-pane",
            "--pane-id",
            "pane-1",
        )

    async def test_close_pane_idempotent(self, patch_exec):
        """close_pane swallows non-zero exit (pane already gone)."""
        proc = FakeProc(returncode=1, stdout_data=b"", stderr_data=b"pane not found")
        patch_exec._queue.append(proc)

        mgr = ZellijManager()
        # Should not raise
        await mgr.close_pane("pane-1")

    async def test_spawn_task_success(self, patch_exec, load_fixture):
        """spawn_task snapshots, runs, polls, and returns new pane id."""
        # First call: ensure_session_alive
        proc_ensure = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_ensure)

        # Second call: list-panes (before spawn)
        list_panes_before = {
            "tabs": [{"panes": [{"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False}]}]
        }
        proc_list_before = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_panes_before).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_before)

        # Third call: run (spawn)
        proc_run = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_run)

        # Fourth call: list-panes (poll) - now has new pane
        list_panes_after = {
            "tabs": [
                {
                    "panes": [
                        {"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False},
                        {
                            "id": "terminal_1",
                            "pwd": "/tmp",
                            "terminal_command": "claude",
                            "exited": False,
                        },
                    ]
                }
            ]
        }
        proc_list_after = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_panes_after).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_after)

        mgr = ZellijManager()
        pane_id = await mgr.spawn_task("/tmp", {"FOO": "bar"}, "cc-abc123")

        # Should return the new pane id
        assert pane_id == "terminal_1"

        # Check calls
        assert len(patch_exec._call_log) >= 4
        # Third call should be the run command with env
        argv_run, kwargs_run = patch_exec._call_log[2]
        assert argv_run[0:4] == ("zellij", "--session", "bridge", "run")
        assert "--cwd" in argv_run
        assert "/tmp" in argv_run
        assert "--name" in argv_run
        assert "cc-abc123" in argv_run
        assert "--" in argv_run
        assert "claude" in argv_run
        assert "env" in kwargs_run
        assert kwargs_run["env"]["FOO"] == "bar"

    async def test_spawn_task_timeout(self, patch_exec):
        """spawn_task raises ZellijSpawnError on poll timeout."""
        # First call: ensure_session_alive
        proc_ensure = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_ensure)

        # Second call: list-panes (before spawn)
        list_panes_before = {
            "tabs": [{"panes": [{"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False}]}]
        }
        proc_list_before = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_panes_before).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_before)

        # Third call: run (spawn)
        proc_run = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_run)

        # Subsequent list-panes calls always return same (no new pane)
        for _ in range(100):  # Many polls
            proc_list = FakeProc(
                returncode=0,
                stdout_data=json.dumps(list_panes_before).encode(),
                stderr_data=b"",
            )
            patch_exec._queue.append(proc_list)

        # Create manager with short poll timeout for this test
        mgr = ZellijManager(poll_timeout=0.2)
        with pytest.raises(ZellijSpawnError, match="pane not visible"):
            await mgr.spawn_task("/tmp", {}, "cc-abc123")

    async def test_spawn_task_concurrent_serialized(self, patch_exec):
        """Concurrent spawn_task calls are serialized by the lock; each gets the right pane."""
        # Setup for first spawn
        proc_ensure1 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_ensure1)

        list_before1 = {
            "tabs": [{"panes": [{"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False}]}]
        }
        proc_list_before1 = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_before1).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_before1)

        proc_run1 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_run1)

        list_after1 = {
            "tabs": [
                {
                    "panes": [
                        {"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False},
                        {
                            "id": "terminal_1",
                            "pwd": "/tmp",
                            "terminal_command": "claude",
                            "exited": False,
                        },
                    ]
                }
            ]
        }
        proc_list_after1 = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_after1).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_after1)

        # Setup for second spawn (sees terminal_1 as existing now)
        proc_ensure2 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_ensure2)

        list_before2 = {
            "tabs": [
                {
                    "panes": [
                        {"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False},
                        {
                            "id": "terminal_1",
                            "pwd": "/tmp",
                            "terminal_command": "claude",
                            "exited": False,
                        },
                    ]
                }
            ]
        }
        proc_list_before2 = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_before2).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_before2)

        proc_run2 = FakeProc(returncode=0, stdout_data=b"", stderr_data=b"")
        patch_exec._queue.append(proc_run2)

        list_after2 = {
            "tabs": [
                {
                    "panes": [
                        {"id": "terminal_0", "pwd": "/", "terminal_command": "zsh", "exited": False},
                        {
                            "id": "terminal_1",
                            "pwd": "/tmp",
                            "terminal_command": "claude",
                            "exited": False,
                        },
                        {
                            "id": "terminal_2",
                            "pwd": "/home",
                            "terminal_command": "claude",
                            "exited": False,
                        },
                    ]
                }
            ]
        }
        proc_list_after2 = FakeProc(
            returncode=0,
            stdout_data=json.dumps(list_after2).encode(),
            stderr_data=b"",
        )
        patch_exec._queue.append(proc_list_after2)

        mgr = ZellijManager()

        # Fire both concurrently; the lock ensures they don't interleave badly
        pane1, pane2 = await asyncio.gather(
            mgr.spawn_task("/tmp", {}, "cc-spawn1"),
            mgr.spawn_task("/home", {}, "cc-spawn2"),
        )

        # Each should get its own pane
        assert pane1 == "terminal_1"
        assert pane2 == "terminal_2"
        assert pane1 != pane2
