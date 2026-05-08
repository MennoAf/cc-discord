"""Tests for ZellijManager (tab-name-based addressing on zellij ≥ 0.43)."""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from bridge import zellij as zellij_module
from bridge.zellij import (
    ZellijError,
    ZellijManager,
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
        return self.returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return (self.stdout_data, self.stderr_data)


@pytest.fixture
def patch_exec(monkeypatch):
    """Monkeypatch asyncio.create_subprocess_exec with a FakeProc queue."""
    queue: list[FakeProc] = []
    call_log: list[tuple] = []

    async def _create_subprocess_exec(*argv: str, **kwargs: Any) -> FakeProc:
        call_log.append((argv, kwargs))
        if not queue:
            raise AssertionError(f"Unexpected zellij call: {argv}")
        return queue.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create_subprocess_exec)
    _create_subprocess_exec._queue = queue  # type: ignore
    _create_subprocess_exec._call_log = call_log  # type: ignore
    return _create_subprocess_exec


@pytest.fixture(autouse=True)
def fixed_session_name(monkeypatch):
    """Pin SESSION_NAME to 'bridge' and clear ZELLIJ_SESSION_NAME so tests don't
    accidentally trigger the colocated-session shortcut when run from a shell
    that's itself inside zellij."""
    monkeypatch.setattr(zellij_module, "SESSION_NAME", "bridge")
    monkeypatch.delenv("ZELLIJ_SESSION_NAME", raising=False)


@pytest.mark.asyncio
class TestZellijManager:
    async def test_ensure_session_alive_success(self, patch_exec):
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))
        mgr = ZellijManager()
        await mgr.ensure_session_alive()
        argv, _ = patch_exec._call_log[0]
        assert argv == ("zellij", "attach", "--create-background", "bridge")

    async def test_ensure_session_alive_idempotent(self, patch_exec):
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))
        mgr = ZellijManager()
        await mgr.ensure_session_alive()
        await mgr.ensure_session_alive()
        assert len(patch_exec._call_log) == 2

    async def test_ensure_session_alive_failure(self, patch_exec):
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"permission denied")
        )
        mgr = ZellijManager()
        with pytest.raises(ZellijSessionMissing, match="permission denied"):
            await mgr.ensure_session_alive()

    async def test_ensure_session_alive_tolerates_already_exists(self, patch_exec):
        """zellij ≥ 0.43 returns non-zero with 'Session already exists' on second attach."""
        patch_exec._queue.append(
            FakeProc(returncode=2, stdout_data=b"", stderr_data=b"Session already exists")
        )
        mgr = ZellijManager()
        await mgr.ensure_session_alive()  # should not raise

    async def test_ensure_session_alive_skips_attach_when_colocated(
        self, patch_exec, monkeypatch
    ):
        """When ZELLIJ_SESSION_NAME matches the target session, skip attach
        (zellij panics on self-attach in that case)."""
        monkeypatch.setenv("ZELLIJ_SESSION_NAME", "bridge")
        mgr = ZellijManager()
        await mgr.ensure_session_alive()
        assert patch_exec._call_log == []

    async def test_list_panes_filters_cc_prefix(self, patch_exec):
        """list_panes runs query-tab-names and returns only cc- prefixed tabs."""
        patch_exec._queue.append(
            FakeProc(
                returncode=0,
                stdout_data=b"Tab #1\ncc-aa429dc4\nTab #2\ncc-deadbeef\n",
                stderr_data=b"",
            )
        )
        mgr = ZellijManager()
        panes = await mgr.list_panes()
        assert panes == [
            {"id": "cc-aa429dc4", "exited": False},
            {"id": "cc-deadbeef", "exited": False},
        ]
        argv, _ = patch_exec._call_log[0]
        assert argv == ("zellij", "--session", "bridge", "action", "query-tab-names")

    async def test_list_panes_failure(self, patch_exec):
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"session not found")
        )
        mgr = ZellijManager()
        with pytest.raises(ZellijError, match="query-tab-names failed"):
            await mgr.list_panes()

    async def test_write_to_pane_single_line(self, patch_exec):
        """No newline → focus tab + one byte-encoded write, no Enter."""
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # go-to-tab-name
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # action write

        mgr = ZellijManager()
        await mgr.write_to_pane("cc-aa429dc4", "hello")

        assert len(patch_exec._call_log) == 2
        assert patch_exec._call_log[0][0] == (
            "zellij", "--session", "bridge", "action", "go-to-tab-name", "cc-aa429dc4"
        )
        assert patch_exec._call_log[1][0] == (
            "zellij", "--session", "bridge", "action", "write",
            "104", "101", "108", "108", "111",
        )

    async def test_write_to_pane_with_newline(self, patch_exec):
        """Trailing newline → focus + byte-encoded write + write 13 (Enter)."""
        for _ in range(3):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        await mgr.write_to_pane("cc-aa429dc4", "hello\n")

        assert len(patch_exec._call_log) == 3
        assert patch_exec._call_log[0][0][4:] == ("go-to-tab-name", "cc-aa429dc4")
        assert patch_exec._call_log[1][0][4:] == ("write", "104", "101", "108", "108", "111")
        assert patch_exec._call_log[2][0][4:] == ("write", "13")

    async def test_write_to_pane_multiple_lines(self, patch_exec):
        """Multi-line input is wrapped in bracketed paste so embedded newlines
        don't submit early. Trailing newline still submits at the end."""
        for _ in range(7):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        await mgr.write_to_pane("cc-aa429dc4", "hello\nworld\n")

        assert len(patch_exec._call_log) == 7
        assert patch_exec._call_log[0][0][4:] == ("go-to-tab-name", "cc-aa429dc4")
        # ESC [ 2 0 0 ~  — begin bracketed paste
        assert patch_exec._call_log[1][0][4:] == ("write", "27", "91", "50", "48", "48", "126")
        assert patch_exec._call_log[2][0][4:] == ("write", "104", "101", "108", "108", "111")
        # LF between segments, inside the paste block (becomes content newline)
        assert patch_exec._call_log[3][0][4:] == ("write", "10")
        assert patch_exec._call_log[4][0][4:] == ("write", "119", "111", "114", "108", "100")
        # ESC [ 2 0 1 ~  — end bracketed paste
        assert patch_exec._call_log[5][0][4:] == ("write", "27", "91", "50", "48", "49", "126")
        # Trailing CR submits the prompt
        assert patch_exec._call_log[6][0][4:] == ("write", "13")

    async def test_write_to_pane_multi_line_no_trailing_newline(self, patch_exec):
        """Multi-line without trailing newline: paste block, no final submit."""
        for _ in range(6):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        await mgr.write_to_pane("cc-aa429dc4", "hello\nworld")

        assert len(patch_exec._call_log) == 6
        assert patch_exec._call_log[0][0][4:] == ("go-to-tab-name", "cc-aa429dc4")
        assert patch_exec._call_log[1][0][4:] == ("write", "27", "91", "50", "48", "48", "126")
        assert patch_exec._call_log[2][0][4:] == ("write", "104", "101", "108", "108", "111")
        assert patch_exec._call_log[3][0][4:] == ("write", "10")
        assert patch_exec._call_log[4][0][4:] == ("write", "119", "111", "114", "108", "100")
        assert patch_exec._call_log[5][0][4:] == ("write", "27", "91", "50", "48", "49", "126")

    async def test_write_to_pane_focus_failure_raises(self, patch_exec):
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"no such tab")
        )
        mgr = ZellijManager()
        with pytest.raises(ZellijError, match="go-to-tab-name"):
            await mgr.write_to_pane("cc-missing", "hello")

    async def test_close_pane_success(self, patch_exec):
        """close_pane focuses the tab then issues close-tab."""
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # go-to-tab-name
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # close-tab

        mgr = ZellijManager()
        await mgr.close_pane("cc-aa429dc4")

        assert len(patch_exec._call_log) == 2
        assert patch_exec._call_log[0][0][4:] == ("go-to-tab-name", "cc-aa429dc4")
        assert patch_exec._call_log[1][0][4:] == ("close-tab",)

    async def test_close_pane_idempotent_when_tab_missing(self, patch_exec):
        """close_pane is silent when the tab is already gone."""
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"no such tab")
        )
        mgr = ZellijManager()
        await mgr.close_pane("cc-aa429dc4")  # should not raise; only one call (no close-tab issued)
        assert len(patch_exec._call_log) == 1

    async def test_spawn_task_success(self, patch_exec):
        """Three subprocess calls: attach, new-tab, run; returns the tab name.

        env is injected via the `env(1)` prefix because zellij's client-server
        model means the subprocess env we'd otherwise pass is invisible to the
        spawned claude.
        """
        for _ in range(3):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        tab = await mgr.spawn_task("/tmp", {"FOO": "bar"}, "cc-aa429dc4")

        assert tab == "cc-aa429dc4"
        assert len(patch_exec._call_log) == 3
        assert patch_exec._call_log[0][0] == (
            "zellij", "attach", "--create-background", "bridge"
        )
        assert patch_exec._call_log[1][0] == (
            "zellij", "--session", "bridge", "action", "new-tab",
            "--name", "cc-aa429dc4", "--cwd", "/tmp",
        )
        run_argv, _ = patch_exec._call_log[2]
        assert run_argv == (
            "zellij", "--session", "bridge", "run",
            "--close-on-exit", "--cwd", "/tmp", "--",
            "env", "FOO=bar", "claude",
        )

    async def test_spawn_task_extra_argv(self, patch_exec):
        """spawn_task appends extra_argv after `claude` (used for --resume)."""
        for _ in range(3):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        await mgr.spawn_task(
            "/tmp", {}, "cc-aa429dc4", extra_argv=["--resume", "sess-xyz"]
        )

        run_argv, _ = patch_exec._call_log[2]
        assert run_argv[-3:] == ("claude", "--resume", "sess-xyz")
        # With empty env, the prefix is just `env claude ...`.
        assert "env" in run_argv
        assert run_argv.index("env") < run_argv.index("claude")

    async def test_spawn_task_session_already_exists_is_success(self, patch_exec):
        """attach returning 'Session already exists' is treated as success."""
        patch_exec._queue.append(
            FakeProc(returncode=2, stdout_data=b"", stderr_data=b"Session already exists")
        )
        for _ in range(2):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        tab = await mgr.spawn_task("/tmp", {}, "cc-aa429dc4")
        assert tab == "cc-aa429dc4"

    async def test_spawn_task_skips_attach_when_colocated(self, patch_exec, monkeypatch):
        """When running inside the target session, spawn_task skips the attach
        call (only new-tab + run remain)."""
        monkeypatch.setenv("ZELLIJ_SESSION_NAME", "bridge")
        for _ in range(2):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        tab = await mgr.spawn_task("/tmp", {}, "cc-aa429dc4")

        assert tab == "cc-aa429dc4"
        assert len(patch_exec._call_log) == 2
        assert patch_exec._call_log[0][0][4] == "new-tab"
        assert patch_exec._call_log[1][0][3] == "run"

    async def test_spawn_task_new_tab_failure(self, patch_exec):
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # attach
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"new-tab refused")
        )
        mgr = ZellijManager()
        with pytest.raises(ZellijSpawnError, match="new-tab"):
            await mgr.spawn_task("/tmp", {}, "cc-aa429dc4")

    async def test_spawn_task_run_failure(self, patch_exec):
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # attach
        patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))  # new-tab
        patch_exec._queue.append(
            FakeProc(returncode=1, stdout_data=b"", stderr_data=b"command not found: claude")
        )
        mgr = ZellijManager()
        with pytest.raises(ZellijSpawnError, match="run claude failed"):
            await mgr.spawn_task("/tmp", {}, "cc-aa429dc4")

    async def test_spawn_task_concurrent_serialized(self, patch_exec):
        """Concurrent spawn_task calls serialize on the session lock; each sees its own
        attach/new-tab/run sequence ungaroupbled."""
        # 3 calls per spawn × 2 spawns = 6 total
        for _ in range(6):
            patch_exec._queue.append(FakeProc(returncode=0, stdout_data=b"", stderr_data=b""))

        mgr = ZellijManager()
        tab1, tab2 = await asyncio.gather(
            mgr.spawn_task("/tmp", {}, "cc-spawn1"),
            mgr.spawn_task("/home", {}, "cc-spawn2"),
        )
        assert {tab1, tab2} == {"cc-spawn1", "cc-spawn2"}

        # The 6 calls should appear as two interleaved-but-not-tangled groups.
        # Check that within each spawn's group, attach precedes new-tab precedes run.
        log = patch_exec._call_log
        # Find the new-tab calls and verify each is preceded by an attach
        # within the same lock-held window.
        for i, (argv, _) in enumerate(log):
            if "new-tab" in argv:
                # Previous call in log should be attach
                prev_argv = log[i - 1][0]
                assert prev_argv[1] == "attach"
                # Next call should be run for the same tab name
                next_argv = log[i + 1][0]
                assert "run" in next_argv
                tab_name = argv[argv.index("--name") + 1]
                assert tab_name in {"cc-spawn1", "cc-spawn2"}
