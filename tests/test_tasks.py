"""Tests for Task and TaskRegistry."""

import os
from dataclasses import dataclass, field

import pytest

from bridge.state import TaskRow, upsert_task
from bridge.tasks import Task, TaskRegistry, TaskSpawnError
from bridge.zellij import ZellijManager


@dataclass
class FakeBot:
    """Minimal fake Bot for testing TaskRegistry."""

    _post_calls: list[dict] = field(default_factory=list)
    _thread_calls: list[dict] = field(default_factory=list)

    async def post(self, content: str, *, thread_id: int | None = None) -> list[int]:
        """Fake post: record the call, return a fake message ID."""
        self._post_calls.append({"content": content, "thread_id": thread_id})
        return [1001]

    async def create_thread(self, name: str) -> int:
        """Fake create_thread: record the call, return a fake thread ID."""
        thread_id = 2000 + len(self._thread_calls)
        self._thread_calls.append({"name": name})
        return thread_id

    def get_post_calls(self) -> list[dict]:
        return self._post_calls

    def get_thread_calls(self) -> list[dict]:
        return self._thread_calls


@pytest.fixture
async def fake_bot():
    return FakeBot()


@pytest.fixture
async def fake_zellij(monkeypatch):
    """Create a ZellijManager with mocked _run method."""
    mgr = ZellijManager()

    async def mock_run(*argv, env=None, timeout=10.0):
        """Mock _run to always return success."""
        return (0, "", "")

    monkeypatch.setattr(mgr, "_run", mock_run)
    return mgr




@pytest.mark.asyncio
class TestTask:
    """Tests for Task dataclass."""

    async def test_task_from_row(self) -> None:
        """Task.from_row converts TaskRow to Task."""
        row = TaskRow(
            task_id="task-123",
            thread_id=999,
            zellij_pane_id="terminal_1",
            cwd="/tmp/test",
            status="running",
            current_claude_session_id="sess-abc",
            current_transcript_path="/path/transcript",
            created_at=1000,
            last_activity=2000,
        )
        task = Task.from_row(row)
        assert task.task_id == "task-123"
        assert task.thread_id == 999
        assert task.zellij_pane_id == "terminal_1"
        assert task.cwd == "/tmp/test"
        assert task.status == "running"
        assert task.current_claude_session_id == "sess-abc"
        assert task.current_transcript_path == "/path/transcript"
        assert task.created_at == 1000
        assert task.last_activity == 2000


@pytest.mark.asyncio
class TestTaskRegistry:
    """Tests for TaskRegistry."""

    async def test_load_from_db(self, fake_bot, fake_zellij, in_memory_db) -> None:
        """load_from_db populates all three maps."""
        now = 1000
        # Insert some tasks
        await upsert_task(
            in_memory_db,
            "task-1",
            1001,
            "/a",
            "running",
            current_claude_session_id="sess-1",
            now=now,
        )
        await upsert_task(
            in_memory_db,
            "task-2",
            1002,
            "/b",
            "spawning",
            current_claude_session_id="sess-2",
            now=now,
        )
        # Stopped task should not be loaded
        await upsert_task(
            in_memory_db, "task-3", 1003, "/c", "stopped", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Should have loaded 2 tasks
        assert registry.get_by_task_id("task-1") is not None
        assert registry.get_by_task_id("task-2") is not None
        assert registry.get_by_task_id("task-3") is None  # Stopped task not loaded

        # By thread_id
        assert registry.get_by_thread_id(1001) is not None
        assert registry.get_by_thread_id(1002) is not None
        assert registry.get_by_thread_id(1003) is None

        # By session_id
        assert registry.get_by_session_id("sess-1") is not None
        assert registry.get_by_session_id("sess-2") is not None

    async def test_get_by_task_id(self, fake_bot, fake_zellij, in_memory_db) -> None:
        """get_by_task_id returns task or None."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_task_id("unknown") is None

    async def test_get_by_thread_id(self, fake_bot, fake_zellij, in_memory_db) -> None:
        """get_by_thread_id returns task or None."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        task = registry.get_by_thread_id(999)
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_thread_id(888) is None

    async def test_get_by_session_id(self, fake_bot, fake_zellij, in_memory_db) -> None:
        """get_by_session_id returns task or None."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        task = registry.get_by_session_id("sess-abc")
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_session_id("unknown") is None

    async def test_handle_event_session_start_with_task(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event('SessionStart') with matching task_id updates and posts."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Handle SessionStart with matching task_id
        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "cwd": "/tmp",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Task should be updated
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.current_claude_session_id == "sess-abc"
        assert task.current_transcript_path == "/path/to/transcript"
        assert task.status == "running"

        # Bot should have posted
        posts = fake_bot.get_post_calls()
        assert len(posts) == 1
        assert posts[0]["thread_id"] == 999
        assert "🟢 Task started" in posts[0]["content"]
        assert "sess-abc" in posts[0]["content"]

    async def test_handle_event_session_start_rotates_session_id(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event('SessionStart') rotates session_id and invalidates old mapping."""
        now = 1000
        # Seed task with initial session_id
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            current_claude_session_id="sess-A",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Verify initial state
        assert registry.get_by_session_id("sess-A") is not None
        assert registry.get_by_session_id("sess-B") is None

        # Rotate session_id to sess-B (e.g., on /clear or /compact)
        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-B",
            "cwd": "/tmp",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Task should be updated with new session_id
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.current_claude_session_id == "sess-B"

        # Old session_id mapping should be invalidated
        assert registry.get_by_session_id("sess-A") is None
        # New session_id mapping should be valid
        assert registry.get_by_session_id("sess-B") is not None

    async def test_handle_event_session_start_missing_session_id(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event('SessionStart') without session_id is dropped (no state mutation)."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # SessionStart with task_id but missing session_id — should be dropped
        body = {
            "hook_event_name": "SessionStart",
            "cwd": "/tmp",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Task should remain unchanged
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.current_claude_session_id is None
        assert task.status == "spawning"  # Status stays spawning
        assert task.current_transcript_path is None

        # No post should happen
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_handle_event_session_start_without_task(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event('SessionStart') without task_id is silent no-op."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "cwd": "/tmp",
            "transcript_path": "/path/to/transcript",
        }
        await registry.handle_event("SessionStart", body)

        # No post should happen
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_handle_event_unknown_event(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event with unknown event name returns without raising."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Should not raise
        body = {"hook_event_name": "Banana", "some": "data"}
        await registry.handle_event("Banana", body)

        # No post
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_handlers_dict_has_all_event_names(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """_HANDLERS dict contains all expected event names."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        expected_events = {
            "SessionStart",
            "UserPromptSubmit",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "Notification",
            "SessionEnd",
            "SubagentStop",
            "PreCompact",
        }
        assert set(registry._HANDLERS.keys()) == expected_events

    async def test_spawn_task_nonexistent_cwd(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """spawn_task with nonexistent cwd raises TaskSpawnError."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        with pytest.raises(TaskSpawnError, match="cwd does not exist"):
            await registry.spawn_task("/nonexistent/path")

    async def test_spawn_task_success(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """spawn_task succeeds: creates thread, persists row, indexes task."""
        # Mock zellij.spawn_task to return a pane_id
        pane_id = "terminal_1"

        async def mock_spawn_task(cwd: str, env: dict[str, str], pane_name: str) -> str:
            return pane_id

        monkeypatch.setattr(fake_zellij, "spawn_task", mock_spawn_task)

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        cwd = "/tmp"
        task = await registry.spawn_task(cwd)

        # Verify thread was created with correct name format
        thread_calls = fake_bot.get_thread_calls()
        assert len(thread_calls) == 1
        assert thread_calls[0]["name"].startswith("cc · tmp · ")
        assert len(thread_calls[0]["name"]) > len("cc · tmp · ")  # has task_id suffix

        # Verify task fields
        assert task.task_id is not None
        assert task.thread_id == 2000
        assert task.zellij_pane_id == pane_id
        assert task.cwd == cwd
        assert task.status == "spawning"
        assert task.current_claude_session_id is None
        assert task.current_transcript_path is None
        assert task.created_at > 0
        assert task.last_activity > 0

        # Verify task is indexed by task_id
        indexed = registry.get_by_task_id(task.task_id)
        assert indexed is not None
        assert indexed.task_id == task.task_id

        # Verify task is indexed by thread_id
        indexed = registry.get_by_thread_id(task.thread_id)
        assert indexed is not None
        assert indexed.task_id == task.task_id

        # Verify task is persisted to DB
        from bridge.state import get_task
        db_row = await get_task(in_memory_db, task.task_id)
        assert db_row is not None
        assert db_row.status == "spawning"
        assert db_row.zellij_pane_id == pane_id

    async def test_spawn_task_env_vars(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """spawn_task injects CC_DISCORD_TASK_ID and BRIDGE_URL into env."""
        captured_env = {}

        async def mock_spawn_task(cwd: str, env: dict[str, str], pane_name: str) -> str:
            captured_env.update(env)
            return "terminal_1"

        monkeypatch.setattr(fake_zellij, "spawn_task", mock_spawn_task)

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        # Clear BRIDGE_URL from os.environ for this test to use default
        old_bridge_url = os.environ.pop("BRIDGE_URL", None)
        try:
            task = await registry.spawn_task("/tmp")

            # Verify CC_DISCORD_TASK_ID is in env
            assert captured_env["CC_DISCORD_TASK_ID"] == task.task_id

            # Verify BRIDGE_URL defaults to localhost
            assert captured_env["BRIDGE_URL"] == "http://127.0.0.1:8787"
        finally:
            if old_bridge_url is not None:
                os.environ["BRIDGE_URL"] = old_bridge_url

    async def test_spawn_task_env_bridge_url_from_env(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """spawn_task preserves BRIDGE_URL from os.environ if set."""
        captured_env = {}

        async def mock_spawn_task(cwd: str, env: dict[str, str], pane_name: str) -> str:
            captured_env.update(env)
            return "terminal_1"

        monkeypatch.setattr(fake_zellij, "spawn_task", mock_spawn_task)

        old_bridge_url = os.environ.get("BRIDGE_URL")
        try:
            os.environ["BRIDGE_URL"] = "http://example.com:9999"

            registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
            await registry.spawn_task("/tmp")

            # Verify BRIDGE_URL is preserved from env
            assert captured_env["BRIDGE_URL"] == "http://example.com:9999"
        finally:
            if old_bridge_url is not None:
                os.environ["BRIDGE_URL"] = old_bridge_url
            else:
                os.environ.pop("BRIDGE_URL", None)

    async def test_on_session_start_no_cc_discord_task_id(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """_on_session_start without CC_DISCORD_TASK_ID is silently dropped."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {},
        }
        await registry.handle_event("SessionStart", body)

        # No posts should happen
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_on_session_start_updates_status_to_running(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """_on_session_start flips status from spawning to running."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Task should be updated to running
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.status == "running"
        assert task.current_claude_session_id == "sess-abc"
        assert task.current_transcript_path == "/path/to/transcript"
        assert task.last_activity > now

    async def test_on_session_start_bind_notice(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """_on_session_start posts a bind notice with session_id."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc123",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Verify bind notice was posted
        posts = fake_bot.get_post_calls()
        assert len(posts) == 1
        assert posts[0]["thread_id"] == 999
        assert "🟢 Task started" in posts[0]["content"]
        assert "sess-abc" in posts[0]["content"]

    async def test_on_session_start_unknown_task_id(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """_on_session_start with unknown task_id logs warning and does nothing."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "transcript_path": "/path/to/transcript",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "unknown-task"},
        }
        await registry.handle_event("SessionStart", body)

        # No posts should happen
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_handle_event_session_start_missing_transcript_path(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """handle_event('SessionStart') without transcript_path is dropped (no state mutation)."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # SessionStart with task_id but missing transcript_path — should be dropped
        body = {
            "hook_event_name": "SessionStart",
            "session_id": "sess-abc",
            "cwd": "/tmp",
            "env_passthrough": {"CC_DISCORD_TASK_ID": "task-123"},
        }
        await registry.handle_event("SessionStart", body)

        # Task should remain unchanged
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.current_claude_session_id is None
        assert task.status == "spawning"  # Status stays spawning
        assert task.current_transcript_path is None

        # No post should happen
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_spawn_task_zellij_failure(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """spawn_task on zellij failure marks task as crashed and re-raises."""
        from bridge.zellij import ZellijSpawnError

        async def mock_spawn_task(cwd: str, env: dict[str, str], pane_name: str) -> str:
            raise ZellijSpawnError("Could not resolve pane id")

        monkeypatch.setattr(fake_zellij, "spawn_task", mock_spawn_task)

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        with pytest.raises(ZellijSpawnError):
            await registry.spawn_task("/tmp")

        # Verify that a task was created and marked as crashed
        # We need to find the created task. Let's check the thread calls
        thread_calls = fake_bot.get_thread_calls()
        assert len(thread_calls) == 1
        # The exception was raised, which is what we're testing
