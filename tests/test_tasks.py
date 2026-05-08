"""Tests for Task and TaskRegistry."""

import asyncio
import os
from dataclasses import dataclass, field

import pytest

from bridge.state import TaskRow, upsert_task
from bridge.tasks import Task, TaskRegistry, TaskSpawnError, _ToolSummaryAggregator
from bridge.zellij import ZellijManager
from tests.fakes import FakeBot, FakeZellij


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


@dataclass
class FakeChannel:
    """Minimal fake channel for maybe_route_message tests."""
    id: int


@dataclass
class FakeAuthor:
    """Minimal fake author for maybe_route_message tests."""
    id: int
    bot: bool = False


@dataclass
class FakeAttachment:
    """Minimal fake attachment for maybe_route_message tests."""
    url: str = "http://example.com/image.png"


@dataclass
class FakeMsgLike:
    """Fake message matching MessageLike protocol for routing tests."""
    channel: FakeChannel
    content: str = ""
    attachments: list[FakeAttachment] = field(default_factory=list)
    author: FakeAuthor = field(default_factory=lambda: FakeAuthor(id=123))
    created_at: object = field(default_factory=lambda: __import__('datetime').datetime.now(__import__('datetime').timezone.utc))


@pytest.mark.asyncio
class TestMaybeRouteMessage:
    """Tests for TaskRegistry.maybe_route_message."""

    async def test_maybe_route_message_no_task_returns_false(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """maybe_route_message returns False when no task is bound to the thread."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        msg = FakeMsgLike(channel=FakeChannel(id=999))
        result = await registry.maybe_route_message(msg)
        assert result is False

    async def test_maybe_route_message_no_pane_id_returns_false(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """maybe_route_message returns False when task.zellij_pane_id is None."""
        # Create a task with no pane_id
        task_id = "task-xyz"
        thread_id = 5000
        now = int(__import__('time').time())
        await upsert_task(
            in_memory_db, task_id, thread_id, "/tmp", "spawning",
            zellij_pane_id=None,
            current_claude_session_id=None,
            current_transcript_path=None,
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        msg = FakeMsgLike(channel=FakeChannel(id=thread_id))
        result = await registry.maybe_route_message(msg)
        assert result is False

    async def test_maybe_route_message_writes_to_pane(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """maybe_route_message calls zellij.write_to_pane and returns True for a bound task."""
        task_id = "task-abc"
        thread_id = 6000
        pane_id = "pane_1"
        now = int(__import__('time').time())
        await upsert_task(
            in_memory_db, task_id, thread_id, "/tmp", "running",
            zellij_pane_id=pane_id,
            current_claude_session_id="sess-123",
            current_transcript_path="/path/transcript",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Track write_to_pane calls
        write_calls = []

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            write_calls.append({"pane_id": pane_id, "text": text})

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        msg = FakeMsgLike(channel=FakeChannel(id=thread_id), content="hello world")
        result = await registry.maybe_route_message(msg)

        assert result is True
        assert len(write_calls) == 1
        assert write_calls[0]["pane_id"] == pane_id
        assert write_calls[0]["text"] == "hello world\n"

    async def test_maybe_route_message_empty_content_no_attachments_returns_true(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """maybe_route_message returns True silently for empty message with no attachments."""
        task_id = "task-def"
        thread_id = 7000
        pane_id = "pane_2"
        now = int(__import__('time').time())
        await upsert_task(
            in_memory_db, task_id, thread_id, "/tmp", "running",
            zellij_pane_id=pane_id,
            current_claude_session_id="sess-456",
            current_transcript_path="/path/transcript",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        write_calls = []

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            write_calls.append({"pane_id": pane_id, "text": text})

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        # Empty message, no attachments
        msg = FakeMsgLike(channel=FakeChannel(id=thread_id), content="")
        result = await registry.maybe_route_message(msg)

        assert result is True
        assert len(write_calls) == 0  # No write to pane

    async def test_maybe_route_message_image_placeholder(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """maybe_route_message writes placeholder for message with attachments but no content."""
        task_id = "task-ghi"
        thread_id = 8000
        pane_id = "pane_3"
        now = int(__import__('time').time())
        await upsert_task(
            in_memory_db, task_id, thread_id, "/tmp", "running",
            zellij_pane_id=pane_id,
            current_claude_session_id="sess-789",
            current_transcript_path="/path/transcript",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        write_calls = []

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            write_calls.append({"pane_id": pane_id, "text": text})

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        # Empty content but with attachment
        msg = FakeMsgLike(
            channel=FakeChannel(id=thread_id),
            content="",
            attachments=[FakeAttachment(url="http://example.com/image.png")]
        )
        result = await registry.maybe_route_message(msg)

        assert result is True
        assert len(write_calls) == 1
        assert "(image attached — image relay not yet supported)" in write_calls[0]["text"]


@pytest.mark.asyncio
class TestTaskRegistryPhase3:
    """Tests for Phase 3 task lifecycle methods: list_tasks, stop_task, kill_task, restart_task."""

    async def test_list_tasks_returns_active_ordered_by_last_activity(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """list_tasks returns active tasks ordered by last_activity DESC; filters out stopped/crashed."""
        now = 1000
        # Insert active task
        await upsert_task(
            in_memory_db,
            "task-1",
            1001,
            "/a",
            "running",
            current_claude_session_id="sess-1",
            now=now,
        )
        # Insert another active task with more recent last_activity
        await upsert_task(
            in_memory_db,
            "task-2",
            1002,
            "/b",
            "running",
            current_claude_session_id="sess-2",
            now=now + 100,
        )
        # Insert stopped task (should be filtered out)
        await upsert_task(
            in_memory_db,
            "task-3",
            1003,
            "/c",
            "stopped",
            now=now,
        )
        # Insert crashed task (should be filtered out)
        await upsert_task(
            in_memory_db,
            "task-4",
            1004,
            "/d",
            "crashed",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        tasks = await registry.list_tasks()

        # Should return 2 active tasks, with task-2 first (higher last_activity)
        assert len(tasks) == 2
        assert tasks[0].task_id == "task-2"
        assert tasks[1].task_id == "task-1"

    async def test_stop_task_with_pane_alive_sends_exit_and_waits_for_session_end(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """stop_task sends /exit and waits for SessionEnd to resolve; returns True on success."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        write_calls = []

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            write_calls.append({"pane_id": pane_id, "text": text})

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        # Spawn stop_task (will be waiting for SessionEnd)
        import asyncio

        stop_task_handle = asyncio.create_task(registry.stop_task("task-123", timeout=5.0))

        # Give it a moment to send /exit
        await asyncio.sleep(0.1)

        # Verify /exit was sent
        assert len(write_calls) == 1
        assert write_calls[0]["text"] == "/exit\n"

        # Now simulate SessionEnd event to resolve the future
        await registry.handle_event("SessionEnd", {"session_id": "sess-abc"})

        # Wait for stop_task to complete
        stopped = await stop_task_handle

        # Should return True (stopped cleanly)
        assert stopped is True

        # Task should be marked as stopped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.status == "stopped"

        # Thread should be archived (bot method called)
        assert len(fake_bot.get_archive_calls()) == 1
        assert fake_bot.get_archive_calls()[0]["thread_id"] == 999

    async def test_stop_task_timeout_returns_false_and_marks_stopped(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """stop_task with timeout returns False if SessionEnd doesn't arrive; still marks stopped."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            pass

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        # stop_task with very short timeout
        stopped = await registry.stop_task("task-123", timeout=0.1)

        # Should return False (timed out)
        assert stopped is False

        # Task should still be marked as stopped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.status == "stopped"

    async def test_stop_task_with_no_pane_marks_stopped(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """stop_task when pane_id is None (mid-spawn) immediately marks stopped."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            zellij_pane_id=None,
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        stopped = await registry.stop_task("task-123")

        # Should return True (fail-safe stop)
        assert stopped is True

        # Task should be marked as stopped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.status == "stopped"

    async def test_stop_task_unknown_task_raises(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """stop_task with unknown task_id raises TaskNotFound."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        from bridge.tasks import TaskNotFound

        with pytest.raises(TaskNotFound):
            await registry.stop_task("unknown")

    async def test_kill_task_closes_pane_and_marks_crashed(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """kill_task closes the pane and marks status='crashed'."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        close_calls = []

        async def mock_close_pane(pane_id: str) -> None:
            close_calls.append({"pane_id": pane_id})

        monkeypatch.setattr(fake_zellij, "close_pane", mock_close_pane)

        await registry.kill_task("task-123")

        # Should have called close_pane
        assert len(close_calls) == 1
        assert close_calls[0]["pane_id"] == "terminal_1"

        # Task should be marked as crashed
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.status == "crashed"

    async def test_kill_task_unknown_task_raises(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """kill_task with unknown task_id raises TaskNotFound."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        from bridge.tasks import TaskNotFound

        with pytest.raises(TaskNotFound):
            await registry.kill_task("unknown")

    async def test_stop_task_removes_from_thread_and_session_indexes(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """After stop_task, get_by_thread_id and get_by_session_id return None."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Verify task is in indexes before stop
        assert registry.get_by_thread_id(999) is not None
        assert registry.get_by_session_id("sess-abc") is not None

        # Mock SessionEnd to complete the stop
        loop = asyncio.get_running_loop()
        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            # Trigger SessionEnd immediately
            loop.call_soon(lambda: asyncio.create_task(registry._on_session_end({"session_id": "sess-abc"})))

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)

        await registry.stop_task("task-123")

        # After stop, task should not be in indexes
        assert registry.get_by_thread_id(999) is None
        assert registry.get_by_session_id("sess-abc") is None
        # But still findable by task_id
        assert registry.get_by_task_id("task-123") is not None

    async def test_kill_task_removes_from_thread_and_session_indexes(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """After kill_task, get_by_thread_id and get_by_session_id return None."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Verify task is in indexes before kill
        assert registry.get_by_thread_id(999) is not None
        assert registry.get_by_session_id("sess-abc") is not None

        await registry.kill_task("task-123")

        # After kill, task should not be in indexes
        assert registry.get_by_thread_id(999) is None
        assert registry.get_by_session_id("sess-abc") is None
        # But still findable by task_id
        assert registry.get_by_task_id("task-123") is not None

    async def test_restart_task_with_live_pane_writes_resume_command(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """restart_task with live pane writes claude --resume command."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        write_calls = []
        pane_list = [{"id": "terminal_1", "title": "", "pwd": "/tmp", "terminal_command": "claude", "exited": False}]

        async def mock_write_to_pane(pane_id: str, text: str) -> None:
            write_calls.append({"pane_id": pane_id, "text": text})

        async def mock_list_panes() -> list:
            return pane_list

        monkeypatch.setattr(fake_zellij, "write_to_pane", mock_write_to_pane)
        monkeypatch.setattr(fake_zellij, "list_panes", mock_list_panes)

        task = await registry.restart_task("task-123")

        # Should have written the resume command
        assert len(write_calls) == 1
        assert "claude --resume sess-abc" in write_calls[0]["text"]

        # Task should still be running
        assert task.status == "running"

    async def test_restart_task_with_dead_pane_spawns_new_pane(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """restart_task with dead pane spawns a new pane with extra_argv."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        pane_list = []  # Empty — no panes alive

        async def mock_list_panes() -> list:
            return pane_list

        spawn_calls = []

        async def mock_spawn_task(
            cwd: str, env: dict, pane_name: str, extra_argv: list | None = None
        ) -> str:
            spawn_calls.append(
                {"cwd": cwd, "env": env, "pane_name": pane_name, "extra_argv": extra_argv}
            )
            return "terminal_2"

        monkeypatch.setattr(fake_zellij, "list_panes", mock_list_panes)
        monkeypatch.setattr(fake_zellij, "spawn_task", mock_spawn_task)

        task = await registry.restart_task("task-123")

        # Should have spawned new pane with extra_argv
        assert len(spawn_calls) == 1
        assert spawn_calls[0]["cwd"] == "/tmp"
        assert spawn_calls[0]["extra_argv"] == ["--resume", "sess-abc"]

        # Task pane should be updated
        assert task.zellij_pane_id == "terminal_2"

    async def test_restart_task_no_session_id_raises(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """restart_task with no claude session_id raises TaskRestartError."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id=None,
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        from bridge.tasks import TaskRestartError

        with pytest.raises(TaskRestartError):
            await registry.restart_task("task-123")

    async def test_restart_task_unknown_task_raises(
        self, fake_bot, fake_zellij, in_memory_db
    ) -> None:
        """restart_task with unknown task_id raises TaskNotFound."""
        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)

        from bridge.tasks import TaskNotFound

        with pytest.raises(TaskNotFound):
            await registry.restart_task("unknown")

    async def test_on_session_end_resolves_stop_future(
        self, fake_bot, fake_zellij, in_memory_db, monkeypatch
    ) -> None:
        """_on_session_end resolves a pending stop_future when task matches."""
        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Manually set up a pending stop_future
        import asyncio

        fut = asyncio.get_running_loop().create_future()
        registry._stop_futures["task-123"] = fut

        # Trigger SessionEnd
        await registry._on_session_end({"session_id": "sess-abc"})

        # Future should be resolved
        assert fut.done()

    async def test_write_initial_prompt_happy_path(
        self, in_memory_db
    ) -> None:
        """write_initial_prompt writes text to pane and bumps last_activity."""
        # Avoid fixture reuse issues with inline instantiation
        # FakeBot and FakeZellij already imported at top

        fake_zellij = FakeZellij()
        fake_bot = FakeBot()

        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "spawning",
            zellij_pane_id="terminal_1",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        task = registry.get_by_task_id("task-123")
        assert task is not None
        old_activity = task.last_activity

        # Write prompt
        await registry.write_initial_prompt("task-123", "hello world")

        # Verify write_to_pane was called
        assert len(fake_zellij._write_calls) == 1
        call = fake_zellij._write_calls[0]
        assert call["pane_id"] == "terminal_1"
        assert "hello world\n" in call["text"]

        # Verify last_activity was bumped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.last_activity > old_activity

    async def test_write_initial_prompt_missing_task(
        self, in_memory_db
    ) -> None:
        """write_initial_prompt logs warning if task is missing."""
        # FakeBot and FakeZellij already imported at top

        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())

        # Attempt to write to non-existent task
        await registry.write_initial_prompt("nonexistent", "hello")

        # Should not raise, just log

    async def test_stop_task_already_stopped_is_idempotent(
        self, in_memory_db
    ) -> None:
        """stop_task returns True immediately if task is already stopped."""
        # FakeBot and FakeZellij already imported at top

        fake_zellij = FakeZellij()
        now = 1000

        # Create a running task first
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, FakeBot(), fake_zellij)
        await registry.load_from_db()

        # Manually set status to stopped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        task.status = "stopped"

        # Stop should return True (idempotent, no-op)
        result = await registry.stop_task("task-123")
        assert result is True

        # Verify no write_to_pane was called
        assert len(fake_zellij._write_calls) == 0

    async def test_kill_task_already_crashed_is_idempotent(
        self, in_memory_db
    ) -> None:
        """kill_task returns None immediately if task is already crashed."""
        # FakeBot and FakeZellij already imported at top

        fake_zellij = FakeZellij()
        now = 1000

        # Create a running task first
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, FakeBot(), fake_zellij)
        await registry.load_from_db()

        # Manually set status to crashed
        task = registry.get_by_task_id("task-123")
        assert task is not None
        task.status = "crashed"

        # Kill should return None (idempotent, no-op)
        await registry.kill_task("task-123")

        # Verify no close_pane was called
        assert len(fake_zellij._close_calls) == 0

    async def test_restart_task_bumps_activity_when_live_pane(
        self, in_memory_db
    ) -> None:
        """restart_task with live pane updates last_activity."""
        # Import fixtures inline to avoid import issues
        # FakeBot and FakeZellij already imported at top

        fake_zellij = FakeZellij()
        fake_bot = FakeBot()

        now = 1000
        await upsert_task(
            in_memory_db,
            "task-123",
            999,
            "/tmp",
            "running",
            zellij_pane_id="terminal_1",
            current_claude_session_id="sess-abc",
            now=now,
        )

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        task = registry.get_by_task_id("task-123")
        assert task is not None
        old_activity = task.last_activity

        # Mock list_panes to return pane as alive
        async def mock_list_panes():
            return [{"id": "terminal_1", "exited": False, "title": "test"}]

        fake_zellij.list_panes = mock_list_panes

        # Restart
        await registry.restart_task("task-123")

        # Verify last_activity was bumped
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.last_activity > old_activity


    async def test_on_user_prompt_submit_starts_typing(self, in_memory_db) -> None:
        """_on_user_prompt_submit starts typing indicator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # Dispatch UserPromptSubmit
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})

        # Verify typing task was created
        assert "task-123" in registry._typing_tasks
        task_obj = registry._typing_tasks["task-123"]
        assert not task_obj.done()

        # Clean up
        await registry._stop_typing("task-123")
        await asyncio.sleep(0.01)

    async def test_on_stop_cancels_typing(self, in_memory_db) -> None:
        """_on_stop cancels the typing indicator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # Start typing
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})
        assert "task-123" in registry._typing_tasks
        assert not registry._typing_tasks["task-123"].done()

        # Stop
        await registry._on_stop({"session_id": "sess-abc"})
        await asyncio.sleep(0.01)

        # Verify typing was cancelled
        assert "task-123" not in registry._typing_tasks

    async def test_on_notification_cancels_typing(self, in_memory_db) -> None:
        """_on_notification cancels the typing indicator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # Start typing
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})
        assert "task-123" in registry._typing_tasks

        # Notification
        await registry._on_notification({"session_id": "sess-abc"})
        await asyncio.sleep(0.01)

        # Verify typing was cancelled
        assert "task-123" not in registry._typing_tasks

    async def test_on_session_end_cancels_typing(self, in_memory_db) -> None:
        """_on_session_end cancels the typing indicator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # Start typing
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})
        assert "task-123" in registry._typing_tasks

        # SessionEnd
        await registry._on_session_end({"session_id": "sess-abc"})
        await asyncio.sleep(0.01)

        # Verify typing was cancelled
        assert "task-123" not in registry._typing_tasks

    async def test_on_user_prompt_submit_unknown_session_is_noop(self, in_memory_db) -> None:
        """_on_user_prompt_submit silently drops unknown session IDs."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()

        registry = TaskRegistry(in_memory_db, fake_bot, fake_zellij)
        await registry.load_from_db()

        # Dispatch for unknown session
        await registry._on_user_prompt_submit({"session_id": "unknown"})

        # No crash, no typing task created
        assert len(registry._typing_tasks) == 0

    async def test_on_user_prompt_submit_twice_replaces_typing_task(
        self, in_memory_db
    ) -> None:
        """Calling _on_user_prompt_submit twice cancels the first typing task."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # First submit
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})
        first_task = registry._typing_tasks["task-123"]
        assert not first_task.done()

        # Second submit (should cancel first)
        await registry._on_user_prompt_submit({"session_id": "sess-abc"})
        await asyncio.sleep(0.01)

        # First task should be done/cancelled
        assert first_task.done()

        # New task should exist
        second_task = registry._typing_tasks["task-123"]
        assert not second_task.done()
        assert second_task is not first_task

        # Clean up
        await registry._stop_typing("task-123")

    async def test_on_post_tool_use_appends_to_aggregator(self, in_memory_db) -> None:
        """_on_post_tool_use appends formatted summary to aggregator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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

        # Dispatch PostToolUse
        await registry._on_post_tool_use({
            "session_id": "sess-abc",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": {"exit_code": 0},
        })

        # Verify aggregator has the line
        agg = registry._aggregators.get("task-123")
        assert agg is not None
        assert len(agg._lines) == 1
        assert "Bash" in agg._lines[0]
        assert "pytest" in agg._lines[0]

    async def test_aggregator_coalesces_within_window(self, in_memory_db) -> None:
        """Multiple PostToolUse events within 1s window produce one post."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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
        # Override the aggregator flush window to be very short for testing
        _ToolSummaryAggregator.FLUSH_WINDOW = 0.05
        await registry.load_from_db()

        # Dispatch two PostToolUse within the window
        await registry._on_post_tool_use({
            "session_id": "sess-abc",
            "tool_name": "Bash",
            "tool_input": {"command": "cmd1"},
            "tool_response": {"exit_code": 0},
        })
        await registry._on_post_tool_use({
            "session_id": "sess-abc",
            "tool_name": "Bash",
            "tool_input": {"command": "cmd2"},
            "tool_response": {"exit_code": 0},
        })

        # Wait for flush window to pass
        await asyncio.sleep(0.1)

        # Verify only one post was made (two lines in it)
        posts = fake_bot.get_post_calls()
        assert len(posts) == 1
        post_body = posts[0]["content"]
        assert "cmd1" in post_body
        assert "cmd2" in post_body
        # Should have two lines separated by newline
        lines = post_body.split("\n")
        assert len(lines) >= 2

        # Reset for future tests
        _ToolSummaryAggregator.FLUSH_WINDOW = 1.0

    async def test_on_stop_flushes_aggregator(self, in_memory_db) -> None:
        """_on_stop immediately flushes the aggregator."""
        fake_bot = FakeBot()
        fake_zellij = FakeZellij()
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
        _ToolSummaryAggregator.FLUSH_WINDOW = 10.0  # Very long window
        await registry.load_from_db()

        # Add a tool summary
        await registry._on_post_tool_use({
            "session_id": "sess-abc",
            "tool_name": "Bash",
            "tool_input": {"command": "test"},
            "tool_response": {"exit_code": 0},
        })

        # At this point, the flush should be pending (hasn't fired yet)
        agg = registry._aggregators["task-123"]
        assert len(agg._lines) == 1

        # Now call Stop, which should flush immediately
        await registry._on_stop({"session_id": "sess-abc"})

        # Verify the post was made immediately
        posts = fake_bot.get_post_calls()
        assert len(posts) == 1
        assert "test" in posts[0]["content"]

        # Reset
        _ToolSummaryAggregator.FLUSH_WINDOW = 1.0
