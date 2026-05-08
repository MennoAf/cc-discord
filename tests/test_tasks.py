"""Tests for Task and TaskRegistry."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite
import pytest

from bridge.state import TaskRow, open_db, upsert_task
from bridge.tasks import Task, TaskRegistry


@dataclass
class FakeBot:
    """Minimal fake Bot for testing TaskRegistry."""

    _post_calls: list[dict] = field(default_factory=list)

    async def post(self, content: str, *, thread_id: int | None = None) -> list[int]:
        """Fake post: record the call, return a fake message ID."""
        self._post_calls.append({"content": content, "thread_id": thread_id})
        return [1001]

    def get_post_calls(self) -> list[dict]:
        return self._post_calls


@pytest.fixture
async def fake_bot():
    return FakeBot()


@pytest.fixture
async def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = await aiosqlite.connect(":memory:")
    # Initialize the schema (same as state.open_db does)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            thread_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            thread_id INTEGER NOT NULL,
            zellij_pane_id TEXT,
            cwd TEXT NOT NULL,
            status TEXT NOT NULL,
            current_claude_session_id TEXT,
            current_transcript_path TEXT,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_thread_id ON tasks(thread_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(current_claude_session_id)"
    )
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_log (
            request_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            tool_input_json TEXT NOT NULL,
            decision TEXT NOT NULL,
            decision_reason TEXT NOT NULL,
            decided_at INTEGER NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_approval_log_task_id ON approval_log(task_id)"
    )
    await conn.commit()
    yield conn
    await conn.close()


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

    async def test_load_from_db(self, fake_bot, in_memory_db) -> None:
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

        registry = TaskRegistry(in_memory_db, fake_bot)
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

    async def test_get_by_task_id(self, fake_bot, in_memory_db) -> None:
        """get_by_task_id returns task or None."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot)
        await registry.load_from_db()

        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_task_id("unknown") is None

    async def test_get_by_thread_id(self, fake_bot, in_memory_db) -> None:
        """get_by_thread_id returns task or None."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot)
        await registry.load_from_db()

        task = registry.get_by_thread_id(999)
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_thread_id(888) is None

    async def test_get_by_session_id(self, fake_bot, in_memory_db) -> None:
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

        registry = TaskRegistry(in_memory_db, fake_bot)
        await registry.load_from_db()

        task = registry.get_by_session_id("sess-abc")
        assert task is not None
        assert task.task_id == "task-123"

        assert registry.get_by_session_id("unknown") is None

    async def test_handle_event_session_start_with_task(
        self, fake_bot, in_memory_db
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

        registry = TaskRegistry(in_memory_db, fake_bot)
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

        # Bot should have posted
        posts = fake_bot.get_post_calls()
        assert len(posts) == 1
        assert posts[0]["thread_id"] == 999
        assert "🟢 SessionStart" in posts[0]["content"]

    async def test_handle_event_session_start_without_task(
        self, fake_bot, in_memory_db
    ) -> None:
        """handle_event('SessionStart') without task_id is silent no-op."""
        registry = TaskRegistry(in_memory_db, fake_bot)
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
        self, fake_bot, in_memory_db
    ) -> None:
        """handle_event with unknown event name returns without raising."""
        now = 1000
        await upsert_task(
            in_memory_db, "task-123", 999, "/tmp", "running", now=now
        )

        registry = TaskRegistry(in_memory_db, fake_bot)
        await registry.load_from_db()

        # Should not raise
        body = {"hook_event_name": "Banana", "some": "data"}
        await registry.handle_event("Banana", body)

        # No post
        posts = fake_bot.get_post_calls()
        assert len(posts) == 0

    async def test_handlers_dict_has_all_event_names(
        self, fake_bot, in_memory_db
    ) -> None:
        """_HANDLERS dict contains all expected event names."""
        registry = TaskRegistry(in_memory_db, fake_bot)

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
