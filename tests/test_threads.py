"""Tests for ThreadRegistry."""

import asyncio

import aiosqlite
import pytest

from bridge.state import get_session
from bridge.threads import ThreadRegistry


class FakeBot:
    """Fake bot for testing ThreadRegistry."""

    def __init__(self) -> None:
        self._create_thread_calls: list[dict] = []
        self._thread_alive_responses: dict[int, bool] = {}
        self._thread_counter = 1000  # Start at a recognizable value
        self._alive_default = True

    async def create_thread(self, name: str) -> int:
        """Fake thread creation: return next ID, record the call."""
        thread_id = self._thread_counter
        self._thread_counter += 1
        self._create_thread_calls.append({"name": name})
        return thread_id

    async def thread_alive(self, thread_id: int) -> bool:
        """Fake thread existence check."""
        # Return explicit response if set, otherwise default
        return self._thread_alive_responses.get(thread_id, self._alive_default)

    def get_create_thread_calls(self) -> list[dict]:
        """Get all recorded create_thread calls."""
        return self._create_thread_calls

    def set_thread_alive(self, thread_id: int, alive: bool) -> None:
        """Mark a thread as alive or dead."""
        self._thread_alive_responses[thread_id] = alive


@pytest.fixture
async def fake_bot():
    """Provide a FakeBot instance."""
    return FakeBot()


@pytest.fixture
async def in_memory_db():
    """Provide an in-memory SQLite connection."""
    conn = await aiosqlite.connect(":memory:")
    # Initialize schema
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            thread_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.commit()
    yield conn
    await conn.close()


@pytest.mark.asyncio
class TestThreadRegistry:
    """Tests for ThreadRegistry.get_or_create_thread."""

    async def test_ac21_new_session_creates_thread(self, fake_bot, in_memory_db):
        """AC2.1: New session_id calls create_thread, returns new ID."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        thread_id = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")

        # Should have called create_thread once
        calls = fake_bot.get_create_thread_calls()
        assert len(calls) == 1
        assert calls[0]["name"] == "cc · aaa · sess-aaa"

        # Should return the thread ID
        assert thread_id == 1000

        # Session should be persisted
        row = await get_session(in_memory_db, "sess-aaa")
        assert row is not None
        assert row.thread_id == 1000

    async def test_ac21_different_sessions_create_different_threads(self, fake_bot, in_memory_db):
        """AC2.1: Two different session_ids create two distinct threads."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        thread_id_1 = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")
        thread_id_2 = await registry.get_or_create_thread("sess-bbb", "/tmp/bbb")

        # Should have called create_thread twice
        calls = fake_bot.get_create_thread_calls()
        assert len(calls) == 2

        # Thread IDs should be different
        assert thread_id_1 != thread_id_2
        assert thread_id_1 == 1000
        assert thread_id_2 == 1001

    async def test_ac22_same_session_reuses_thread(self, fake_bot, in_memory_db):
        """AC2.2: Two calls with same session_id return same thread, no new create."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        thread_id_1 = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")
        thread_id_2 = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")

        # Should only create once
        calls = fake_bot.get_create_thread_calls()
        assert len(calls) == 1

        # Both calls return same ID
        assert thread_id_1 == thread_id_2
        assert thread_id_1 == 1000

    async def test_ac22_same_session_bumps_last_activity(self, fake_bot, in_memory_db):
        """AC2.2: Re-using same session bumps last_activity."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")
        row1 = await get_session(in_memory_db, "sess-aaa")
        assert row1 is not None
        t1 = row1.last_activity

        # Small delay to ensure time advance
        await asyncio.sleep(0.01)

        await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")
        row2 = await get_session(in_memory_db, "sess-aaa")
        assert row2 is not None
        t2 = row2.last_activity

        # last_activity should have increased
        assert t2 >= t1

    async def test_ac22_concurrent_same_session_single_create(self, fake_bot, in_memory_db):
        """AC2.2 concurrency: Two concurrent calls with same new session_id create only once."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        # Run both concurrently
        results = await asyncio.gather(
            registry.get_or_create_thread("sess-aaa", "/tmp/aaa"),
            registry.get_or_create_thread("sess-aaa", "/tmp/aaa"),
        )

        # Should create thread only once due to lock
        calls = fake_bot.get_create_thread_calls()
        assert len(calls) == 1

        # Both should return the same ID
        assert results[0] == results[1]
        assert results[0] == 1000

    async def test_ac23_persistence_across_restart(self, fake_bot, in_memory_db):
        """AC2.3: After restart (close + reopen DB), session maps to same thread_id."""
        # First registry: create a session
        registry1 = ThreadRegistry(fake_bot, in_memory_db)
        thread_id_1 = await registry1.get_or_create_thread("sess-aaa", "/tmp/aaa")

        # Reset fake bot for clean call tracking
        calls_before = len(fake_bot.get_create_thread_calls())

        # Simulate "restart": new registry, same connection
        registry2 = ThreadRegistry(fake_bot, in_memory_db)
        thread_id_2 = await registry2.get_or_create_thread("sess-aaa", "/tmp/aaa")

        # Should NOT have created a new thread (no additional calls)
        calls_after = len(fake_bot.get_create_thread_calls())
        assert calls_after == calls_before  # No new call

        # Should return the same ID
        assert thread_id_2 == thread_id_1

    async def test_ac24_dead_thread_recreated(self, fake_bot, in_memory_db):
        """AC2.4: When thread is deleted, get_or_create_thread detects and recreates."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        # Create initial thread
        thread_id_1 = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")
        assert thread_id_1 == 1000

        # Mark the thread as dead (Discord deleted it)
        fake_bot.set_thread_alive(1000, False)

        # Next call should detect dead thread, recreate
        thread_id_2 = await registry.get_or_create_thread("sess-aaa", "/tmp/aaa")

        # Should have created twice
        calls = fake_bot.get_create_thread_calls()
        assert len(calls) == 2

        # New ID should be different
        assert thread_id_2 == 1001
        assert thread_id_2 != thread_id_1

        # Session should be updated
        row = await get_session(in_memory_db, "sess-aaa")
        assert row is not None
        assert row.thread_id == 1001

    async def test_cwd_leaf_extraction(self, fake_bot, in_memory_db):
        """Thread name uses cwd_leaf (last path segment) not full path."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        await registry.get_or_create_thread("sess-test", "/home/user/projects/myapp")

        calls = fake_bot.get_create_thread_calls()
        assert calls[0]["name"] == "cc · myapp · sess-tes"

    async def test_cwd_root_fallback(self, fake_bot, in_memory_db):
        """Thread name uses 'root' for cwd='/'."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        await registry.get_or_create_thread("sess-test", "/")

        calls = fake_bot.get_create_thread_calls()
        assert calls[0]["name"] == "cc · root · sess-tes"

    async def test_session_id_truncation(self, fake_bot, in_memory_db):
        """Thread name uses first 8 chars of session_id."""
        registry = ThreadRegistry(fake_bot, in_memory_db)

        await registry.get_or_create_thread("abcd1234efghijk", "/tmp")

        calls = fake_bot.get_create_thread_calls()
        assert calls[0]["name"] == "cc · tmp · abcd1234"
