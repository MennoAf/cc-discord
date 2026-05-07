"""Tests for the aiohttp server (POST /v1/notify, GET /v1/health)."""

import asyncio
import json
import time
from pathlib import Path

import aiosqlite
import pytest
from aiohttp import test_utils, web

from bridge.bot import BotNotReady
from bridge.server import build_app
from bridge.threads import ThreadRegistry
from bridge import state


class FakeBot:
    """Minimal fake Bot for testing the server without real Discord."""

    def __init__(self, channel_id: int = 12345, is_ready: bool = False) -> None:
        self._channel_id = channel_id
        self._is_ready = is_ready
        self._post_calls: list[dict] = []
        self._create_thread_calls: list[dict] = []
        self._next_thread_id = 2000
        self._thread_alive_map: dict[int, bool] = {}

    @property
    def channel_id(self) -> int:
        return self._channel_id

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def set_ready(self, ready: bool) -> None:
        self._is_ready = ready

    async def post(self, message: str, *, thread_id: int | None = None) -> list[int]:
        """Fake post: record the call, return a fake message ID."""
        self._post_calls.append(
            {"message": message, "thread_id": thread_id}
        )
        if not self.is_ready:
            raise BotNotReady("bot not connected to Discord")
        # Return a fake ID (first chunk always gets ID 1001)
        return [1001]

    async def create_thread(self, name: str) -> int:
        """Fake create_thread: record the call and return a fake thread ID."""
        thread_id = self._next_thread_id
        self._next_thread_id += 1
        self._create_thread_calls.append({"name": name, "id": thread_id})
        self._thread_alive_map[thread_id] = True
        return thread_id

    async def thread_alive(self, thread_id: int) -> bool:
        """Fake thread_alive: check the map."""
        return self._thread_alive_map.get(thread_id, True)

    def set_thread_alive(self, thread_id: int, alive: bool) -> None:
        """Set whether a thread is considered alive."""
        self._thread_alive_map[thread_id] = alive

    def get_post_calls(self) -> list[dict]:
        return self._post_calls

    def get_create_thread_calls(self) -> list[dict]:
        return self._create_thread_calls


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
    await conn.commit()
    yield conn
    await conn.close()


@pytest.fixture
async def client(fake_bot, in_memory_db):
    """Create a test client for the aiohttp app with ThreadRegistry wired in."""
    started_at = time.monotonic()
    app = await build_app(fake_bot, started_at=started_at)
    registry = ThreadRegistry(fake_bot, in_memory_db)
    app["threads"] = registry
    async with test_utils.TestClient(test_utils.TestServer(app)) as client:
        yield client


@pytest.mark.asyncio
async def test_health_bot_not_ready(client, fake_bot):
    """GET /v1/health with bot not ready returns bot_connected: false."""
    fake_bot.set_ready(False)
    resp = await client.get("/v1/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["bot_connected"] is False
    assert body["channel_id"] == 12345
    assert body["uptime_secs"] >= 0


@pytest.mark.asyncio
async def test_health_bot_ready(client, fake_bot):
    """GET /v1/health with bot ready returns bot_connected: true."""
    fake_bot.set_ready(True)
    resp = await client.get("/v1/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["bot_connected"] is True
    assert body["channel_id"] == 12345
    assert body["uptime_secs"] >= 0


@pytest.mark.asyncio
async def test_notify_success(client, fake_bot):
    """POST /v1/notify with bot ready and valid body returns 200."""
    fake_bot.set_ready(True)
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world",
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body["thread_id"], int)
    assert body["message_id"] == 1001

    # Verify bot.post was called with the right thread_id
    calls = fake_bot.get_post_calls()
    assert len(calls) == 1
    assert calls[0]["message"] == "hello world"
    assert calls[0]["thread_id"] == body["thread_id"]

    # Verify create_thread was called once
    create_calls = fake_bot.get_create_thread_calls()
    assert len(create_calls) == 1
    assert "cc · tmp ·" in create_calls[0]["name"]


@pytest.mark.asyncio
async def test_notify_bot_not_ready(client, fake_bot):
    """POST /v1/notify with bot not ready returns 503."""
    fake_bot.set_ready(False)
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world",
        },
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "bot_not_connected"

    # Verify bot.post was NOT called
    calls = fake_bot.get_post_calls()
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_notify_empty_body(client, fake_bot):
    """POST /v1/notify with empty body returns 400."""
    fake_bot.set_ready(True)
    resp = await client.post("/v1/notify", json={})
    assert resp.status == 400
    body = await resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_notify_missing_session_id(client, fake_bot):
    """POST /v1/notify missing session_id returns 400."""
    fake_bot.set_ready(True)
    resp = await client.post(
        "/v1/notify",
        json={
            "cwd": "/tmp",
            "message": "hello world",
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_notify_missing_cwd(client, fake_bot):
    """POST /v1/notify missing cwd returns 400."""
    fake_bot.set_ready(True)
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "message": "hello world",
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_notify_missing_message(client, fake_bot):
    """POST /v1/notify missing message returns 400."""
    fake_bot.set_ready(True)
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_notify_bot_error(client, fake_bot):
    """POST /v1/notify where bot.post raises exception returns 500."""
    fake_bot.set_ready(True)
    # Patch the bot to raise a generic exception
    original_post = fake_bot.post

    async def failing_post(*args, **kwargs):
        raise RuntimeError("Something went wrong")

    fake_bot.post = failing_post
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world",
        },
    )
    assert resp.status == 500
    body = await resp.json()
    assert body["error"] == "internal"
    # Should NOT contain stack trace
    assert "Traceback" not in body.get("error", "")


@pytest.mark.asyncio
async def test_notify_recovers_after_error(client, fake_bot):
    """Server recovers cleanly after a 503 error."""
    fake_bot.set_ready(False)
    resp1 = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world",
        },
    )
    assert resp1.status == 503

    # Now make the bot ready and try again
    fake_bot.set_ready(True)
    resp2 = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world 2",
        },
    )
    assert resp2.status == 200
    body = await resp2.json()
    assert isinstance(body["thread_id"], int)
    assert body["message_id"] == 1001


@pytest.mark.asyncio
async def test_notify_with_optional_fields(client, fake_bot):
    """POST /v1/notify accepts optional title and level fields."""
    fake_bot.set_ready(True)
    resp = await client.post(
        "/v1/notify",
        json={
            "session_id": "test-session",
            "cwd": "/tmp",
            "message": "hello world",
            "title": "Test Title",
            "level": "info",
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body["thread_id"], int)
    assert body["message_id"] == 1001


@pytest.mark.asyncio
async def test_notify_distinct_sessions_distinct_threads(client, fake_bot):
    """AC2.1: Two POST /v1/notify calls with different session_ids create distinct threads."""
    fake_bot.set_ready(True)
    resp1 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-aaa",
            "cwd": "/tmp/aaa",
            "message": "alpha",
        },
    )
    assert resp1.status == 200
    body1 = await resp1.json()
    thread_id_1 = body1["thread_id"]

    resp2 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-bbb",
            "cwd": "/tmp/bbb",
            "message": "beta",
        },
    )
    assert resp2.status == 200
    body2 = await resp2.json()
    thread_id_2 = body2["thread_id"]

    # Different sessions should create different threads
    assert thread_id_1 != thread_id_2
    # Each should have called create_thread once
    create_calls = fake_bot.get_create_thread_calls()
    assert len(create_calls) == 2


@pytest.mark.asyncio
async def test_notify_same_session_reuses_thread(client, fake_bot):
    """AC2.2: Two POST /v1/notify calls with same session_id route to same thread."""
    fake_bot.set_ready(True)
    resp1 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-aaa",
            "cwd": "/tmp/aaa",
            "message": "first",
        },
    )
    assert resp1.status == 200
    body1 = await resp1.json()
    thread_id_1 = body1["thread_id"]

    resp2 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-aaa",
            "cwd": "/tmp/aaa",
            "message": "second",
        },
    )
    assert resp2.status == 200
    body2 = await resp2.json()
    thread_id_2 = body2["thread_id"]

    # Same session should reuse the thread
    assert thread_id_1 == thread_id_2
    # Should have only called create_thread once
    create_calls = fake_bot.get_create_thread_calls()
    assert len(create_calls) == 1


@pytest.mark.asyncio
async def test_notify_404_recovery_creates_new_thread(client, fake_bot):
    """AC2.4: When a thread is deleted, next call recreates it and updates mapping."""
    fake_bot.set_ready(True)
    resp1 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-aaa",
            "cwd": "/tmp/aaa",
            "message": "first",
        },
    )
    assert resp1.status == 200
    body1 = await resp1.json()
    thread_id_1 = body1["thread_id"]

    # Simulate thread deletion
    fake_bot.set_thread_alive(thread_id_1, False)

    # Next call should detect dead thread and create a new one
    resp2 = await client.post(
        "/v1/notify",
        json={
            "session_id": "sess-aaa",
            "cwd": "/tmp/aaa",
            "message": "second",
        },
    )
    assert resp2.status == 200
    body2 = await resp2.json()
    thread_id_2 = body2["thread_id"]

    # Should have created a new thread (different ID)
    assert thread_id_1 != thread_id_2
    # Should have called create_thread twice total
    create_calls = fake_bot.get_create_thread_calls()
    assert len(create_calls) == 2
