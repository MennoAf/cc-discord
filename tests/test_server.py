"""Tests for the aiohttp server (POST /v1/notify, GET /v1/health)."""

import asyncio
import json
import time

import pytest
from aiohttp import test_utils, web

from bridge.bot import BotNotReady
from bridge.server import build_app


class FakeBot:
    """Minimal fake Bot for testing the server without real Discord."""

    def __init__(self, channel_id: int = 12345, is_ready: bool = False) -> None:
        self._channel_id = channel_id
        self._is_ready = is_ready
        self._post_calls: list[dict] = []

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

    def get_post_calls(self) -> list[dict]:
        return self._post_calls


@pytest.fixture
async def fake_bot():
    return FakeBot()


@pytest.fixture
async def client(fake_bot):
    """Create a test client for the aiohttp app."""
    started_at = time.monotonic()
    app = await build_app(fake_bot, started_at=started_at)
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
    assert body["thread_id"] is None
    assert body["message_id"] == 1001

    # Verify bot.post was called with the right args
    calls = fake_bot.get_post_calls()
    assert len(calls) == 1
    assert calls[0]["message"] == "hello world"
    assert calls[0]["thread_id"] is None


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
    assert body["thread_id"] is None
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
    assert body["thread_id"] is None
    assert body["message_id"] == 1001
