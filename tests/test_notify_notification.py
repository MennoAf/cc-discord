"""Tests for the notify-notification.py hook script."""

import asyncio
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from aiohttp import test_utils, web


@pytest.fixture
async def fake_bridge():
    """Create a fake bridge server for testing (using test_utils.TestServer)."""
    seen = []

    async def handle_notify(req):
        seen.append(await req.json())
        return web.json_response({"thread_id": 999, "message_id": 123})

    app = web.Application()
    app.router.add_post("/v1/notify", handle_notify)

    server = test_utils.TestServer(app)
    await server.start_server()

    base_url = f"http://{server.host}:{server.port}"

    try:
        yield {"url": base_url, "seen": seen}
    finally:
        await server.close()


@pytest.fixture
async def fake_webhook():
    """Create a fake webhook server for testing (using test_utils.TestServer)."""
    seen = []

    async def handle_webhook(req):
        seen.append(await req.json())
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/", handle_webhook)

    server = test_utils.TestServer(app)
    await server.start_server()

    base_url = f"http://{server.host}:{server.port}"

    try:
        yield {"url": base_url, "seen": seen}
    finally:
        await server.close()


async def _run_hook(
    payload: dict,
    bridge_url: str | None = None,
    home_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    """
    Run the notify-notification.py hook with the given payload and environment.
    Runs in an executor to avoid blocking the event loop.
    Returns the completed process.
    """
    stdin_data = json.dumps(payload)

    # Build environment - start with minimal env to avoid leaking real home
    import os

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home_dir) if home_dir else "/tmp",
    }
    if bridge_url:
        env["BRIDGE_URL"] = bridge_url

    # Run the hook script in an executor to avoid blocking the event loop
    hook_path = Path(__file__).parent.parent / "hooks" / "notify-notification.py"

    def _run():
        return subprocess.run(
            ["python3", str(hook_path)],
            input=stdin_data.encode("utf-8"),
            capture_output=True,
            env=env,
        )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, _run)
    return result


@pytest.mark.asyncio
async def test_bridge_happy_path_with_notification(fake_bridge):
    """
    Bridge happy path: stdin has notification field.
    Hook exits 0; bridge sees POST with title='⏸ awaiting input', level='warn', message=<notification text>.
    """
    payload = {
        "session_id": "smoke",
        "cwd": "/tmp/test",
        "notification": "permission required for tool",
    }

    result = await _run_hook(payload, bridge_url=fake_bridge["url"])

    assert result.returncode == 0
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]

    assert body["session_id"] == "smoke"
    assert body["cwd"] == "/tmp/test"
    assert body["title"] == "⏸ awaiting input"
    assert body["level"] == "warn"
    assert body["message"] == "permission required for tool"


@pytest.mark.asyncio
async def test_fallback_to_message_field(fake_bridge):
    """
    Stdin has message field but no notification field.
    Hook uses message as the body text.
    """
    payload = {
        "session_id": "smoke",
        "cwd": "/tmp/test",
        "message": "awaiting approval",
    }

    result = await _run_hook(payload, bridge_url=fake_bridge["url"])

    assert result.returncode == 0
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]

    assert body["message"] == "awaiting approval"
    assert body["title"] == "⏸ awaiting input"
    assert body["level"] == "warn"


@pytest.mark.asyncio
async def test_fallback_to_default_awaiting_input(fake_bridge):
    """
    Stdin has neither notification nor message field.
    Hook uses 'awaiting input' as the message body.
    """
    payload = {
        "session_id": "smoke",
        "cwd": "/tmp/test",
    }

    result = await _run_hook(payload, bridge_url=fake_bridge["url"])

    assert result.returncode == 0
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]

    assert body["message"] == "awaiting input"
    assert body["title"] == "⏸ awaiting input"
    assert body["level"] == "warn"


@pytest.mark.asyncio
async def test_bridge_down_webhook_fallback(fake_bridge, fake_webhook, tmp_path):
    """
    Bridge is unreachable → falls back to webhook.
    Webhook message format: '⏸ awaiting input — {notification_text} ({cwd}, session {short})'.
    Hook exits 0.
    """
    # Create fake home with webhook file
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    claude_dir = fake_home / ".claude"
    claude_dir.mkdir()
    webhook_file = claude_dir / "discord-notify-webhook"
    webhook_file.write_text(fake_webhook["url"])

    payload = {
        "session_id": "fallbacktest1234abcd",
        "cwd": "/tmp/work",
        "notification": "please approve",
    }

    result = await _run_hook(
        payload,
        bridge_url="http://127.0.0.1:54321",  # closed port
        home_dir=fake_home,
    )

    assert result.returncode == 0
    assert len(fake_webhook["seen"]) == 1
    body = fake_webhook["seen"][0]

    # Check webhook message format
    assert body["content"].startswith("⏸ awaiting input — ")
    assert "please approve" in body["content"]
    assert "/tmp/work" in body["content"]
    assert "fallback" in body["content"]  # first 8 chars of session_id


@pytest.mark.asyncio
async def test_malformed_stdin_json(fake_bridge):
    """
    Malformed JSON on stdin → hook exits 0, no POST.
    """
    hook_path = Path(__file__).parent.parent / "hooks" / "notify-notification.py"

    def _run():
        return subprocess.run(
            ["python3", str(hook_path)],
            input=b"this is not json",
            capture_output=True,
            env={"BRIDGE_URL": fake_bridge["url"], "HOME": "/tmp", "PATH": "/usr/bin:/bin"},
        )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, _run)

    assert result.returncode == 0
    assert len(fake_bridge["seen"]) == 0


@pytest.mark.asyncio
async def test_non_2xx_from_bridge_falls_through_to_webhook(
    fake_bridge, fake_webhook, tmp_path
):
    """
    Bridge returns non-2xx status → hook falls through to webhook fallback.
    Hook exits 0.
    """

    async def handle_notify_error(req):
        return web.json_response({"error": "server error"}, status=500)

    app = web.Application()
    app.router.add_post("/v1/notify", handle_notify_error)

    server = test_utils.TestServer(app)
    await server.start_server()

    try:
        bridge_url = f"http://{server.host}:{server.port}"

        # Create fake home with webhook file
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        claude_dir = fake_home / ".claude"
        claude_dir.mkdir()
        webhook_file = claude_dir / "discord-notify-webhook"
        webhook_file.write_text(fake_webhook["url"])

        payload = {
            "session_id": "error-test",
            "cwd": "/tmp/error",
            "notification": "test message",
        }

        result = await _run_hook(
            payload,
            bridge_url=bridge_url,
            home_dir=fake_home,
        )

        assert result.returncode == 0
        assert len(fake_webhook["seen"]) == 1
        body = fake_webhook["seen"][0]
        assert "⏸ awaiting input — " in body["content"]
    finally:
        await server.close()
