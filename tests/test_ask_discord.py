"""Tests for the ask_discord.py skill script."""

import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from aiohttp import test_utils, web


@pytest.fixture
async def fake_bridge():
    """Create a fake bridge server for testing."""
    seen = []

    async def handle_ask(req):
        seen.append(await req.json())
        return web.json_response({"reply": "yes", "replied_at": "2026-05-07T12:00:00Z"})

    app = web.Application()
    app.router.add_post("/v1/ask", handle_ask)

    server = test_utils.TestServer(app)
    await server.start_server()

    base_url = f"http://{server.host}:{server.port}"

    try:
        yield {"url": base_url, "seen": seen}
    finally:
        await server.close()


async def _run_ask_discord(
    question: str,
    session_id: str | None = None,
    cwd: str | None = None,
    timeout_secs: int | None = None,
    bridge_url: str | None = None,
    env_vars: dict | None = None,
) -> subprocess.CompletedProcess:
    """
    Run the ask_discord.py script with the given arguments.
    Returns the completed process.
    """
    script_path = Path(__file__).parent.parent / "skills" / "ask_discord.py"

    # Build command
    cmd = ["python3", str(script_path), question]
    if session_id is not None:
        cmd.extend(["--session-id", session_id])
    if cwd is not None:
        cmd.extend(["--cwd", cwd])
    if timeout_secs is not None:
        cmd.extend(["--timeout-secs", str(timeout_secs)])

    # Build environment
    env = dict(os.environ)
    if env_vars:
        env.update(env_vars)
    if bridge_url:
        env["BRIDGE_URL"] = bridge_url

    def _run():
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, _run)
    return result


@pytest.mark.asyncio
async def test_ac51_happy_path(fake_bridge):
    """
    AC5.1 happy path: Fake bridge returns 200 with reply.
    Run ask_discord.py "yes or no?" --session-id sess --cwd /tmp.
    Assert exit 0 and stdout is exactly "yes".
    Verify request body has all four expected fields.
    """
    result = await _run_ask_discord(
        "yes or no?",
        session_id="sess",
        cwd="/tmp",
        bridge_url=fake_bridge["url"],
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify stdout is exactly "yes"
    assert result.stdout.strip() == "yes"

    # Verify the request body sent
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]
    assert body["session_id"] == "sess"
    assert body["cwd"] == "/tmp"
    assert body["question"] == "yes or no?"
    assert body["timeout_secs"] == 900  # default 15m


@pytest.mark.asyncio
async def test_ac52_timeout(fake_bridge):
    """
    AC5.2 timeout: Fake bridge returns 408.
    Stdout starts with "no reply within" and ends with "; proceeding with best-guess".
    Exit 0.
    """

    async def handle_timeout(req):
        return web.json_response({"error": "timeout"}, status=408)

    app = web.Application()
    app.router.add_post("/v1/ask", handle_timeout)

    server = test_utils.TestServer(app)
    await server.start_server()

    try:
        bridge_url = f"http://{server.host}:{server.port}"

        result = await _run_ask_discord(
            "test question",
            session_id="sess",
            cwd="/tmp",
            timeout_secs=600,
            bridge_url=bridge_url,
        )

        # Verify exit code
        assert result.returncode == 0

        # Verify stdout format
        stdout = result.stdout.strip()
        assert stdout.startswith("no reply within")
        assert stdout.endswith("; proceeding with best-guess")

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_ac53_bridge_down_connection_refused(fake_bridge):
    """
    AC5.3 bridge down (connection refused): Pass BRIDGE_URL to unused port.
    Stdout matches "bridge daemon is not reachable".
    Exit 0.
    """
    # Use a port that's unlikely to be listening
    bridge_url = "http://127.0.0.1:54321"

    result = await _run_ask_discord(
        "test question",
        session_id="sess",
        cwd="/tmp",
        bridge_url=bridge_url,
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify stdout contains expected error message
    assert "bridge daemon is not reachable" in result.stdout
    assert "54321" in result.stdout or "127.0.0.1" in result.stdout


@pytest.mark.asyncio
async def test_ac53_bot_disconnected_503(fake_bridge):
    """
    AC5.3 bot disconnected: Fake bridge returns 503.
    Stdout matches "bot is not connected".
    Exit 0.
    """

    async def handle_unavailable(req):
        return web.json_response({"error": "bot_not_connected"}, status=503)

    app = web.Application()
    app.router.add_post("/v1/ask", handle_unavailable)

    server = test_utils.TestServer(app)
    await server.start_server()

    try:
        bridge_url = f"http://{server.host}:{server.port}"

        result = await _run_ask_discord(
            "test question",
            session_id="sess",
            cwd="/tmp",
            bridge_url=bridge_url,
        )

        # Verify exit code
        assert result.returncode == 0

        # Verify stdout contains expected error message
        assert "bot is not connected" in result.stdout

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_http_500_error(fake_bridge):
    """
    Bridge returns 500: Stdout starts with "ask-discord error: HTTP 500".
    Exit 0.
    """

    async def handle_error(req):
        return web.json_response({"error": "internal error"}, status=500)

    app = web.Application()
    app.router.add_post("/v1/ask", handle_error)

    server = test_utils.TestServer(app)
    await server.start_server()

    try:
        bridge_url = f"http://{server.host}:{server.port}"

        result = await _run_ask_discord(
            "test question",
            session_id="sess",
            cwd="/tmp",
            bridge_url=bridge_url,
        )

        # Verify exit code
        assert result.returncode == 0

        # Verify stdout
        assert result.stdout.strip().startswith("ask-discord error: HTTP 500")

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_malformed_json_response(fake_bridge):
    """
    Bridge returns 200 but malformed JSON: Stdout starts with "ask-discord error:".
    Exit 0.
    """

    async def handle_malformed(req):
        return web.Response(text="not json", status=200)

    app = web.Application()
    app.router.add_post("/v1/ask", handle_malformed)

    server = test_utils.TestServer(app)
    await server.start_server()

    try:
        bridge_url = f"http://{server.host}:{server.port}"

        result = await _run_ask_discord(
            "test question",
            session_id="sess",
            cwd="/tmp",
            bridge_url=bridge_url,
        )

        # Verify exit code
        assert result.returncode == 0

        # Verify stdout starts with error prefix
        assert result.stdout.strip().startswith("ask-discord error:")

    finally:
        await server.close()


@pytest.mark.asyncio
async def test_session_id_resolution_from_env(fake_bridge):
    """
    With no --session-id and CLAUDE_SESSION_ID=abc in env,
    the request body contains "session_id":"abc".
    """
    result = await _run_ask_discord(
        "test question",
        cwd="/tmp",
        bridge_url=fake_bridge["url"],
        env_vars={"CLAUDE_SESSION_ID": "abc"},
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify request body
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]
    assert body["session_id"] == "abc"


@pytest.mark.asyncio
async def test_session_id_absent_everywhere(fake_bridge):
    """
    With no --session-id and no CLAUDE_SESSION_ID env var,
    stderr contains a warning; body has session_id="unknown-session".
    Exit 0.
    """
    result = await _run_ask_discord(
        "test question",
        cwd="/tmp",
        bridge_url=fake_bridge["url"],
        env_vars={"CLAUDE_SESSION_ID": ""},  # Unset it
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify stderr warning
    assert "unknown-session" in result.stderr or "session" in result.stderr.lower()

    # Verify request body
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]
    assert body["session_id"] == "unknown-session"


@pytest.mark.asyncio
async def test_timeout_secs_clamping_low(fake_bridge):
    """
    --timeout-secs 1 → body has timeout_secs: 5 (clamped to minimum).
    """
    result = await _run_ask_discord(
        "test question",
        session_id="sess",
        cwd="/tmp",
        timeout_secs=1,
        bridge_url=fake_bridge["url"],
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify request body has clamped timeout
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]
    assert body["timeout_secs"] == 5


@pytest.mark.asyncio
async def test_timeout_secs_clamping_high(fake_bridge):
    """
    --timeout-secs 99999 → body has timeout_secs: 3600 (clamped to maximum).
    """
    result = await _run_ask_discord(
        "test question",
        session_id="sess",
        cwd="/tmp",
        timeout_secs=99999,
        bridge_url=fake_bridge["url"],
    )

    # Verify exit code
    assert result.returncode == 0

    # Verify request body has clamped timeout
    assert len(fake_bridge["seen"]) == 1
    body = fake_bridge["seen"][0]
    assert body["timeout_secs"] == 3600
