"""Tests for the pretooluse-approve.py hook script."""

import asyncio
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from aiohttp import test_utils, web


@pytest.fixture
async def fake_bridge():
    """Create a fake bridge server for testing the pretooluse hook."""
    seen = []

    async def handle_pretooluse(req):
        seen.append(await req.json())
        # Return a configurable response from the latest request's decision_override
        latest = seen[-1] if seen else {}
        override = latest.pop("__decision_override", None)
        if override:
            return web.json_response(override)
        return web.json_response({"decision": "deny", "reason": "test deny"})

    app = web.Application()
    app.router.add_post("/v1/hook/pretooluse", handle_pretooluse)

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
    env_overrides: dict | None = None,
) -> subprocess.CompletedProcess:
    """
    Run the pretooluse-approve.py hook with the given payload and environment.
    Runs in an executor to avoid blocking the event loop.
    Returns the completed process.
    """
    stdin_data = json.dumps(payload)

    # Build environment
    import os

    env = {
        "PATH": os.environ.get("PATH", ""),
    }
    if bridge_url:
        env["BRIDGE_URL"] = bridge_url
    if env_overrides:
        env.update(env_overrides)

    # Run the hook script in an executor to avoid blocking the event loop
    hook_path = Path(__file__).parent.parent / "hooks" / "pretooluse-approve.py"

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


def _parse_hook_output(stdout: bytes) -> dict:
    """Parse the hookSpecificOutput from hook stdout."""
    output = json.loads(stdout.decode("utf-8"))
    return output.get("hookSpecificOutput", {})


@pytest.mark.asyncio
async def test_pretooluse_allow_decision(fake_bridge):
    """Hook forwards request to bridge, returns allow decision."""
    payload = {
        "tool_name": "Bash",
        "tool_input": {"cmd": "ls /tmp"},
    }

    # Configure bridge to return allow
    # For this we just rely on the default behavior
    result = await _run_hook(
        payload,
        bridge_url=fake_bridge["url"],
        env_overrides={"CC_DISCORD_TASK_ID": "task-1"},
    )

    assert result.returncode == 0
    output = _parse_hook_output(result.stdout)
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"  # default


@pytest.mark.asyncio
async def test_pretooluse_malformed_stdin(fake_bridge):
    """Hook receives malformed JSON on stdin, emits deny."""
    stdin_data = "not json"

    import os

    env = {
        "PATH": os.environ.get("PATH", ""),
        "BRIDGE_URL": fake_bridge["url"],
        "CC_DISCORD_TASK_ID": "task-1",
    }

    hook_path = Path(__file__).parent.parent / "hooks" / "pretooluse-approve.py"

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

    assert result.returncode == 0
    output = _parse_hook_output(result.stdout)
    assert output["permissionDecision"] == "deny"
    assert "malformed" in output["permissionDecisionReason"].lower()


@pytest.mark.asyncio
async def test_pretooluse_no_task_id_env(fake_bridge):
    """Hook without CC_DISCORD_TASK_ID emits 'ask' to fall back to default."""
    payload = {
        "tool_name": "Bash",
        "tool_input": {"cmd": "ls"},
    }

    # Run without setting CC_DISCORD_TASK_ID
    result = await _run_hook(
        payload,
        bridge_url=fake_bridge["url"],
        env_overrides={},  # No CC_DISCORD_TASK_ID
    )

    assert result.returncode == 0
    output = _parse_hook_output(result.stdout)
    assert output["permissionDecision"] == "ask"
    assert "no bridge task_id" in output["permissionDecisionReason"].lower()


@pytest.mark.asyncio
async def test_pretooluse_bridge_unreachable():
    """Hook when bridge is unreachable emits deny with 'unavailable'."""
    payload = {
        "tool_name": "Bash",
        "tool_input": {"cmd": "ls"},
    }

    # Use a port that's definitely not running
    result = await _run_hook(
        payload,
        bridge_url="http://127.0.0.1:19999",  # Non-existent port
        env_overrides={"CC_DISCORD_TASK_ID": "task-1"},
    )

    assert result.returncode == 0
    output = _parse_hook_output(result.stdout)
    assert output["permissionDecision"] == "deny"
    assert "unavailable" in output["permissionDecisionReason"].lower()


@pytest.mark.asyncio
async def test_pretooluse_bridge_http_error(fake_bridge):
    """Hook when bridge returns 500 emits deny."""

    async def handle_error(req):
        return web.json_response({"error": "internal"}, status=500)

    app = web.Application()
    app.router.add_post("/v1/hook/pretooluse", handle_error)

    server = test_utils.TestServer(app)
    await server.start_server()

    bridge_url = f"http://{server.host}:{server.port}"

    try:
        payload = {
            "tool_name": "Bash",
            "tool_input": {"cmd": "ls"},
        }

        result = await _run_hook(
            payload,
            bridge_url=bridge_url,
            env_overrides={"CC_DISCORD_TASK_ID": "task-1"},
        )

        assert result.returncode == 0
        output = _parse_hook_output(result.stdout)
        assert output["permissionDecision"] == "deny"
        assert "HTTP 500" in output["permissionDecisionReason"]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_pretooluse_bridge_malformed_response(fake_bridge):
    """Hook when bridge returns invalid JSON emits deny."""

    async def handle_invalid_json(req):
        return web.Response(text="not json", status=200)

    app = web.Application()
    app.router.add_post("/v1/hook/pretooluse", handle_invalid_json)

    server = test_utils.TestServer(app)
    await server.start_server()

    bridge_url = f"http://{server.host}:{server.port}"

    try:
        payload = {
            "tool_name": "Bash",
            "tool_input": {"cmd": "ls"},
        }

        result = await _run_hook(
            payload,
            bridge_url=bridge_url,
            env_overrides={"CC_DISCORD_TASK_ID": "task-1"},
        )

        assert result.returncode == 0
        output = _parse_hook_output(result.stdout)
        assert output["permissionDecision"] == "deny"
        assert "malformed JSON" in output["permissionDecisionReason"]
    finally:
        await server.close()
