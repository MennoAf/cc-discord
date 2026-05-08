"""Tests for slash commands."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from bridge.tasks import TaskRegistry


@dataclass
class FakeResponse:
    """Fake discord.Interaction.response."""

    _deferred: bool = False

    async def defer(self, ephemeral: bool = False) -> None:
        """Record deferred call."""
        self._deferred = True


@dataclass
class FakeFollowup:
    """Fake discord.Interaction.followup."""

    _sends: list[dict] = field(default_factory=list)

    async def send(self, content: str, *, ephemeral: bool = False) -> Any:
        """Record send call, return a fake message."""
        self._sends.append({"content": content, "ephemeral": ephemeral})
        return None


@dataclass
class FakeFakeChannel:
    """Represents a Discord channel (for thread: parameter)."""

    id: int


@dataclass
class FakeInteraction:
    """Fake discord.Interaction for testing command handlers."""

    channel_id: int
    guild_id: int
    response: FakeResponse = field(default_factory=FakeResponse)
    followup: FakeFollowup = field(default_factory=FakeFollowup)

    def __post_init__(self) -> None:
        self.response = FakeResponse()
        self.followup = FakeFollowup()


@dataclass
class FakeBot:
    """Minimal fake Bot for testing commands."""

    _client: Any = field(default_factory=lambda: FakeClient())
    _post_calls: list[dict] = field(default_factory=list)
    _thread_calls: list[dict] = field(default_factory=list)
    _archive_calls: list[dict] = field(default_factory=list)

    @property
    def client(self) -> Any:
        return self._client

    @property
    def channel(self) -> Any:
        return FakeBotChannel()

    async def post(self, content: str, *, thread_id: int | None = None) -> list[int]:
        """Fake post: record the call, return a fake message ID."""
        self._post_calls.append({"content": content, "thread_id": thread_id})
        return [1001]

    async def create_thread(self, name: str) -> int:
        """Fake create_thread: record the call, return a fake thread ID."""
        thread_id = 2000 + len(self._thread_calls)
        self._thread_calls.append({"name": name})
        return thread_id

    async def archive_thread(self, thread_id: int) -> None:
        """Fake archive_thread: record the call."""
        self._archive_calls.append({"thread_id": thread_id})


@dataclass
class FakeBotChannel:
    """Fake channel object from bot."""

    id: int = 1000


@dataclass
class FakeHTTP:
    """Fake discord.http.HTTPClient."""

    pass


@dataclass
class FakeClient:
    """Fake discord.Client."""

    http: FakeHTTP = field(default_factory=FakeHTTP)


@dataclass
class FakeZellij:
    """Minimal fake ZellijManager for testing commands."""

    _spawn_calls: list[dict] = field(default_factory=list)
    _write_calls: list[dict] = field(default_factory=list)
    _close_calls: list[dict] = field(default_factory=list)

    async def spawn_task(
        self, cwd: str, env: dict, pane_name: str, *, extra_argv: list[str] | None = None
    ) -> str:
        """Fake spawn_task."""
        self._spawn_calls.append(
            {"cwd": cwd, "env": env, "pane_name": pane_name, "extra_argv": extra_argv}
        )
        return "terminal_1"

    async def write_to_pane(self, pane_id: str, text: str) -> None:
        """Fake write_to_pane."""
        self._write_calls.append({"pane_id": pane_id, "text": text})

    async def close_pane(self, pane_id: str) -> None:
        """Fake close_pane."""
        self._close_calls.append({"pane_id": pane_id})

    async def list_panes(self) -> list[dict]:
        """Fake list_panes."""
        return []


@pytest.fixture
def fake_bot() -> FakeBot:
    return FakeBot()


@pytest.fixture
def fake_zellij() -> FakeZellij:
    return FakeZellij()


@pytest.mark.asyncio
class TestCommands:
    """Tests for slash command handlers and utilities."""

    async def test_humanize_age_seconds(self) -> None:
        """_humanize_age formats seconds correctly."""
        from bridge.commands import _humanize_age
        import time

        now = int(time.time())

        # 30 seconds ago
        result = _humanize_age(now - 30)
        assert "s ago" in result

        # 5 minutes ago
        result = _humanize_age(now - 300)
        assert "m ago" in result

        # 2 hours ago
        result = _humanize_age(now - 7200)
        assert "h ago" in result

        # 3 days ago
        result = _humanize_age(now - 259200)
        assert "d ago" in result

    async def test_wait_for_session_bind_polls_until_ready(
        self, in_memory_db, fake_bot, fake_zellij
    ) -> None:
        """_wait_for_session_bind polls until session_id is set or timeout."""
        from bridge.commands import _wait_for_session_bind
        from bridge.state import upsert_task

        now = int(time.time())
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

        async def simulate_bind() -> None:
            """Simulate SessionStart binding after a short delay."""
            await asyncio.sleep(0.2)
            # Manually update the task's session_id
            task = registry.get_by_task_id("task-123")
            if task:
                task.current_claude_session_id = "sess-abc"
                await registry._index(task)

        # Start the bind simulation
        asyncio.create_task(simulate_bind())

        # Wait for session bind with 2s timeout
        await _wait_for_session_bind(registry, "task-123", timeout=2.0)

        # Task should now have session_id
        task = registry.get_by_task_id("task-123")
        assert task is not None
        assert task.current_claude_session_id == "sess-abc"

    async def test_wait_for_session_bind_timeout(
        self, in_memory_db, fake_bot, fake_zellij
    ) -> None:
        """_wait_for_session_bind raises asyncio.TimeoutError if session_id doesn't arrive."""
        from bridge.commands import _wait_for_session_bind
        from bridge.state import upsert_task

        now = int(time.time())
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

        # Wait with very short timeout (will timeout before bind)
        with pytest.raises(asyncio.TimeoutError):
            await _wait_for_session_bind(registry, "task-123", timeout=0.05)
