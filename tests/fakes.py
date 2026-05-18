"""Shared test fixtures for FakeBot and FakeZellij."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTypingContext:
    """Fake typing context manager."""

    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> "FakeTypingContext":
        self.entered = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited = True


@dataclass
class FakeBotChannel:
    """Fake channel object from bot."""

    id: int = 1000
    typing_context: FakeTypingContext = field(default_factory=FakeTypingContext)

    def typing(self) -> FakeTypingContext:
        """Return a fake typing context manager."""
        return self.typing_context


@dataclass
class FakeHTTP:
    """Fake discord.http.HTTPClient."""

    pass


@dataclass
class FakeConnection:
    """Fake discord.gateway.DiscordWebSocket state."""

    _command_tree: Any = None


@dataclass
class FakeClient:
    """Fake discord.Client with minimal attributes needed for CommandTree."""

    http: FakeHTTP = field(default_factory=FakeHTTP)
    _connection: FakeConnection = field(default_factory=FakeConnection)


@dataclass
class FakeBot:
    """Minimal fake Bot for testing commands and tasks."""

    _client: Any = field(default_factory=lambda: FakeClient())
    _post_calls: list[dict] = field(default_factory=list)
    _thread_calls: list[dict] = field(default_factory=list)
    _channel_calls: list[dict] = field(default_factory=list)
    _archive_calls: list[dict] = field(default_factory=list)
    _reaction_calls: list[dict] = field(default_factory=list)
    _fake_channels: dict[int, FakeBotChannel] = field(default_factory=dict)
    is_ready: bool = True

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

    async def create_channel(self, name: str) -> int:
        """Fake create_channel: record the call, return a fake channel ID."""
        channel_id = 3000 + len(self._channel_calls)
        self._channel_calls.append({"name": name})
        return channel_id

    async def archive_thread(self, thread_id: int) -> None:
        """Fake archive_thread: record the call."""
        self._archive_calls.append({"thread_id": thread_id})

    async def add_reactions(self, message_id: int, thread_id: int, emoji: list[str]) -> None:
        """Fake add_reactions: record the call."""
        self._reaction_calls.append({"message_id": message_id, "thread_id": thread_id, "emoji": emoji})

    async def fetch_messageable(self, thread_id: int) -> FakeBotChannel:
        """Fake fetch_messageable: return a FakeBotChannel."""
        if thread_id not in self._fake_channels:
            self._fake_channels[thread_id] = FakeBotChannel(id=thread_id)
        return self._fake_channels[thread_id]

    def get_post_calls(self) -> list[dict]:
        return self._post_calls

    def get_thread_calls(self) -> list[dict]:
        return self._thread_calls

    def get_archive_calls(self) -> list[dict]:
        return self._archive_calls

    def get_reaction_calls(self) -> list[dict]:
        return self._reaction_calls


@dataclass
class FakeZellij:
    """Minimal fake ZellijManager for testing."""

    _spawn_calls: list[dict] = field(default_factory=list)
    _write_calls: list[dict] = field(default_factory=list)
    _close_calls: list[dict] = field(default_factory=list)
    _send_keys_calls: list[dict] = field(default_factory=list)

    async def spawn_task(
        self, cwd: str, pane_name: str, layout_path: str
    ) -> str:
        """Fake spawn_task. The new contract takes a layout file path
        instead of env+extra_argv (env vars and claude argv now live in
        the layout)."""
        self._spawn_calls.append(
            {"cwd": cwd, "pane_name": pane_name, "layout_path": layout_path}
        )
        return "terminal_1"

    async def write_to_pane(self, pane_id: str, text: str) -> None:
        """Fake write_to_pane."""
        self._write_calls.append({"pane_id": pane_id, "text": text})

    async def send_keys(self, pane_id: str, *byte_vals: int) -> None:
        """Fake send_keys."""
        self._send_keys_calls.append({"pane_id": pane_id, "bytes": list(byte_vals)})

    async def close_pane(self, pane_id: str) -> None:
        """Fake close_pane."""
        self._close_calls.append({"pane_id": pane_id})

    async def list_panes(self) -> list[dict]:
        """Fake list_panes."""
        return []
