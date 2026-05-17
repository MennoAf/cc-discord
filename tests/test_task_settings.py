"""Tests for task settings file generation and cleanup."""

import json
from pathlib import Path

import pytest

from bridge.tasks import (
    TaskRegistry,
    _cleanup_task_settings,
    _read_user_mcp_servers,
    _write_task_settings,
)
from tests.fakes import FakeBot, FakeZellij


class TestWriteTaskSettings:
    """Tests for _write_task_settings helper."""

    def test_write_task_settings_creates_file(self, tmp_path: Path) -> None:
        """_write_task_settings creates a JSON file with the expected structure."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        assert out_path == settings_dir / "abc123.json"
        assert out_path.exists()

    def test_write_task_settings_json_structure(self, tmp_path: Path) -> None:
        """_write_task_settings registers the observability events plus
        PreToolUse scoped to AskUserQuestion / ExitPlanMode.

        Wider PreToolUse matchers are intentionally omitted so the user's
        auto-mode classifier drives approvals for everything else.
        """
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data.get("hooks", {})

        expected_events = [
            "PreToolUse",
            "SessionStart",
            "UserPromptSubmit",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "Notification",
            "SessionEnd",
        ]
        for event in expected_events:
            assert event in hooks, f"Missing event: {event}"
        # PreToolUse only matches AskUserQuestion + ExitPlanMode, NOT *.
        matchers = {m.get("matcher") for m in hooks["PreToolUse"]}
        assert matchers == {"AskUserQuestion", "ExitPlanMode"}, matchers

    def test_write_task_settings_hook_paths_absolute(self, tmp_path: Path) -> None:
        """_write_task_settings uses absolute paths for hook scripts."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data["hooks"]

        # All registered hooks point to absolute paths under hooks_dir.
        session_start_hooks = hooks["SessionStart"][0]["hooks"]
        assert len(session_start_hooks) == 1
        cmd = session_start_hooks[0]["command"]
        assert "event.py" in cmd
        assert str(hooks_dir) in cmd

    def test_write_task_settings_creates_parent_dirs(self, tmp_path: Path) -> None:
        """_write_task_settings creates parent directories if they don't exist."""
        settings_dir = tmp_path / "deep" / "nested" / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        assert out_path.parent.exists()
        assert out_path.exists()

    def test_write_task_settings_each_event_one_matcher(self, tmp_path: Path) -> None:
        """Observability events use a `*` matcher; PreToolUse has narrow
        matchers for the two interactive tools we intercept."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data["hooks"]

        for event, matchers in hooks.items():
            for entry in matchers:
                assert len(entry["hooks"]) == 1
            if event == "PreToolUse":
                assert {m["matcher"] for m in matchers} == {
                    "AskUserQuestion",
                    "ExitPlanMode",
                }
            else:
                assert len(matchers) == 1, f"Event {event} should have exactly 1 matcher"
                assert matchers[0]["matcher"] == "*"


class TestCleanupTaskSettings:
    """Tests for _cleanup_task_settings helper."""

    def test_cleanup_task_settings_removes_file(self, tmp_path: Path) -> None:
        """_cleanup_task_settings removes the task settings file."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")

        # Create the file first
        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )
        assert out_path.exists()

        # Now clean it up
        _cleanup_task_settings("abc123", settings_dir=settings_dir)
        assert not out_path.exists()

    def test_cleanup_task_settings_missing_file_silent(self, tmp_path: Path) -> None:
        """_cleanup_task_settings is silent when the file doesn't exist."""
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()

        # Should not raise an exception
        _cleanup_task_settings("missing", settings_dir=settings_dir)


class TestReadUserMcpServers:
    """Tests for _read_user_mcp_servers helper."""

    def test_returns_servers_from_user_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ~/.claude/settings.json exists with mcpServers, return them."""
        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text(json.dumps({
            "mcpServers": {
                "loom": {"command": "uv", "args": ["run", "python", "-m", "loom.mcp"]},
                "weft": {"type": "http", "url": "https://weft.example/mcp"},
            }
        }))
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        servers = _read_user_mcp_servers()

        assert set(servers.keys()) == {"loom", "weft"}
        assert servers["loom"]["command"] == "uv"

    def test_missing_file_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing ~/.claude/settings.json returns empty dict, doesn't raise."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        assert _read_user_mcp_servers() == {}

    def test_no_mcp_servers_key_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings file without mcpServers key returns empty dict."""
        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"allow": []}})
        )
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        assert _read_user_mcp_servers() == {}

    def test_malformed_json_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed JSON returns empty dict, doesn't raise (logs warning)."""
        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text("{not valid json")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        assert _read_user_mcp_servers() == {}


class TestWriteTaskSettingsInjectsMcpServers:
    """_write_task_settings copies user mcpServers into the task settings."""

    def test_user_servers_injected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User-scope mcpServers land in the generated task settings file."""
        fake_home = tmp_path / "home"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / "settings.json").write_text(json.dumps({
            "mcpServers": {
                "loom": {"command": "uv", "args": ["run", "python", "-m", "loom.mcp"]},
            }
        }))
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        written = json.loads(out_path.read_text())
        assert "mcpServers" in written
        assert written["mcpServers"]["loom"]["command"] == "uv"
        # Hooks remain alongside MCP servers.
        assert "hooks" in written

    def test_no_user_servers_means_no_mcp_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When user has no mcpServers, the task settings file omits the key
        entirely (not an empty dict) so Claude Code's existing behavior is
        unchanged for users who don't use MCP."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "event.py").write_text("# event hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        written = json.loads(out_path.read_text())
        assert "mcpServers" not in written
        assert "hooks" in written


class TestBuildSpawnEnvForwardsPath:
    """_build_spawn_env forwards the daemon's current PATH so spawned tabs
    don't inherit the (frozen-at-session-create) zellij-server PATH."""

    @pytest.mark.asyncio
    async def test_path_present_in_spawn_env(
        self, in_memory_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The spawn env dict carries PATH copied from os.environ."""
        monkeypatch.setenv("PATH", "/test/bin:/usr/local/bin:/usr/bin")
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())

        env = registry._build_spawn_env("task-xyz")

        assert env["PATH"] == "/test/bin:/usr/local/bin:/usr/bin"
        # Bridge-owned keys still set.
        assert env["CC_DISCORD_TASK_ID"] == "task-xyz"
        assert "BRIDGE_URL" in env

    @pytest.mark.asyncio
    async def test_path_empty_when_unset(
        self, in_memory_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing PATH in env yields empty string, doesn't raise."""
        monkeypatch.delenv("PATH", raising=False)
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())

        env = registry._build_spawn_env("task-xyz")

        assert env["PATH"] == ""
