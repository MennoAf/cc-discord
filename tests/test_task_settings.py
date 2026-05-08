"""Tests for task settings file generation and cleanup."""

import json
from pathlib import Path

from bridge.tasks import _cleanup_task_settings, _write_task_settings


class TestWriteTaskSettings:
    """Tests for _write_task_settings helper."""

    def test_write_task_settings_creates_file(self, tmp_path: Path) -> None:
        """_write_task_settings creates a JSON file with the expected structure."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        # Create dummy hook files
        (hooks_dir / "event.py").write_text("# event hook")
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        assert out_path == settings_dir / "abc123.json"
        assert out_path.exists()

    def test_write_task_settings_json_structure(self, tmp_path: Path) -> None:
        """_write_task_settings generates valid JSON with all 8 hook events."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data.get("hooks", {})

        # Verify all 8 events are present
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

    def test_write_task_settings_hook_paths_absolute(self, tmp_path: Path) -> None:
        """_write_task_settings uses absolute paths for hook scripts."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data["hooks"]

        # Check PreToolUse points to pretooluse-approve.py
        pretooluse_hooks = hooks["PreToolUse"][0]["hooks"]
        assert len(pretooluse_hooks) == 1
        cmd = pretooluse_hooks[0]["command"]
        assert "pretooluse-approve.py" in cmd
        assert str(hooks_dir) in cmd

        # Check one event hook points to event.py
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
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        assert out_path.parent.exists()
        assert out_path.exists()

    def test_write_task_settings_each_event_one_matcher(self, tmp_path: Path) -> None:
        """_write_task_settings creates exactly one matcher per event."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

        out_path = _write_task_settings(
            "abc123", settings_dir=settings_dir, hooks_dir=hooks_dir
        )

        data = json.loads(out_path.read_text())
        hooks = data["hooks"]

        for event, matchers in hooks.items():
            assert len(matchers) == 1, f"Event {event} should have exactly 1 matcher"
            assert matchers[0]["matcher"] == "*"
            assert len(matchers[0]["hooks"]) == 1


class TestCleanupTaskSettings:
    """Tests for _cleanup_task_settings helper."""

    def test_cleanup_task_settings_removes_file(self, tmp_path: Path) -> None:
        """_cleanup_task_settings removes the task settings file."""
        settings_dir = tmp_path / "settings"
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        (hooks_dir / "event.py").write_text("# event hook")
        (hooks_dir / "pretooluse-approve.py").write_text("# pretooluse hook")

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
