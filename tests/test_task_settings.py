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
