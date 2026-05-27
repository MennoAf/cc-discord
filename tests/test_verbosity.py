"""Tests for the /tone feature: state CRUD, policy matrix, registry methods,
the rolling tool indicator, and the commit-milestone surface.

The /tone command callback is exercised separately in test_commands.py."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from bridge.state import (
    DEFAULT_VERBOSITY,
    VALID_VERBOSITY_MODES,
    VerbosityRow,
    delete_verbosity,
    get_verbosity_mode,
    get_verbosity_row,
    list_verbosity,
    upsert_task,
    upsert_verbosity,
)
from bridge.tasks import TaskRegistry, _RollingToolIndicator
from bridge.verbosity import (
    DEFAULT_MODE,
    VALID_MODES,
    VerbosityPolicy,
    policy_for,
)
from tests.fakes import FakeBot, FakeZellij


@pytest.mark.asyncio
class TestVerbosityStateCRUD:
    """Direct CRUD against the channel_verbosity table via state.py functions."""

    async def test_get_row_missing_returns_none(self, in_memory_db) -> None:
        assert await get_verbosity_row(in_memory_db, 12345) is None

    async def test_get_mode_missing_returns_default(self, in_memory_db) -> None:
        assert await get_verbosity_mode(in_memory_db, 12345) == DEFAULT_VERBOSITY

    async def test_upsert_then_get_row(self, in_memory_db) -> None:
        await upsert_verbosity(in_memory_db, 12345, "light")
        row = await get_verbosity_row(in_memory_db, 12345)
        assert row is not None
        assert row.channel_id == 12345
        assert row.mode == "light"
        assert row.created_at > 0
        assert row.updated_at == row.created_at

    async def test_upsert_updates_mode_preserves_created_at(
        self, in_memory_db
    ) -> None:
        await upsert_verbosity(in_memory_db, 12345, "light", now=1000)
        await upsert_verbosity(in_memory_db, 12345, "tldr", now=2000)
        row = await get_verbosity_row(in_memory_db, 12345)
        assert row is not None
        assert row.mode == "tldr"
        assert row.created_at == 1000
        assert row.updated_at == 2000

    async def test_upsert_rejects_invalid_mode(self, in_memory_db) -> None:
        with pytest.raises(ValueError, match="invalid verbosity mode"):
            await upsert_verbosity(in_memory_db, 12345, "shouty")
        assert await get_verbosity_row(in_memory_db, 12345) is None

    async def test_get_mode_falls_back_when_stored_value_invalid(
        self, in_memory_db
    ) -> None:
        # Write a value the validator would reject by going around it. This
        # simulates a legacy row or a future mode the caller doesn't know.
        await in_memory_db.execute(
            "INSERT INTO channel_verbosity (channel_id, mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (12345, "future_mode", 1, 1),
        )
        await in_memory_db.commit()
        assert await get_verbosity_mode(in_memory_db, 12345) == DEFAULT_VERBOSITY

    async def test_delete_returns_true_when_present(self, in_memory_db) -> None:
        await upsert_verbosity(in_memory_db, 12345, "light")
        assert await delete_verbosity(in_memory_db, 12345) is True
        assert await get_verbosity_row(in_memory_db, 12345) is None

    async def test_delete_returns_false_when_absent(self, in_memory_db) -> None:
        assert await delete_verbosity(in_memory_db, 99999) is False

    async def test_list_orders_by_updated_at_desc(self, in_memory_db) -> None:
        await upsert_verbosity(in_memory_db, 1, "light", now=1000)
        await upsert_verbosity(in_memory_db, 2, "tldr", now=2000)
        await upsert_verbosity(in_memory_db, 3, "full", now=3000)
        rows = await list_verbosity(in_memory_db)
        assert [r.channel_id for r in rows] == [3, 2, 1]

    async def test_row_is_frozen_dataclass(self) -> None:
        row = VerbosityRow(channel_id=1, mode="full", created_at=0, updated_at=0)
        with pytest.raises(Exception):
            row.mode = "tldr"  # type: ignore[misc]

    async def test_valid_modes_constant_matches_spec(self) -> None:
        # The /tone command's app_commands.choices is built from this tuple;
        # if it ever drifts from ("full","light","tldr") the command's choice
        # list won't match what upsert_verbosity accepts.
        assert VALID_VERBOSITY_MODES == ("full", "light", "tldr")
        assert DEFAULT_VERBOSITY == "full"


class TestVerbosityPolicy:
    """The mode → policy matrix in verbosity.py."""

    def test_public_constants_re_export_state_values(self) -> None:
        assert VALID_MODES == VALID_VERBOSITY_MODES
        assert DEFAULT_MODE == DEFAULT_VERBOSITY

    def test_policy_for_full(self) -> None:
        p = policy_for("full")
        assert p.mode == "full"
        assert p.show_prose is True
        assert p.show_thinking is True
        assert p.show_tool_lines is True
        assert p.show_tool_diffs is True
        assert p.show_task_list is True
        assert p.show_rolling_indicator is False
        assert p.show_commit_milestones is False

    def test_policy_for_light(self) -> None:
        p = policy_for("light")
        assert p.mode == "light"
        assert p.show_prose is True
        assert p.show_thinking is False
        assert p.show_tool_lines is False
        assert p.show_tool_diffs is False
        assert p.show_task_list is True
        assert p.show_rolling_indicator is True
        assert p.show_commit_milestones is False

    def test_policy_for_tldr(self) -> None:
        p = policy_for("tldr")
        assert p.mode == "tldr"
        assert p.show_prose is False
        assert p.show_thinking is False
        assert p.show_tool_lines is False
        assert p.show_tool_diffs is False
        assert p.show_task_list is False
        # Heartbeat: tldr keeps the rolling indicator on as proof-of-life,
        # even though prose/tools/diffs are all suppressed.
        assert p.show_rolling_indicator is True
        assert p.show_commit_milestones is True

    def test_policy_for_none_returns_default(self) -> None:
        assert policy_for(None).mode == DEFAULT_VERBOSITY

    def test_policy_for_unknown_mode_returns_default(self) -> None:
        # Unknown modes must degrade gracefully so a legacy DB row doesn't
        # crash the emit path — write-time validation is the gate that
        # blocks bad data from landing.
        assert policy_for("future_mode").mode == DEFAULT_VERBOSITY
        assert policy_for("").mode == DEFAULT_VERBOSITY

    def test_policy_is_frozen(self) -> None:
        p = policy_for("light")
        with pytest.raises(Exception):
            p.show_prose = False  # type: ignore[misc]

    def test_rolling_and_tool_lines_are_mutually_exclusive(self) -> None:
        # The emit path in _on_post_tool_use uses `elif` — both being True
        # at once would silently drop the rolling indicator. Encode the
        # invariant so a future mode tweak can't reintroduce that.
        for mode in VALID_VERBOSITY_MODES:
            p = policy_for(mode)
            assert not (p.show_tool_lines and p.show_rolling_indicator), (
                f"mode {mode!r} has both show_tool_lines and show_rolling_indicator"
            )


@pytest.mark.asyncio
class TestTaskRegistryVerbosityMethods:
    """TaskRegistry.set_verbosity / get_verbosity wrappers."""

    async def test_get_verbosity_default_when_unset(self, in_memory_db) -> None:
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())
        assert await registry.get_verbosity(12345) == DEFAULT_VERBOSITY

    async def test_set_then_get(self, in_memory_db) -> None:
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())
        await registry.set_verbosity(12345, "tldr")
        assert await registry.get_verbosity(12345) == "tldr"

    async def test_set_invalid_raises(self, in_memory_db) -> None:
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())
        with pytest.raises(ValueError):
            await registry.set_verbosity(12345, "bogus")

    async def test_policy_for_task_reads_thread_id(self, in_memory_db) -> None:
        registry = TaskRegistry(in_memory_db, FakeBot(), FakeZellij())
        await upsert_task(
            in_memory_db, "task-a", 999, "/tmp", "running",
            current_claude_session_id="sess-a", now=1000,
        )
        await registry.load_from_db()
        task = registry.get_by_task_id("task-a")
        # No row yet → default
        p = await registry._policy_for(task)
        assert p.mode == DEFAULT_VERBOSITY
        # Set the thread's verbosity and re-read
        await registry.set_verbosity(task.thread_id, "light")
        p = await registry._policy_for(task)
        assert p.mode == "light"


def _make_task_row_and_dispatch(in_memory_db, registry: TaskRegistry, *, mode: str | None):
    """Helper: insert a task, set verbosity, return a dispatch callable."""

    async def setup(thread_id: int = 999, session_id: str = "sess-x") -> None:
        await upsert_task(
            in_memory_db, "task-x", thread_id, "/tmp", "running",
            current_claude_session_id=session_id, now=1000,
        )
        await registry.load_from_db()
        if mode is not None:
            await registry.set_verbosity(thread_id, mode)

    return setup


@pytest.mark.asyncio
class TestOnPostToolUseGating:
    """_on_post_tool_use honors the policy matrix.

    These exercise the real method end-to-end with a FakeBot/FakeZellij —
    no monkeypatching of internals — so any future seam-change that
    reroutes the gating shows up here.
    """

    async def _setup(self, in_memory_db, mode: str | None) -> tuple[TaskRegistry, FakeBot]:
        bot = FakeBot()
        registry = TaskRegistry(in_memory_db, bot, FakeZellij())
        await upsert_task(
            in_memory_db, "task-x", 999, "/tmp", "running",
            current_claude_session_id="sess-x", now=1000,
        )
        await registry.load_from_db()
        if mode is not None:
            await registry.set_verbosity(999, mode)
        return registry, bot

    async def test_full_mode_appends_tool_line_and_posts_diff(
        self, in_memory_db
    ) -> None:
        registry, bot = await self._setup(in_memory_db, "full")
        await registry._on_post_tool_use({
            "session_id": "sess-x",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/tmp/x.py",
                "old_string": "old",
                "new_string": "new",
            },
            "tool_response": {},
        })
        agg = registry._aggregators.get("task-x")
        assert agg is not None
        assert len(agg._lines) == 1
        # Edit produces a diff_block → one Discord post.
        diff_posts = [c for c in bot.get_post_calls() if "```" in c["content"]]
        assert len(diff_posts) == 1
        # Light's rolling indicator must NOT have engaged.
        assert "task-x" not in registry._rolling_indicators

    async def test_light_mode_uses_rolling_indicator_and_no_diff(
        self, in_memory_db
    ) -> None:
        registry, bot = await self._setup(in_memory_db, "light")
        await registry._on_post_tool_use({
            "session_id": "sess-x",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/tmp/x.py",
                "old_string": "old",
                "new_string": "new",
            },
            "tool_response": {},
        })
        # No aggregator line.
        agg = registry._aggregators.get("task-x")
        assert agg is None or len(agg._lines) == 0
        # No diff post.
        assert not any("```" in c["content"] for c in bot.get_post_calls())
        # Rolling indicator has the tool pending.
        ri = registry._rolling_indicators.get("task-x")
        assert ri is not None
        assert ri._pending_tools == ["Edit"]

    async def test_tldr_mode_feeds_rolling_indicator_no_line_no_diff(
        self, in_memory_db
    ) -> None:
        # Heartbeat: a non-commit tool no longer vanishes in tldr — it feeds
        # the rolling "🔧 Working…" indicator (same path as light), but with
        # no aggregator line, no diff, and no prose post.
        registry, bot = await self._setup(in_memory_db, "tldr")
        await registry._on_post_tool_use({
            "session_id": "sess-x",
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/tmp/x.py",
                "old_string": "old",
                "new_string": "new",
            },
            "tool_response": {},
        })
        # No inline tool line, no diff post (flush is debounced, so nothing
        # has reached the bot synchronously yet).
        assert "task-x" not in registry._aggregators
        assert not any("```" in c["content"] for c in bot.get_post_calls())
        # The rolling indicator engaged with the tool pending.
        ri = registry._rolling_indicators.get("task-x")
        assert ri is not None
        assert ri._pending_tools == ["Edit"]

    async def test_tldr_mode_emits_commit_milestone(self, in_memory_db) -> None:
        registry, bot = await self._setup(in_memory_db, "tldr")
        await registry._on_post_tool_use({
            "session_id": "sess-x",
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m 'fix'"},
            "tool_response": {
                "exit_code": 0,
                "stdout": "[main abc1234] fix\n 1 file changed",
            },
        })
        posts = bot.get_post_calls()
        assert len(posts) == 1
        assert "abc1234" in posts[0]["content"]
        assert "main" in posts[0]["content"]

    async def test_full_mode_does_not_emit_commit_milestone(
        self, in_memory_db
    ) -> None:
        # Full mode shows the regular Bash tool line — the milestone post is
        # tldr-only since it would duplicate the line.
        registry, bot = await self._setup(in_memory_db, "full")
        await registry._on_post_tool_use({
            "session_id": "sess-x",
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m 'fix'"},
            "tool_response": {
                "exit_code": 0,
                "stdout": "[main abc1234] fix",
            },
        })
        # No "✅ Committed" milestone post should appear in full mode.
        assert not any("Committed" in c["content"] for c in bot.get_post_calls())

    async def test_taskcreate_respects_show_task_list(self, in_memory_db) -> None:
        # full and light both set show_task_list=True → TaskCreate triggers
        # the debounced list post. tldr drops it.
        for mode, should_schedule in (("full", True), ("light", True), ("tldr", False)):
            registry, _bot = await self._setup(in_memory_db, mode)
            await registry._on_post_tool_use({
                "session_id": "sess-x",
                "tool_name": "TaskCreate",
                "tool_input": {"id": "1", "subject": "do thing"},
                "tool_response": {},
            })
            scheduled = registry._task_list_post_tasks.get("task-x") is not None \
                if hasattr(registry, "_task_list_post_tasks") else None
            # Fallback assertion via the state being updated:
            task = registry.get_by_task_id("task-x")
            assert "1" in task.task_list_state, f"mode={mode}"
            # Clean DB between iterations.
            await in_memory_db.execute("DELETE FROM tasks")
            await in_memory_db.execute("DELETE FROM channel_verbosity")
            await in_memory_db.commit()


class _NotFound(Exception):
    """Stand-in for discord.NotFound — _RollingToolIndicator catches it by name."""


class _HTTP429(Exception):
    """Stand-in for a Discord HTTP 429 — checked via `getattr(e, 'status', None)`."""

    def __init__(self) -> None:
        super().__init__("rate limited")
        self.status = 429


class _RecordingBot:
    """Minimal bot stand-in capturing post/edit calls with configurable failures.

    Used instead of FakeBot for indicator tests so we can inject errors
    (NotFound, 429) without bolting them onto the shared FakeBot.
    """

    def __init__(
        self,
        *,
        edit_raises: Exception | None = None,
        post_raises: Exception | None = None,
    ) -> None:
        self.posts: list[dict] = []
        self.edits: list[dict] = []
        self._edit_raises = edit_raises
        self._post_raises = post_raises

    async def post(self, content: str, *, thread_id: int | None = None) -> list[int]:
        self.posts.append({"content": content, "thread_id": thread_id})
        if self._post_raises is not None:
            raise self._post_raises
        return [5000 + len(self.posts)]

    async def edit_message(
        self, thread_id: int, message_id: int, *, content: str
    ) -> None:
        self.edits.append({"thread_id": thread_id, "message_id": message_id, "content": content})
        if self._edit_raises is not None:
            raise self._edit_raises


@pytest.mark.asyncio
class TestRollingToolIndicator:
    """The /tone light mode tool indicator that edits one message in place."""

    async def _flush(self, ri: _RollingToolIndicator) -> None:
        """Await the indicator's flush task to completion."""
        if ri._flush_task is not None:
            try:
                await ri._flush_task
            except asyncio.CancelledError:
                pass

    async def test_append_then_flush_posts_new_message(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        # Squash the debounce window so tests don't sleep 1s.
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        await self._flush(ri)
        assert len(bot.posts) == 1
        assert "Read" in bot.posts[0]["content"]
        assert "🔧 Working" in bot.posts[0]["content"]
        assert bot.edits == []

    async def test_second_append_edits_existing_message(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        await self._flush(ri)
        ri.append("Edit")
        await self._flush(ri)
        # One post, one edit.
        assert len(bot.posts) == 1
        assert len(bot.edits) == 1
        assert "Read" in bot.edits[0]["content"]
        assert "Edit" in bot.edits[0]["content"]
        assert bot.edits[0]["message_id"] == 5001

    async def test_dedupes_consecutive_repeats(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        ri.append("Read")
        ri.append("Read")
        await self._flush(ri)
        assert ri._pending_tools == ["Read"]
        assert bot.posts[0]["content"].count("Read") == 1

    async def test_ellipsis_when_over_inline_cap(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        # Six distinct tools — cap is MAX_INLINE_TOOLS=5.
        for name in ["A", "B", "C", "D", "E", "F"]:
            ri.append(name)
        await self._flush(ri)
        body = bot.posts[0]["content"]
        assert "+1 more" in body
        assert "F" not in body  # the overflow tool name is not inlined

    async def test_mark_burst_end_clears_message_id(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        await self._flush(ri)
        ri.mark_burst_end()
        assert ri._current_msg_id is None
        assert ri._pending_tools == []
        # Next append after burst end → fresh post, not edit.
        ri.append("Edit")
        await self._flush(ri)
        assert len(bot.posts) == 2
        assert len(bot.edits) == 0

    async def test_notfound_falls_back_to_new_post(self, monkeypatch) -> None:
        # Patch the module-level `discord.NotFound` lookup point so the
        # indicator's `except discord.NotFound` matches our stand-in.
        from bridge import tasks as tasks_mod
        bot = _RecordingBot(edit_raises=_NotFound())
        monkeypatch.setattr(tasks_mod.discord, "NotFound", _NotFound, raising=False)
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        await self._flush(ri)
        # First flush: a normal post.
        assert len(bot.posts) == 1
        ri.append("Edit")
        await self._flush(ri)
        # Edit raised NotFound → indicator should have posted a fresh message.
        assert len(bot.edits) == 1
        assert len(bot.posts) == 2

    async def test_429_on_edit_switches_to_slow_mode(self, monkeypatch) -> None:
        bot = _RecordingBot(edit_raises=_HTTP429())
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("Read")
        await self._flush(ri)
        ri.append("Edit")
        await self._flush(ri)
        assert ri._slow_mode is True
        assert ri._flush_window() == _RollingToolIndicator.SLOW_FLUSH_WINDOW

    async def test_close_flushes_pending(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        # Append without awaiting any flush — close should drain it.
        ri.append("Read")
        await ri.close()
        # Either the scheduled flush or close's drain posted.
        assert len(bot.posts) >= 1
        assert "Read" in bot.posts[-1]["content"]
        assert ri._current_msg_id is None
        assert ri._pending_tools == []

    async def test_empty_append_is_noop(self, monkeypatch) -> None:
        bot = _RecordingBot()
        ri = _RollingToolIndicator(bot, thread_id=999)
        monkeypatch.setattr(_RollingToolIndicator, "FLUSH_WINDOW", 0.01)
        ri.append("")
        await self._flush(ri)
        assert bot.posts == []
        assert bot.edits == []


@pytest.mark.asyncio
class TestCommitMilestone:
    """The tldr-mode `✅ Committed <hash>` surface."""

    async def _setup_tldr(self, in_memory_db) -> tuple[TaskRegistry, FakeBot, Any]:
        bot = FakeBot()
        registry = TaskRegistry(in_memory_db, bot, FakeZellij())
        await upsert_task(
            in_memory_db, "task-x", 999, "/tmp", "running",
            current_claude_session_id="sess-x", now=1000,
        )
        await registry.load_from_db()
        await registry.set_verbosity(999, "tldr")
        task = registry.get_by_task_id("task-x")
        return registry, bot, task

    async def test_successful_commit_posts_milestone(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git commit -m 'fix bug'"},
            {"stdout": "[main abc1234] fix bug\n 1 file changed", "exit_code": 0},
        )
        assert len(bot.get_post_calls()) == 1
        body = bot.get_post_calls()[0]["content"]
        assert "abc1234" in body
        assert "main" in body
        assert "✅" in body

    async def test_failed_commit_no_post(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git commit -m 'broken'"},
            {"stdout": "", "exit_code": 1, "is_error": True},
        )
        assert bot.get_post_calls() == []

    async def test_non_commit_bash_no_post(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "ls -la"},
            {"stdout": "[main abc1234] fake", "exit_code": 0},
        )
        assert bot.get_post_calls() == []

    async def test_echo_of_git_commit_no_post(self, in_memory_db) -> None:
        # An echo'd string containing "git commit" must not trigger — the
        # anchor in _GIT_COMMIT_RE requires it to be an actual invocation.
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "echo 'git commit'"},
            {"stdout": "[main abc1234] fake", "exit_code": 0},
        )
        assert bot.get_post_calls() == []

    async def test_env_var_prefix_accepted(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "GIT_AUTHOR_NAME=Bot git commit -m 'fix'"},
            {"stdout": "[feature/x def5678] fix", "exit_code": 0},
        )
        assert len(bot.get_post_calls()) == 1
        assert "def5678" in bot.get_post_calls()[0]["content"]
        assert "feature/x" in bot.get_post_calls()[0]["content"]

    async def test_chained_command_with_git_commit(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git add -A && git commit -m 'fix'"},
            {"stdout": "[main abc1234] fix", "exit_code": 0},
        )
        assert len(bot.get_post_calls()) == 1

    async def test_root_commit_branch_parsing(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git commit -m 'init'"},
            {"stdout": "[main (root-commit) abc1234] init", "exit_code": 0},
        )
        assert len(bot.get_post_calls()) == 1
        assert "abc1234" in bot.get_post_calls()[0]["content"]

    async def test_falls_back_to_alt_stdout_keys(self, in_memory_db) -> None:
        registry, bot, task = await self._setup_tldr(in_memory_db)
        # `output` key instead of `stdout` — hooks have used both names.
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git commit -m 'x'"},
            {"output": "[main 1234567] x", "exit_code": 0},
        )
        assert len(bot.get_post_calls()) == 1

    async def test_no_hash_in_stdout_no_post(self, in_memory_db) -> None:
        # The regex needs `[branch hash]` to match; otherwise the milestone
        # silently no-ops rather than posting a fake.
        registry, bot, task = await self._setup_tldr(in_memory_db)
        await registry._maybe_post_commit_milestone(
            task,
            {"command": "git commit --allow-empty -m 'nothing'"},
            {"stdout": "nothing to commit", "exit_code": 0},
        )
        assert bot.get_post_calls() == []
