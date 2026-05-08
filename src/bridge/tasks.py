"""Task and TaskRegistry for managing discord-driven sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

import bridge as _bridge_pkg
from bridge import tool_summary, transcript
from bridge.listener import MessageLike
from bridge.state import TaskRow, list_active_tasks, upsert_task
from bridge.zellij import ZellijError, ZellijManager

if TYPE_CHECKING:
    from bridge.approvals import ApprovalRouter
    from bridge.bot import Bot

logger = logging.getLogger(__name__)

# Task-scoped settings directory
TASK_SETTINGS_DIR = Path.home() / ".local" / "state" / "claude-discord-bridge" / "task-settings"

# Hook scripts directory — resolved at import time for test monkeypatch support
HOOKS_DIR = Path(_bridge_pkg.__file__).parent.parent.parent / "hooks"


def _write_task_settings(
    task_id: str, *, settings_dir: Path = TASK_SETTINGS_DIR, hooks_dir: Path = HOOKS_DIR
) -> Path:
    """Generate the task-scoped settings JSON. Returns the absolute path written.

    The file registers `event.py` for the read-only event types and
    `pretooluse-approve.py` for `PreToolUse`. All paths are absolute.
    """
    settings_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings_dir / f"{task_id}.json"
    event_script = str(hooks_dir / "event.py")
    pretooluse_script = str(hooks_dir / "pretooluse-approve.py")

    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": pretooluse_script},
                    ],
                },
            ],
            "SessionStart": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "UserPromptSubmit": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "PostToolUse": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "PostToolUseFailure": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "Stop": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "Notification": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
            "SessionEnd": [
                {"matcher": "*", "hooks": [{"type": "command", "command": event_script}]},
            ],
        }
    }

    out_path.write_text(json.dumps(settings, indent=2))
    return out_path


def _cleanup_task_settings(task_id: str, *, settings_dir: Path = TASK_SETTINGS_DIR) -> None:
    """Remove the task-scoped settings file. Idempotent — silent on missing file."""
    p = settings_dir / f"{task_id}.json"
    try:
        p.unlink()
    except FileNotFoundError:
        return
    except Exception:
        logger.exception("failed to remove task settings file %s", p)


class TaskSpawnError(Exception):
    """Raised when a task cannot be spawned."""

    pass


class TaskNotFound(Exception):
    """Raised when a task cannot be found by its ID."""

    pass


class TaskRestartError(Exception):
    """Raised when a task cannot be restarted (e.g., no session to resume)."""

    pass


class _ToolSummaryAggregator:
    """Collects PostToolUse summaries within a 1s window and flushes as one Discord message."""

    FLUSH_WINDOW = 1.0  # seconds

    def __init__(self, bot: Bot, thread_id: int) -> None:
        self._bot = bot
        self._thread_id = thread_id
        self._lines: list[str] = []
        self._flush_task: asyncio.Task | None = None

    def append(self, line: str) -> None:
        self._lines.append(line)
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_window())

    async def _flush_after_window(self) -> None:
        try:
            await asyncio.sleep(self.FLUSH_WINDOW)
        except asyncio.CancelledError:
            return
        if not self._lines:
            return
        # Snapshot lines before clearing so we can restore if the post is cancelled
        local_lines = list(self._lines)
        self._lines.clear()
        body = "\n".join(local_lines)
        try:
            await self._bot.post(body, thread_id=self._thread_id)
        except asyncio.CancelledError:
            # Restore lines so flush_now can re-emit them
            self._lines[:0] = local_lines
            raise
        except Exception:
            logger.exception("failed to post tool summary chunk")

    async def flush_now(self) -> None:
        """Force flush (called on Stop / SessionEnd to avoid orphaned summaries)."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
        if self._lines:
            body = "\n".join(self._lines)
            self._lines.clear()
            try:
                await self._bot.post(body, thread_id=self._thread_id)
            except Exception:
                logger.exception("failed to flush final tool summary chunk")


@dataclass
class Task:
    """An in-memory task representation."""

    task_id: str
    thread_id: int
    zellij_pane_id: str | None
    cwd: str
    status: str
    current_claude_session_id: str | None
    current_transcript_path: str | None
    created_at: int
    last_activity: int

    @classmethod
    def from_row(cls, row: TaskRow) -> "Task":
        """Convert a frozen TaskRow to a mutable Task."""
        return cls(
            task_id=row.task_id,
            thread_id=row.thread_id,
            zellij_pane_id=row.zellij_pane_id,
            cwd=row.cwd,
            status=row.status,
            current_claude_session_id=row.current_claude_session_id,
            current_transcript_path=row.current_transcript_path,
            created_at=row.created_at,
            last_activity=row.last_activity,
        )


class TaskRegistry:
    """In-memory registry of tasks with database persistence."""

    _HANDLERS: dict[str, str] = {
        "SessionStart": "_on_session_start",
        "UserPromptSubmit": "_on_user_prompt_submit",
        "PostToolUse": "_on_post_tool_use",
        "PostToolUseFailure": "_on_post_tool_use_failure",
        "Stop": "_on_stop",
        "Notification": "_on_notification",
        "SessionEnd": "_on_session_end",
        "SubagentStop": "_on_subagent_stop",
        "PreCompact": "_on_pre_compact",
    }

    def __init__(
        self,
        conn: aiosqlite.Connection,
        bot: "Bot",
        zellij: ZellijManager,
        approval_router: "ApprovalRouter | None" = None,
    ) -> None:
        """Initialize with database connection, bot, zellij manager, and optional approval router."""
        self._conn = conn
        self._bot = bot
        self._zellij = zellij
        self._approval_router = approval_router
        self._by_task_id: dict[str, Task] = {}
        self._by_thread_id: dict[int, Task] = {}
        self._by_session_id: dict[str, Task] = {}
        self._stop_futures: dict[str, asyncio.Future] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._aggregators: dict[str, _ToolSummaryAggregator] = {}
        self._tui_handler_tasks: dict[str, asyncio.Task] = {}

    async def load_from_db(self) -> None:
        """Load active tasks from database into memory maps."""
        rows = await list_active_tasks(self._conn)
        for row in rows:
            if row.status not in {"stopped", "crashed"}:
                task = Task.from_row(row)
                await self._index(task)

    def get_by_task_id(self, task_id: str) -> Task | None:
        """Get task by task_id."""
        return self._by_task_id.get(task_id)

    def get_by_thread_id(self, thread_id: int) -> Task | None:
        """Get task by thread_id."""
        return self._by_thread_id.get(thread_id)

    def get_by_session_id(self, session_id: str) -> Task | None:
        """Get task by current_claude_session_id."""
        return self._by_session_id.get(session_id)

    async def _index(self, task: Task, prev_session_id: str | None = None) -> None:
        """Update all three maps for a task. Removes prior session_id entry if it changed."""
        # Remove old session_id entry if this task previously had a different session_id
        if prev_session_id and prev_session_id in self._by_session_id:
            if self._by_session_id[prev_session_id].task_id == task.task_id:
                del self._by_session_id[prev_session_id]

        # Remove old session_id entry if a different task owns the new session_id
        if task.current_claude_session_id:
            old_task = self._by_session_id.get(task.current_claude_session_id)
            if old_task and old_task.task_id != task.task_id:
                # Different task now owns this session_id; remove the old one
                if old_task.task_id in self._by_task_id:
                    del self._by_task_id[old_task.task_id]
                if old_task.thread_id in self._by_thread_id:
                    del self._by_thread_id[old_task.thread_id]

        # Index the task
        self._by_task_id[task.task_id] = task
        self._by_thread_id[task.thread_id] = task
        if task.current_claude_session_id:
            self._by_session_id[task.current_claude_session_id] = task

    async def _persist(self, task: Task) -> None:
        """Persist task to database."""
        await upsert_task(
            self._conn,
            task.task_id,
            task.thread_id,
            task.cwd,
            task.status,
            zellij_pane_id=task.zellij_pane_id,
            current_claude_session_id=task.current_claude_session_id,
            current_transcript_path=task.current_transcript_path,
        )

    def _build_spawn_env(self, task_id: str) -> dict[str, str]:
        """Build environment dict for spawning a claude process.

        Returns os.environ.copy() with CC_DISCORD_TASK_ID and BRIDGE_URL added.
        """
        env = os.environ.copy()
        env["CC_DISCORD_TASK_ID"] = task_id
        if "BRIDGE_URL" not in env:
            env["BRIDGE_URL"] = "http://127.0.0.1:8787"
        return env

    async def _is_pane_alive(self, pane_id: str) -> bool:
        """Check if a pane is still running (returns False if exited)."""
        try:
            panes = await self._zellij.list_panes()
            for pane in panes:
                if pane.get("id") == pane_id and not pane.get("exited", True):
                    return True
            return False
        except Exception:
            logger.exception("_is_pane_alive failed for pane %s", pane_id)
            return False

    async def _mark_stopped(self, task: Task) -> None:
        """Mark a task as stopped and archive its thread."""
        task.status = "stopped"
        task.last_activity = int(time.time())
        await self._persist(task)
        await self._archive_thread(task.thread_id)
        # Remove from live indexes; stopped tasks should not be returned by get_by_thread_id/get_by_session_id
        self._by_thread_id.pop(task.thread_id, None)
        if task.current_claude_session_id is not None:
            self._by_session_id.pop(task.current_claude_session_id, None)
        # Clean up typing task, aggregator, and any pending TUI handler task
        await self._stop_typing(task.task_id)
        agg = self._aggregators.pop(task.task_id, None)
        if agg is not None:
            await agg.flush_now()
        handler_task = self._tui_handler_tasks.pop(task.task_id, None)
        if handler_task is not None and not handler_task.done():
            handler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handler_task
        # Clean up the task-scoped settings file
        _cleanup_task_settings(task.task_id)

    async def _archive_thread(self, thread_id: int) -> None:
        """Archive a Discord thread."""
        try:
            await self._bot.archive_thread(thread_id)
        except Exception:
            logger.exception("Failed to archive thread %d", thread_id)

    async def _start_typing(self, task: Task) -> None:
        """Start a typing indicator for a task. Cancels any prior typing task."""
        prev = self._typing_tasks.pop(task.task_id, None)
        if prev is not None and not prev.done():
            prev.cancel()
        self._typing_tasks[task.task_id] = asyncio.create_task(
            self._run_typing(task), name=f"typing-{task.task_id[:8]}"
        )

    async def _run_typing(self, task: Task) -> None:
        """Run typing indicator in background. Lives until cancelled."""
        try:
            channel = await self._bot.fetch_messageable(task.thread_id)
            async with channel.typing():
                # Sleep until cancelled (Stop/Notification cancels us). Discord.py auto-renews
                # the indicator every 5s under the hood; we just need the context to stay open.
                await asyncio.Future()  # never resolves; we live until cancelled
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("typing indicator failed for task %s", task.task_id)

    async def _stop_typing(self, task_id: str) -> None:
        """Cancel and await a typing task."""
        t = self._typing_tasks.pop(task_id, None)
        if t is not None and not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    def _agg_for(self, task: Task) -> _ToolSummaryAggregator:
        """Lookup or create aggregator for a task."""
        agg = self._aggregators.get(task.task_id)
        if agg is None:
            agg = _ToolSummaryAggregator(self._bot, task.thread_id)
            self._aggregators[task.task_id] = agg
        return agg

    async def spawn_task(self, cwd: str, *, prompt: str | None = None) -> Task:
        """Spawn a new claude session in a Discord-bound task.

        1. Validates cwd is a directory.
        2. Generates a task_id UUID.
        3. Creates a Discord thread.
        4. Persists a row with status='spawning'.
        5. Builds env with CC_DISCORD_TASK_ID and BRIDGE_URL.
        6. Spawns claude in zellij via ZellijManager.
        7. Updates row with zellij_pane_id.
        8. Indexes the task in memory.
        9. Returns the Task.

        The `prompt` parameter is accepted for forward compatibility but ignored
        in Phase 2 — Phase 3 will call write_to_pane after SessionStart binding.

        Raises TaskSpawnError if cwd is not a directory or zellij spawn fails.
        """
        # Validate cwd
        if not Path(cwd).is_dir():
            raise TaskSpawnError(f"cwd does not exist: {cwd}")

        # Generate task_id
        task_id = str(uuid.uuid4())

        # Create Discord thread
        thread_name = f"cc · {Path(cwd).name} · {task_id[:8]}"
        thread_id = await self._bot.create_thread(name=thread_name)

        # Persist row with status='spawning'
        now = int(time.time())
        await upsert_task(
            self._conn,
            task_id,
            thread_id,
            cwd,
            "spawning",
            zellij_pane_id=None,
            current_claude_session_id=None,
            current_transcript_path=None,
            now=now,
        )

        # Write task-scoped settings file with bridge hooks
        settings_path = _write_task_settings(task_id)

        # Build env for spawned claude
        env = self._build_spawn_env(task_id)

        # Spawn via zellij; on failure, mark the task as crashed and re-raise
        try:
            pane_id = await self._zellij.spawn_task(
                cwd=cwd,
                env=env,
                pane_name=f"cc-{task_id[:8]}",
                extra_argv=["--settings", str(settings_path)],
            )
        except ZellijError:
            logger.exception(f"spawn_task failed for task_id {task_id}")
            # Mark the task as crashed in the database
            await upsert_task(
                self._conn,
                task_id,
                thread_id,
                cwd,
                "crashed",
                zellij_pane_id=None,
                current_claude_session_id=None,
                current_transcript_path=None,
            )
            # Phase 3 slash-command handler will see this crashed status.
            # TODO: Phase 3 should clean up the Discord thread on failure.
            raise

        # Update row with zellij_pane_id (bump last_activity)
        now2 = int(time.time())
        await upsert_task(
            self._conn,
            task_id,
            thread_id,
            cwd,
            "spawning",
            zellij_pane_id=pane_id,
            current_claude_session_id=None,
            current_transcript_path=None,
            now=now2,
        )

        # Construct and index the Task
        task = Task(
            task_id=task_id,
            thread_id=thread_id,
            zellij_pane_id=pane_id,
            cwd=cwd,
            status="spawning",
            current_claude_session_id=None,
            current_transcript_path=None,
            created_at=now,
            last_activity=now,
        )
        await self._index(task)

        return task

    async def maybe_route_message(self, msg: MessageLike) -> bool:
        """If msg is in a task-bound thread, write to its zellij pane and return True.
        Otherwise return False so the caller falls through to the existing /v1/ask listener.
        """
        thread_id = msg.channel.id
        task = self.get_by_thread_id(thread_id)
        if task is None:
            return False
        if task.zellij_pane_id is None:
            # Task is mid-spawn; treat as no-op so the message goes to the /v1/ask listener
            # (which will also drop it; net effect: silent ignore — fine).
            return False
        if task.status not in ("running", "spawning"):
            # task is stopped/crashed — silent ignore per AC3.6 spirit
            return True
        text = msg.content or ""
        if not text.strip():
            # Discord empty message (likely image-only). Phase 3 doesn't relay images;
            # surface "(image attachment)" placeholder if attachments exist, else swallow.
            if msg.attachments:
                text = "(image attached — image relay not yet supported)"
            else:
                return True  # consumed silently
        await self._zellij.write_to_pane(task.zellij_pane_id, text + "\n")
        task.last_activity = int(time.time())
        await self._persist(task)
        return True

    async def handle_event(self, hook_event_name: str, body: dict) -> None:
        """Dispatch event to appropriate handler by name."""
        handler_name = self._HANDLERS.get(hook_event_name)
        if handler_name is None:
            logger.info(f"Unknown hook event: {hook_event_name}")
            return

        handler = getattr(self, handler_name, None)
        if handler is None:
            logger.warning(f"Handler not found: {handler_name}")
            return

        try:
            await handler(body)
        except Exception:
            logger.exception(f"Error handling {hook_event_name}")

    async def _on_session_start(self, body: dict) -> None:
        """Handle SessionStart event.

        Branches on matcher value:
        - 'startup': first bind. Look up by CC_DISCORD_TASK_ID, populate session/transcript,
          flip status to running, post '🟢 Task started' notice.
        - 'clear':   /clear inside TUI. Rebind same task to new session id, post '🧹' notice.
        - 'compact': /compact inside TUI. Rebind same task to new session id, post '🧰' notice.
        - 'resume':  claude --resume by /restart. Same as startup but no notice (user already
                     saw the /restart command's reply).
        - default:   unknown matcher → fall through to startup behavior (safest).

        Per AC2.4: silently drops SessionStart events with no CC_DISCORD_TASK_ID.
        Also drops if session_id or transcript_path is missing.
        """
        session_id = body.get("session_id")
        transcript_path = body.get("transcript_path")
        env_passthrough = body.get("env_passthrough", {})
        task_id = env_passthrough.get("CC_DISCORD_TASK_ID")
        matcher = body.get("matcher") or "startup"

        logger.info(f"SessionStart: session_id={session_id}, task_id={task_id}, matcher={matcher}")

        # AC2.4: drop SessionStart without task_id, session_id, or transcript_path
        if not task_id or not session_id or not transcript_path:
            if not task_id:
                logger.debug("SessionStart: no CC_DISCORD_TASK_ID in env_passthrough")
            elif not session_id:
                logger.warning("SessionStart: no session_id in body")
            elif not transcript_path:
                logger.warning("SessionStart: no transcript_path in body")
            return

        task = self.get_by_task_id(task_id)
        if not task:
            logger.warning(f"SessionStart: task_id {task_id} not found in registry")
            return

        # Detach from old session id index; re-index under new
        old_session_id = task.current_claude_session_id
        task.current_claude_session_id = session_id
        task.current_transcript_path = transcript_path
        task.status = "running"
        task.last_activity = int(time.time())
        if old_session_id and old_session_id in self._by_session_id:
            del self._by_session_id[old_session_id]
        if session_id:
            self._by_session_id[session_id] = task

        # Persist
        await self._persist(task)

        # Post appropriate notice based on matcher
        notice = None
        if matcher == "startup":
            short_session_id = session_id[:8] if session_id else "?"
            notice = f"🟢 Task started — claude session `{short_session_id}`"
        elif matcher == "clear":
            short_session_id = session_id[:8] if session_id else "?"
            notice = f"🧹 Context cleared (new session: `{short_session_id}`)"
        elif matcher == "compact":
            notice = "🧰 Context compacted"
        elif matcher == "resume":
            # Don't post — the user just ran /restart; spamming is noisy.
            notice = None
        else:
            # Unknown matcher: log and post a small bind notice
            logger.info("SessionStart with unrecognized matcher %r", matcher)
            short_session_id = session_id[:8] if session_id else "?"
            notice = f"🟢 Bound to session `{short_session_id}` (matcher={matcher})"

        if notice:
            try:
                await self._bot.post(notice, thread_id=task.thread_id)
            except Exception:
                logger.exception("failed to post session start notice")

    async def _on_user_prompt_submit(self, body: dict) -> None:
        """Handle UserPromptSubmit event. Start typing indicator and cancel pending TUI prompts."""
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        # Cancel any pending TUI prompts for this thread — user typed in zellij.
        if self._approval_router:
            cancelled = await self._approval_router.cancel_thread_tui(task.thread_id)
            if cancelled > 0:
                logger.info("cancelled %d pending TUI prompts for task %s (answered in zellij)", cancelled, task.task_id)
        task.last_activity = int(time.time())
        await self._persist(task)
        await self._start_typing(task)

    async def _on_post_tool_use(self, body: dict) -> None:
        """Handle PostToolUse event. Append tool summary to aggregator."""
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        task.last_activity = int(time.time())
        await self._persist(task)

        line = tool_summary.summarize(
            body.get("tool_name", "?"),
            body.get("tool_input", {}) or {},
            body.get("tool_response", {}) or {},
        )
        self._agg_for(task).append(line)

    async def _on_post_tool_use_failure(self, body: dict) -> None:
        """Handle PostToolUseFailure event. Force-failure summary."""
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        task.last_activity = int(time.time())
        await self._persist(task)
        # Force-failure: synthesize a tool_response.is_error=True if missing.
        tool_response = (body.get("tool_response") or {}).copy()
        tool_response["is_error"] = True
        line = tool_summary.summarize(
            body.get("tool_name", "?"),
            body.get("tool_input", {}) or {},
            tool_response,
        )
        self._agg_for(task).append(line)

    async def _on_stop(self, body: dict) -> None:
        """Handle Stop event. Cancel pending TUI, stop typing, flush summaries, and post final turn."""
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        # Defensively cancel any pending TUI prompts when Stop fires
        if self._approval_router:
            await self._approval_router.cancel_thread_tui(task.thread_id)
        task.last_activity = int(time.time())
        await self._persist(task)
        await self._stop_typing(task.task_id)

        # Flush pending tool summaries first
        agg = self._aggregators.get(task.task_id)
        if agg is not None:
            await agg.flush_now()

        # Read transcript and post the final assistant turn
        transcript_path = body.get("transcript_path") or task.current_transcript_path
        if transcript_path:
            text = transcript.extract_final_assistant_text(Path(transcript_path))
            if text:
                try:
                    await self._bot.post(text, thread_id=task.thread_id)
                except Exception:
                    logger.exception("failed to post final assistant turn for task %s", task.task_id)

    async def _on_notification(self, body: dict) -> None:
        """Handle Notification event. Stop typing indicator and spawn TUI handlers asynchronously.

        Spawns TUI handler tasks fire-and-forget so the HTTP request returns immediately.
        Handlers are tracked in _tui_handler_tasks for cancellation on stop_task/kill_task.
        """
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        task.last_activity = int(time.time())
        await self._persist(task)
        await self._stop_typing(task.task_id)

        # Cancel any pending tool-summary aggregator (no harm; flush_now is idempotent)
        agg = self._aggregators.get(task.task_id)
        if agg is not None:
            await agg.flush_now()

        transcript_path = body.get("transcript_path") or task.current_transcript_path
        if not transcript_path:
            # Generic stall — spawn handler task
            handler_task = asyncio.create_task(
                self._handle_free_text_stall(task),
                name=f"tui-free_text-{task.task_id[:8]}",
            )
            self._tui_handler_tasks[task.task_id] = handler_task
            return

        pending = transcript.find_latest_unresolved_tool_use(Path(transcript_path))

        if pending and pending["name"] == "AskUserQuestion":
            handler_task = asyncio.create_task(
                self._handle_ask_user_question(task, pending),
                name=f"tui-ask_question-{task.task_id[:8]}",
            )
            self._tui_handler_tasks[task.task_id] = handler_task
        elif pending and pending["name"] == "ExitPlanMode":
            handler_task = asyncio.create_task(
                self._handle_exit_plan_mode(task, pending),
                name=f"tui-exit_plan-{task.task_id[:8]}",
            )
            self._tui_handler_tasks[task.task_id] = handler_task
        else:
            # Generic free-text stall
            handler_task = asyncio.create_task(
                self._handle_free_text_stall(task),
                name=f"tui-free_text-{task.task_id[:8]}",
            )
            self._tui_handler_tasks[task.task_id] = handler_task

    async def _on_session_end(self, body: dict) -> None:
        """Handle SessionEnd event.

        Stops typing indicator, resolves any pending stop_future for the task so stop_task can proceed.
        Cleans up the task-scoped settings file.
        """
        session_id = body.get("session_id")
        if not session_id:
            return
        task = self.get_by_session_id(session_id)
        if task is None:
            return
        await self._stop_typing(task.task_id)
        fut = self._stop_futures.get(task.task_id)
        if fut is not None and not fut.done():
            fut.set_result(None)
        # Always clean up the settings file when the claude process ends, regardless of
        # whether the status was already stopped/crashed or is about to be flipped by Phase 8's
        # logic. Idempotent — missing-file is silent.
        _cleanup_task_settings(task.task_id)

    async def _on_subagent_stop(self, body: dict) -> None:
        """Handle SubagentStop event (no-op in Phase 1)."""
        logger.debug("SubagentStop received")

    async def _on_pre_compact(self, body: dict) -> None:
        """Handle PreCompact event (no-op in Phase 1)."""
        logger.debug("PreCompact received")

    async def _handle_ask_user_question(self, task: Task, pending: dict) -> None:
        """Handle AskUserQuestion prompt: post to Discord with option reactions."""
        if not self._approval_router:
            logger.warning("approval_router not configured; cannot dispatch TUI prompt for task %s", task.task_id)
            return
        questions = pending["input"].get("questions") or []
        if not questions:
            return await self._handle_free_text_stall(task)
        q = questions[0]  # Phase 6 supports the first question only — multi-question is rare
        options = [opt.get("label", "") for opt in (q.get("options") or [])]
        if not options:
            return await self._handle_free_text_stall(task)
        # Build the Discord message body
        lines = [f"❓ **{q.get('question', '?')}**"]
        for i, opt in enumerate(q.get("options") or [], start=1):
            emoji = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"][i - 1] if i <= 4 else f"{i}."
            label = opt.get("label", "")
            desc = opt.get("description", "")
            if desc:
                lines.append(f"{emoji} **{label}** — {desc}")
            else:
                lines.append(f"{emoji} {label}")
        body = "\n".join(lines)

        request_id = str(uuid.uuid4())
        answer, source = await self._approval_router.request_tui_answer(
            request_id=request_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            pane_id=task.zellij_pane_id or "",
            kind="ask_question",
            prompt_body=body,
            options=options,
        )
        await self._inject_to_pane(task, answer, source)

    async def _handle_exit_plan_mode(self, task: Task, pending: dict) -> None:
        """Handle ExitPlanMode prompt: post plan to Discord with approve/reject reactions."""
        if not self._approval_router:
            logger.warning("approval_router not configured; cannot dispatch TUI prompt for task %s", task.task_id)
            return
        plan = pending["input"].get("plan") or "(empty plan)"
        body = (
            f"📋 **Plan ready for review**\n\n{plan}\n\n"
            f"React ✅ to approve, ❌ to reject, or reply in this thread to leave feedback."
        )
        request_id = str(uuid.uuid4())
        answer, source = await self._approval_router.request_tui_answer(
            request_id=request_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            pane_id=task.zellij_pane_id or "",
            kind="exit_plan",
            prompt_body=body,
        )
        await self._inject_to_pane(task, answer, source)

    async def _handle_free_text_stall(self, task: Task) -> None:
        """Handle free-text stall: post generic waiting notice."""
        if not self._approval_router:
            logger.warning("approval_router not configured; cannot dispatch TUI prompt for task %s", task.task_id)
            return
        request_id = str(uuid.uuid4())
        body = "🟡 Claude is waiting for input. Reply in this thread or type in zellij."
        answer, source = await self._approval_router.request_tui_answer(
            request_id=request_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            pane_id=task.zellij_pane_id or "",
            kind="free_text",
            prompt_body=body,
        )
        await self._inject_to_pane(task, answer, source)

    async def _inject_to_pane(self, task: Task, answer: str, source: str) -> None:
        """Inject the answer to the pane. Short-circuit if cancelled/timed out/post failed."""
        if source in ("cancelled", "timeout", "post_failed"):
            return
        if not answer:
            return
        if not task.zellij_pane_id:
            return
        try:
            await self._zellij.write_to_pane(task.zellij_pane_id, answer + "\n")
        except Exception:
            logger.exception("failed to inject TUI answer into pane for task %s", task.task_id)

    async def write_initial_prompt(self, task_id: str, prompt: str) -> None:
        """Write initial prompt to a task's zellij pane after session bind.

        Looks up the task by task_id. If pane_id is None or task is missing,
        logs warning and returns. Otherwise writes prompt + newline and bumps last_activity.
        """
        task = self.get_by_task_id(task_id)
        if task is None:
            logger.warning("write_initial_prompt: task_id %s not found", task_id)
            return
        if task.zellij_pane_id is None:
            logger.warning("write_initial_prompt: task %s has no pane_id", task_id)
            return
        await self._zellij.write_to_pane(task.zellij_pane_id, prompt + "\n")
        task.last_activity = int(time.time())
        await self._persist(task)

    async def list_tasks(self) -> list[Task]:
        """Return active tasks ordered by last_activity DESC.

        Filters out 'stopped' and 'crashed' rows. Returns in-memory tasks;
        load_from_db at boot ensures freshness.
        """
        return sorted(
            (t for t in self._by_task_id.values() if t.status in ("spawning", "running")),
            key=lambda t: t.last_activity,
            reverse=True,
        )

    async def stop_task(self, task_id: str, *, timeout: float = 5.0) -> bool:
        """Gracefully stop a task.

        Sends `/exit\\n` to the pane, waits up to `timeout` for SessionEnd.
        Returns True if SessionEnd was observed, False if it timed out.
        On success or timeout: archives the thread, flips status to 'stopped'.

        Raises TaskNotFound if task_id doesn't exist.
        """
        task = self.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFound(task_id)

        # Guard: already stopped or crashed; idempotent no-op
        if task.status not in {"running", "spawning"}:
            return True

        if task.zellij_pane_id is None:
            # Mid-spawn; treat as immediate fail-safe stop
            await self._mark_stopped(task)
            return True

        # Set up a future that SessionEnd handler will resolve.
        loop = asyncio.get_running_loop()
        self._stop_futures[task_id] = loop.create_future()
        try:
            await self._zellij.write_to_pane(task.zellij_pane_id, "/exit\n")
            try:
                await asyncio.wait_for(self._stop_futures[task_id], timeout=timeout)
                stopped_cleanly = True
            except asyncio.TimeoutError:
                stopped_cleanly = False
        finally:
            self._stop_futures.pop(task_id, None)

        await self._mark_stopped(task)
        return stopped_cleanly

    async def kill_task(self, task_id: str) -> None:
        """Immediately close a task's pane.

        Marks status='crashed' and archives the thread.
        Raises TaskNotFound if task_id doesn't exist.
        """
        task = self.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFound(task_id)

        # Guard: already stopped or crashed; idempotent no-op
        if task.status not in {"running", "spawning"}:
            return

        if task.zellij_pane_id is not None:
            await self._zellij.close_pane(task.zellij_pane_id)

        task.status = "crashed"
        task.last_activity = int(time.time())
        await self._persist(task)
        await self._archive_thread(task.thread_id)
        # Remove from live indexes; crashed tasks should not be returned by get_by_thread_id/get_by_session_id
        self._by_thread_id.pop(task.thread_id, None)
        if task.current_claude_session_id is not None:
            self._by_session_id.pop(task.current_claude_session_id, None)
        # Clean up typing task (cancel without waiting for graceful stop), aggregator, and TUI handler task
        await self._stop_typing(task.task_id)
        self._aggregators.pop(task.task_id, None)
        handler_task = self._tui_handler_tasks.pop(task.task_id, None)
        if handler_task is not None and not handler_task.done():
            handler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handler_task
        # Clean up the task-scoped settings file
        _cleanup_task_settings(task_id)

    async def restart_task(self, task_id: str) -> Task:
        """Resume a task by spawning a new claude with --resume <session_id>.

        Reuses the existing pane if it's still alive; spawns a new pane otherwise.
        Raises TaskNotFound if task_id doesn't exist.
        Raises TaskRestartError if there's no claude session to resume.
        """
        task = self.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFound(task_id)

        if task.current_claude_session_id is None:
            raise TaskRestartError("no claude session to resume")

        pane_id = task.zellij_pane_id
        pane_alive = pane_id is not None and await self._is_pane_alive(pane_id)

        if pane_alive:
            # Just write the resume command into the existing pane; user sees the new banner.
            await self._zellij.write_to_pane(
                pane_id,
                f"\nclaude --resume {task.current_claude_session_id}\n",
            )
            # Status stays 'running' — SessionStart will rebind on the new session id
            task.last_activity = int(time.time())
            await self._persist(task)
            return task

        # Spawn a fresh pane with the resumed session.
        settings_path = _write_task_settings(task.task_id)
        env = self._build_spawn_env(task.task_id)
        new_pane_id = await self._zellij.spawn_task(
            cwd=task.cwd,
            env=env,
            pane_name=f"cc-{task.task_id[:8]}",
            extra_argv=["--settings", str(settings_path), "--resume", task.current_claude_session_id],
        )
        task.zellij_pane_id = new_pane_id
        task.last_activity = int(time.time())
        await self._index(task)
        await self._persist(task)
        return task
