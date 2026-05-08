"""Task and TaskRegistry for managing discord-driven sessions."""

import logging
from dataclasses import dataclass, field

import aiosqlite

from bridge.state import TaskRow, get_task_by_session_id, list_active_tasks, upsert_task

logger = logging.getLogger(__name__)


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

    def __init__(self, conn: aiosqlite.Connection, bot) -> None:
        """Initialize with database connection and bot."""
        self._conn = conn
        self._bot = bot
        self._by_task_id: dict[str, Task] = {}
        self._by_thread_id: dict[int, Task] = {}
        self._by_session_id: dict[str, Task] = {}
        self._HANDLERS: dict[str, str] = {
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

    async def _index(self, task: Task) -> None:
        """Update all three maps for a task. Removes prior session_id entry if it changed."""
        # Remove old session_id entry if this session_id was previously mapped
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
        """Handle SessionStart event."""
        session_id = body.get("session_id")
        cwd = body.get("cwd")
        transcript_path = body.get("transcript_path")
        env_passthrough = body.get("env_passthrough", {})
        task_id = env_passthrough.get("CC_DISCORD_TASK_ID")

        logger.info(f"SessionStart: session_id={session_id}, task_id={task_id}")

        if task_id:
            task = self.get_by_task_id(task_id)
            if task:
                task.current_claude_session_id = session_id
                task.current_transcript_path = transcript_path
                await self._index(task)
                await self._persist(task)
                await self._bot.post(
                    f"🟢 SessionStart received (task={task_id[:8]})",
                    thread_id=task.thread_id,
                )

    async def _on_user_prompt_submit(self, body: dict) -> None:
        """Handle UserPromptSubmit event."""
        session_id = body.get("session_id")
        logger.info(f"UserPromptSubmit: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "💬 UserPromptSubmit received",
                thread_id=task.thread_id,
            )

    async def _on_post_tool_use(self, body: dict) -> None:
        """Handle PostToolUse event."""
        session_id = body.get("session_id")
        logger.info(f"PostToolUse: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "🔧 PostToolUse received",
                thread_id=task.thread_id,
            )

    async def _on_post_tool_use_failure(self, body: dict) -> None:
        """Handle PostToolUseFailure event."""
        session_id = body.get("session_id")
        logger.info(f"PostToolUseFailure: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "❌ PostToolUseFailure received",
                thread_id=task.thread_id,
            )

    async def _on_stop(self, body: dict) -> None:
        """Handle Stop event."""
        session_id = body.get("session_id")
        logger.info(f"Stop: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "⏹️ Stop received",
                thread_id=task.thread_id,
            )

    async def _on_notification(self, body: dict) -> None:
        """Handle Notification event."""
        session_id = body.get("session_id")
        logger.info(f"Notification: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "🔔 Notification received",
                thread_id=task.thread_id,
            )

    async def _on_session_end(self, body: dict) -> None:
        """Handle SessionEnd event."""
        session_id = body.get("session_id")
        logger.info(f"SessionEnd: session_id={session_id}")
        task = await get_task_by_session_id(self._conn, session_id) if session_id else None
        if task and self.get_by_task_id(task.task_id):
            await self._bot.post(
                "🏁 SessionEnd received",
                thread_id=task.thread_id,
            )

    async def _on_subagent_stop(self, body: dict) -> None:
        """Handle SubagentStop event (no-op in Phase 1)."""
        logger.debug(f"SubagentStop received")

    async def _on_pre_compact(self, body: dict) -> None:
        """Handle PreCompact event (no-op in Phase 1)."""
        logger.debug(f"PreCompact received")
