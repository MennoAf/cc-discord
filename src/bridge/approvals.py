"""ApprovalRouter — bridges PreToolUse hook calls to Discord reaction round-trips.

Holds an in-memory dict[request_id -> Future]. The HTTP handler in server.py registers
a Future, posts the approval prompt in Discord, and awaits the Future with a 600s timeout.
The bot's reaction handler resolves Futures by request_id (looked up via the message_id
the bot just posted).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from bridge import state
from bridge.bot import Bot

logger = logging.getLogger(__name__)


DEFAULT_APPROVAL_TIMEOUT = 600.0  # seconds

# Reaction emoji -> decision
_REACTION_DECISIONS = {
    "✅": ("allow", "approved via reaction"),
    "❌": ("deny", "denied via reaction"),
}


@dataclass
class _PendingApproval:
    request_id: str
    task_id: str
    tool_name: str
    tool_input: dict[str, Any]
    thread_id: int
    created_at: int
    future: asyncio.Future  # resolves to (decision: str, reason: str)
    message_id: int | None = None  # Discord message id; set after we post the prompt


class ApprovalRouter:
    def __init__(
        self,
        bot: Bot,
        conn: aiosqlite.Connection,
        *,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> None:
        self._bot = bot
        self._conn = conn
        self._timeout = timeout
        self._by_request_id: dict[str, _PendingApproval] = {}
        self._by_message_id: dict[int, _PendingApproval] = {}
        self._lock = asyncio.Lock()

    async def request_permission(
        self,
        *,
        request_id: str,
        task_id: str,
        thread_id: int,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> tuple[str, str]:
        """Post the approval prompt to Discord, wait for resolution.

        Returns (decision, reason). Always returns — never raises (timeout → ('deny', '...')).
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        created_at = int(time.time())
        pending = _PendingApproval(
            request_id=request_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_input=tool_input,
            thread_id=thread_id,
            created_at=created_at,
            future=fut,
        )
        async with self._lock:
            self._by_request_id[request_id] = pending

        body = self._format_prompt(tool_name, tool_input)
        try:
            message_ids = await self._bot.post(body, thread_id=thread_id)
            if not message_ids:
                # Empty response - fail closed
                await self._cleanup(request_id)
                return ("deny", "could not post approval prompt to Discord")
            primary_msg_id = message_ids[0]
            pending.message_id = primary_msg_id
            async with self._lock:
                self._by_message_id[primary_msg_id] = pending
            # Add reactions to the FIRST chunk only (the one users react on).
            await self._bot.add_reactions(primary_msg_id, thread_id, ["✅", "❌"])
        except Exception:
            logger.exception("failed to post approval prompt")
            await self._cleanup(request_id)
            return ("deny", "failed to post approval prompt")

        try:
            decision, reason = await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            decision, reason = ("deny", "approval timed out")
            try:
                await self._bot.post("🛡 ❌ Denied (timeout)", thread_id=thread_id)
            except Exception:
                logger.exception("failed to post timeout notice")
        finally:
            await self._cleanup(request_id)

        # Persist to approval_log
        try:
            await state.log_approval(
                self._conn,
                request_id=request_id,
                task_id=task_id,
                tool_name=tool_name,
                tool_input_json=json.dumps(tool_input, default=str),
                decision=decision,
                decision_reason=reason,
            )
        except Exception:
            logger.exception("failed to log approval decision")

        return (decision, reason)

    async def resolve_by_reaction(self, message_id: int, emoji: str, user_is_bot: bool) -> bool:
        """Called by the bot on reaction-add. Returns True if a Future was resolved."""
        if user_is_bot:
            return False
        if emoji not in _REACTION_DECISIONS:
            return False
        async with self._lock:
            pending = self._by_message_id.get(message_id)
            if pending is None:
                return False
        if pending.future.done():
            return False
        decision, reason = _REACTION_DECISIONS[emoji]
        pending.future.set_result((decision, reason))
        return True

    async def resolve_by_text(self, thread_id: int, text: str, author_is_bot: bool) -> bool:
        """Called by the bot on a thread message. Resolves any pending approval for this thread
        as deny-with-reason.

        At most one pending approval per thread at a time is the common case — the design
        implies sequential approvals. When multiple exist, the most recently created one is
        selected.
        """
        if author_is_bot:
            return False
        async with self._lock:
            # Find all pending approvals for this thread
            candidates = [
                p for p in self._by_request_id.values()
                if p.thread_id == thread_id and not p.future.done()
            ]
        if not candidates:
            return False
        # Most-recent-first: select by created_at
        pending = max(candidates, key=lambda p: p.created_at)
        pending.future.set_result(("deny", text.strip() or "denied with empty reply"))
        return True

    def _format_prompt(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """🛡 Approve `<tool>`? followed by compact tool input dump."""
        body = f"🛡 Approve `{tool_name}`?\n```json\n"
        # Truncate huge tool_input
        s = json.dumps(tool_input, indent=2, default=str)
        if len(s) > 1500:
            s = s[:1500] + "\n  ...(truncated)..."
        body += s + "\n```"
        return body

    async def _cleanup(self, request_id: str) -> None:
        async with self._lock:
            pending = self._by_request_id.pop(request_id, None)
            if pending is not None and pending.message_id is not None:
                self._by_message_id.pop(pending.message_id, None)
