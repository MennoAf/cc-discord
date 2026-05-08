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
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from bridge import state
from bridge.bot import Bot

logger = logging.getLogger(__name__)


DEFAULT_APPROVAL_TIMEOUT = 600.0  # seconds

TUI_DEFAULT_TIMEOUT = 600.0


_NUMERIC_REACTIONS = {
    "1️⃣": 0,
    "2️⃣": 1,
    "3️⃣": 2,
    "4️⃣": 3,
}

_PLAN_REACTIONS = {
    "✅": ("approve", "approve plan via reaction"),
    "❌": ("reject", "reject plan via reaction"),
    # 💬 has no entry — the user is expected to type their feedback as a thread reply,
    # which `resolve_tui_by_text` already handles. Adding 💬 to the reaction set would
    # introduce dead state (an "awaiting comment" flag with no semantic difference from
    # the default behavior, since text replies always resolve).
}

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


@dataclass
class _PendingTuiAnswer:
    request_id: str
    task_id: str
    thread_id: int
    pane_id: str
    kind: str  # "ask_question" | "exit_plan" | "free_text"
    options: list[str]  # [] for free_text and exit_plan; option labels for ask_question
    future: asyncio.Future  # resolves to (answer_text: str, source: "reaction"|"reply"|"zellij")
    message_id: int | None = None
    created_at: float = field(default_factory=time.time)


class ApprovalRouter:
    def __init__(
        self,
        bot: Bot,
        conn: aiosqlite.Connection,
        *,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT,
        tui_timeout: float = TUI_DEFAULT_TIMEOUT,
    ) -> None:
        self._bot = bot
        self._conn = conn
        self._timeout = timeout
        self._tui_timeout = tui_timeout
        self._by_request_id: dict[str, _PendingApproval] = {}
        self._by_message_id: dict[int, _PendingApproval] = {}
        self._tui_pending: dict[str, _PendingTuiAnswer] = {}
        self._tui_by_message_id: dict[int, _PendingTuiAnswer] = {}
        self._tui_by_thread_id: dict[int, list[_PendingTuiAnswer]] = {}
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
            try:
                await self._bot.add_reactions(primary_msg_id, thread_id, ["✅", "❌"])
            except Exception:
                logger.exception("failed to add approval reactions")
                await self._cleanup(request_id)
                return ("deny", "failed to add approval reactions (check bot permissions)")
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

    async def resolve_by_reaction(self, message_id: int, emoji: str, user_is_self_bot: bool) -> bool:
        """Called by the bot on reaction-add. Returns True if a Future was resolved.

        Filters out reactions added by the bridge's own bot user (user_is_self_bot=True).
        Reactions from other bots in the channel ARE processed.
        """
        if user_is_self_bot:
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
        as deny-with-reason only if the message contains non-whitespace text.

        Empty or whitespace-only messages (e.g., image-only posts) do NOT resolve the approval;
        they fall through to task routing or the listener.

        At most one pending approval per thread at a time is the common case — the design
        implies sequential approvals. When multiple exist, the most recently created one is
        selected.

        Free-text reply order: if the user reacts AND types text, the first to land wins
        (the loser's resolution attempt is dropped because the future is already done).
        """
        if author_is_bot:
            return False
        # Only resolve if the message contains non-whitespace text
        stripped = text.strip()
        if not stripped:
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
        pending.future.set_result(("deny", stripped))
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

    async def request_tui_answer(
        self,
        *,
        request_id: str,
        task_id: str,
        thread_id: int,
        pane_id: str,
        kind: str,
        prompt_body: str,
        options: list[str] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """Post the prompt, await user response, return (answer_text_to_inject, source).

        Returns ("", "cancelled") if the future is cancelled (e.g., user answered in zellij).
        """
        if timeout is None:
            timeout = self._tui_timeout
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        pending = _PendingTuiAnswer(
            request_id=request_id,
            task_id=task_id,
            thread_id=thread_id,
            pane_id=pane_id,
            kind=kind,
            options=options or [],
            future=fut,
        )

        async with self._lock:
            self._tui_pending[request_id] = pending
            self._tui_by_thread_id.setdefault(thread_id, []).append(pending)

        try:
            message_ids = await self._bot.post(prompt_body, thread_id=thread_id)
            if message_ids:
                pending.message_id = message_ids[0]
                async with self._lock:
                    self._tui_by_message_id[pending.message_id] = pending
                if kind == "ask_question" and options:
                    n = min(len(options), 4)
                    emojis = list(_NUMERIC_REACTIONS.keys())[:n]
                    await self._bot.add_reactions(pending.message_id, thread_id, emojis)
                elif kind == "exit_plan":
                    # Two reactions only. Users who want to leave feedback type a thread
                    # reply directly — text replies always resolve via `resolve_tui_by_text`.
                    await self._bot.add_reactions(pending.message_id, thread_id, ["✅", "❌"])
                # free_text: no reactions
        except Exception:
            logger.exception("failed to post tui prompt")
            await self._cleanup_tui(request_id)
            return ("", "post_failed")

        try:
            answer, source = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            answer, source = ("", "timeout")
            try:
                await self._bot.post("⏱ TUI prompt timed out", thread_id=thread_id)
            except Exception:
                logger.exception("failed to post timeout notice")
        except asyncio.CancelledError:
            answer, source = ("", "cancelled")
            try:
                await self._bot.post("(Answered in zellij)", thread_id=thread_id)
            except Exception:
                logger.exception("failed to post cancel notice")
        finally:
            await self._cleanup_tui(request_id)

        return (answer, source)

    async def resolve_tui_by_reaction(self, message_id: int, emoji: str, user_is_bot: bool) -> bool:
        if user_is_bot:
            return False
        async with self._lock:
            pending = self._tui_by_message_id.get(message_id)
            if pending is None:
                return False
        if pending.future.done():
            return False

        if pending.kind == "ask_question":
            if emoji not in _NUMERIC_REACTIONS:
                return False
            idx = _NUMERIC_REACTIONS[emoji]
            if idx >= len(pending.options):
                return False
            # Inject the option's *number* (1-indexed) — Claude's AskUserQuestion TUI accepts
            # the index. This is the simplest interop; the alternative (typing the label text)
            # depends on how the TUI is currently parsing user input.
            pending.future.set_result((str(idx + 1), "reaction"))
            return True

        if pending.kind == "exit_plan":
            if emoji not in _PLAN_REACTIONS:
                return False
            answer, _reason = _PLAN_REACTIONS[emoji]
            if answer == "approve":
                # ExitPlanMode in TUI is typically resolved via choosing "Yes, proceed" — we
                # inject "1" (the first option) per Claude Code's convention. Verify the
                # injection mapping during smoke. If "1" doesn't work, fall back to the
                # literal label "Yes, proceed" via write-chars.
                pending.future.set_result(("1", "reaction"))
                return True
            if answer == "reject":
                pending.future.set_result(("2", "reaction"))
                return True
        return False

    async def resolve_tui_by_text(self, thread_id: int, text: str, author_is_bot: bool) -> bool:
        if author_is_bot:
            return False
        async with self._lock:
            pendings = self._tui_by_thread_id.get(thread_id, [])
            candidates = [p for p in pendings if not p.future.done()]
        if not candidates:
            return False
        # Most recently created wins
        pending = max(candidates, key=lambda p: p.created_at)
        # For exit_plan we always treat text as the comment body.
        # For ask_question we treat text as the typed answer (could be a number or a free-form
        # answer; inject literally — Claude's TUI parses).
        # For free_text we just inject the text.
        pending.future.set_result((text.strip(), "reply"))
        return True

    async def cancel_thread_tui(self, thread_id: int) -> int:
        """Cancel all pending TUI prompts in this thread (used when zellij UserPromptSubmit
        fires while a Discord prompt is still pending). Returns count cancelled."""
        async with self._lock:
            pendings = self._tui_by_thread_id.get(thread_id, [])
            to_cancel = [p for p in pendings if not p.future.done()]
        for p in to_cancel:
            if not p.future.done():
                p.future.cancel()
        return len(to_cancel)

    async def _cleanup_tui(self, request_id: str) -> None:
        async with self._lock:
            pending = self._tui_pending.pop(request_id, None)
            if pending is None:
                return
            if pending.message_id is not None:
                self._tui_by_message_id.pop(pending.message_id, None)
            tlist = self._tui_by_thread_id.get(pending.thread_id, [])
            tlist[:] = [p for p in tlist if p.request_id != request_id]
            if not tlist:
                self._tui_by_thread_id.pop(pending.thread_id, None)

    async def _cleanup(self, request_id: str) -> None:
        async with self._lock:
            pending = self._by_request_id.pop(request_id, None)
            if pending is not None and pending.message_id is not None:
                self._by_message_id.pop(pending.message_id, None)
