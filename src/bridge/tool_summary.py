"""Pure formatter: convert PostToolUse payloads into one-line Discord messages."""

from typing import Any

# Emoji map. Pick distinct glyphs for at-a-glance scanning.
_EMOJI_FAIL = "✗"
_EMOJI_EDIT = "✏"
_EMOJI_WRITE = "📝"
_EMOJI_READ = "📖"
_EMOJI_BASH = "✓"
_EMOJI_SEARCH = "🔍"
_EMOJI_WEB = "🌐"
_EMOJI_TASK = "🤖"
_EMOJI_OTHER = "•"


def is_failure(tool_response: dict[str, Any] | None, tool_name: str) -> bool:
    """Detect failure across common tool_response shapes."""
    if not tool_response:
        return False
    if tool_response.get("is_error") is True:
        return True
    if tool_name == "Bash":
        if tool_response.get("interrupted") is True:
            return True
        exit_code = tool_response.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
    err = tool_response.get("error")
    if isinstance(err, str) and err.strip():
        return True
    return False


def summarize(tool_name: str, tool_input: dict, tool_response: dict | None) -> str:
    """Return a one-line summary string. Length-bounded ~150 chars."""
    failed = is_failure(tool_response, tool_name)

    if tool_name == "Bash":
        cmd = (tool_input.get("command") or "").strip()
        cmd = _truncate(cmd, 80)
        if failed:
            ec = (tool_response or {}).get("exit_code", "?")
            return f"{_EMOJI_FAIL} Bash: {cmd} (exit {ec})"
        return f"{_EMOJI_BASH} Bash: {cmd}"

    if tool_name == "Edit":
        path = _short_path(tool_input.get("file_path", "?"))
        diff_summary = _diff_summary(tool_input)
        emoji = _EMOJI_FAIL if failed else _EMOJI_EDIT
        return f"{emoji} Edit: {path} {diff_summary}".strip()

    if tool_name == "Write":
        path = _short_path(tool_input.get("file_path", "?"))
        emoji = _EMOJI_FAIL if failed else _EMOJI_WRITE
        n_chars = len((tool_input.get("content") or ""))
        return f"{emoji} Write: {path} ({n_chars} chars)"

    if tool_name == "Read":
        path = _short_path(tool_input.get("file_path", "?"))
        emoji = _EMOJI_FAIL if failed else _EMOJI_READ
        return f"{emoji} Read: {path}"

    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "?")
        return f"{_EMOJI_SEARCH if not failed else _EMOJI_FAIL} Glob: {pattern}"

    if tool_name == "Grep":
        pat = tool_input.get("pattern", "?")
        path = _short_path(tool_input.get("path", "")) if tool_input.get("path") else ""
        return f"{_EMOJI_SEARCH if not failed else _EMOJI_FAIL} Grep: {pat}{(' in ' + path) if path else ''}"

    if tool_name in ("WebFetch", "WebSearch"):
        target = tool_input.get("url") or tool_input.get("query") or "?"
        emoji = _EMOJI_FAIL if failed else _EMOJI_WEB
        return f"{emoji} {tool_name}: {_truncate(target, 80)}"

    if tool_name == "Task":
        desc = tool_input.get("description") or "(subagent)"
        emoji = _EMOJI_FAIL if failed else _EMOJI_TASK
        return f"{emoji} Task: {_truncate(desc, 80)}"

    if tool_name == "Skill":
        # Skill tool input includes `skill` (skill name) and optional `args`.
        name = tool_input.get("skill") or "?"
        emoji = _EMOJI_FAIL if failed else _EMOJI_OTHER
        return f"{emoji} Skill: {name}"

    if tool_name == "TodoWrite":
        todos = tool_input.get("todos") or []
        emoji = _EMOJI_FAIL if failed else "📋"
        if not isinstance(todos, list):
            return f"{emoji} TodoWrite"
        done = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
        return f"{emoji} TodoWrite: {done}/{len(todos)} done"

    # Generic fallback
    emoji = _EMOJI_FAIL if failed else _EMOJI_OTHER
    return f"{emoji} {tool_name}"


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_path(p: str | None) -> str:
    if not p:
        return "?"
    if len(p) <= 50:
        return p
    parts = p.split("/")
    if len(parts) > 3:
        return ".../" + "/".join(parts[-2:])
    return _truncate(p, 50)


def _diff_summary(tool_input: dict) -> str:
    """For Edit: return '+N -M' line counts for the change."""
    old = tool_input.get("old_string") or ""
    new = tool_input.get("new_string") or ""
    plus = len(new.splitlines())
    minus = len(old.splitlines())
    return f"+{plus} -{minus}"


# Discord's hard limit is 2000 chars per message; reserve headroom for the
# fence + truncation marker.
_DISCORD_LIMIT = 2000
_DIFF_BUDGET = _DISCORD_LIMIT - 80


def diff_block(tool_name: str, tool_input: dict) -> str | None:
    """Return a fenced Discord-renderable diff/content block for Edit / MultiEdit
    / Write / TodoWrite, or None for tools that don't have a block to emit.

    Truncates at ~_DISCORD_LIMIT to fit a single Discord message.
    """
    if tool_name == "Edit":
        return _format_edit_diff(
            tool_input.get("file_path") or "?",
            tool_input.get("old_string") or "",
            tool_input.get("new_string") or "",
        )
    if tool_name == "MultiEdit":
        path = tool_input.get("file_path") or "?"
        edits = tool_input.get("edits") or []
        chunks: list[str] = []
        for ed in edits:
            if not isinstance(ed, dict):
                continue
            chunks.append(
                _diff_body(ed.get("old_string") or "", ed.get("new_string") or "")
            )
        if not chunks:
            return None
        return _wrap_diff(f"--- {path}\n+++ {path}\n" + "\n".join(chunks))
    if tool_name == "Write":
        path = tool_input.get("file_path") or "?"
        content = tool_input.get("content") or ""
        return _wrap_code(content, path)
    if tool_name == "TodoWrite":
        todos = tool_input.get("todos") or []
        if not isinstance(todos, list) or not todos:
            return None
        return _format_todo_checklist(todos)
    return None


def _format_todo_checklist(todos: list) -> str:
    """Render a Claude TodoWrite payload as a Discord-friendly checklist.

    Each todo has `content`, `status` (pending|in_progress|completed), and
    `activeForm`. Use ▶ for in_progress, [x] for completed, [ ] otherwise.
    """
    lines = ["**Todos:**"]
    for t in todos:
        if not isinstance(t, dict):
            continue
        status = t.get("status") or ""
        content = t.get("content") or ""
        active = t.get("activeForm") or ""
        if status == "completed":
            mark = "✅"
        elif status == "in_progress":
            mark = "▶️"
            content = active or content
        else:
            mark = "⬜"
        lines.append(f"{mark} {content}")
    body = "\n".join(lines)
    if len(body) > _DISCORD_LIMIT - 50:
        body = body[: _DISCORD_LIMIT - 50] + "\n…"
    return body


def _format_edit_diff(path: str, old: str, new: str) -> str:
    body = f"--- {path}\n+++ {path}\n" + _diff_body(old, new)
    return _wrap_diff(body)


def _diff_body(old: str, new: str) -> str:
    return "\n".join(
        [f"-{line}" for line in old.splitlines()]
        + [f"+{line}" for line in new.splitlines()]
    )


def _wrap_diff(body: str) -> str:
    if len(body) > _DIFF_BUDGET:
        body = body[:_DIFF_BUDGET] + "\n…"
    return f"```diff\n{body}\n```"


def _wrap_code(content: str, path: str) -> str:
    header = f"_(wrote: {path})_\n"
    budget = _DIFF_BUDGET - len(header)
    if len(content) > budget:
        content = content[:budget] + "\n…"
    return f"{header}```\n{content}\n```"
