"""Pure formatter: convert PostToolUse payloads into one-line Discord messages."""

from typing import Any

# Emoji map. Pick distinct glyphs for at-a-glance scanning.
_EMOJI_OK = "✓"
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
