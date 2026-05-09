"""Read and parse Claude Code transcript JSONL."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Iterator


# Cap on entries we'll yield from a single transcript read. Real claude
# sessions stay well under this (a long, dense session is on the order of
# 1-2k entries). The cap exists to defend the daemon against a hostile or
# buggy hook payload pointing `transcript_path` at /dev/zero or a giant
# unrelated file: without it, every consumer that does `list(read_entries
# (path))` would slurp the whole thing into memory.
_MAX_ENTRIES = 50_000


def read_entries(path: Path) -> Iterator[dict]:
    """Yield each JSON entry from a transcript JSONL.

    Skips malformed lines silently. Reads with explicit utf-8 encoding so
    a misconfigured `LANG=POSIX` doesn't break parsing. Yields at most
    `_MAX_ENTRIES` entries — the LAST `_MAX_ENTRIES`, since callers
    almost universally walk newest-last and care about recent state.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            recent: deque[dict] = deque(maxlen=_MAX_ENTRIES)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            yield from recent
    except (IOError, OSError):
        return


def is_recent_tool_use_sidechain(path: Path, tool_name: str) -> bool:
    """Whether the most recent `tool_use` block of `tool_name` came from a
    sidechain (i.e. a subagent dispatched via the Task tool).

    Walks the transcript newest-first and stops at the first matching
    `tool_use` block, returning the parent assistant entry's `isSidechain`
    flag. Returns False if the tool isn't found or the file isn't readable.
    """
    entries = list(read_entries(path))
    for e in reversed(entries):
        if e.get("type") != "assistant":
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == tool_name
            ):
                return e.get("isSidechain") is True
    return False


def find_latest_unresolved_tool_use(path: Path) -> dict | None:
    """Return the latest tool_use block from the most recent assistant entry whose id has
    no matching tool_result block in subsequent user entries. Returns None if no such block.

    Returned dict shape: {"id": str, "name": str, "input": dict}.
    """
    entries = list(read_entries(path))

    # Collect tool_use ids that have been resolved.
    resolved: set[str] = set()
    for e in entries:
        if e.get("type") != "user":
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tu_id = block.get("tool_use_id")
                if isinstance(tu_id, str):
                    resolved.add(tu_id)

    # Walk assistant entries newest-first; return the first unresolved tool_use we find.
    for e in reversed(entries):
        if e.get("type") != "assistant":
            continue
        if e.get("isSidechain") is True or e.get("isMeta") is True:
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Within one assistant entry, walk tool_use blocks newest-last (last in array).
        # The latest unresolved one is the prompt.
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            tid = block.get("id")
            if not isinstance(tid, str):
                continue
            if tid in resolved:
                continue
            return {
                "id": tid,
                "name": block.get("name", ""),
                "input": block.get("input") or {},
            }

    return None


def extract_final_assistant_text(path: Path) -> str:
    """Return the concatenated text of all assistant entries since the last user prompt.

    Walks from EOF backwards: stops at the first non-sidechain, non-meta `user` entry whose
    message.content is a string (a real user prompt — not a tool_result). Collects every
    `assistant` entry between that point and EOF, ignoring `isSidechain`/`isMeta`. From each
    assistant entry, joins all `text`-type content blocks in order. Tool_use blocks are skipped.

    Returns "" if there are no assistant text blocks since the last user prompt.
    """
    entries = list(read_entries(path))
    if not entries:
        return ""

    # Find the index of the last real user prompt.
    last_user_idx = -1
    for i in range(len(entries) - 1, -1, -1):
        e = entries[i]
        if e.get("type") != "user":
            continue
        if e.get("isSidechain") is True:
            continue
        if e.get("isMeta") is True:
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            last_user_idx = i
            break
        # If content is a list of tool_results, this is not a user prompt — skip.

    # Concatenate assistant text blocks from last_user_idx+1 to end.
    parts: list[str] = []
    for e in entries[last_user_idx + 1 :]:
        if e.get("type") != "assistant":
            continue
        if e.get("isSidechain") is True:
            continue
        if e.get("isMeta") is True:
            continue
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)

    return "\n".join(parts).strip()
