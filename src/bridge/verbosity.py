"""Per-channel verbosity policy (the `/tone` feature).

Centralizes which event kinds reach Discord under each mode so call sites
in `tasks.py` can branch on a single boolean instead of duplicating the
mode matrix. Keep this module dependency-free — it must be importable
from anywhere without circular imports.

The matrix:

                          full   light   tldr
    show_prose             ✓      ✓
    show_thinking          ✓
    show_tool_lines        ✓
    show_tool_diffs        ✓
    show_task_list         ✓      ✓
    show_rolling_indicator        ✓       ✓
    show_commit_milestones                ✓

Events not listed above are unconditional: AskUserQuestion, approval
prompts, errors, and final session-end messages always reach Discord —
filtering process narration must never silence a user-actionable signal.
"""

from dataclasses import dataclass

from bridge.state import DEFAULT_VERBOSITY, VALID_VERBOSITY_MODES


@dataclass(frozen=True)
class VerbosityPolicy:
    """Boolean matrix derived from a /tone mode.

    Call sites query one field; this dataclass owns the mapping.
    """

    mode: str
    show_prose: bool
    show_thinking: bool
    show_tool_lines: bool
    show_tool_diffs: bool
    show_task_list: bool
    show_rolling_indicator: bool
    show_commit_milestones: bool


_FULL = VerbosityPolicy(
    mode="full",
    show_prose=True,
    show_thinking=True,
    show_tool_lines=True,
    show_tool_diffs=True,
    show_task_list=True,
    show_rolling_indicator=False,
    show_commit_milestones=False,
)

_LIGHT = VerbosityPolicy(
    mode="light",
    show_prose=True,
    show_thinking=False,
    show_tool_lines=False,
    show_tool_diffs=False,
    show_task_list=True,
    show_rolling_indicator=True,
    show_commit_milestones=False,
)

_TLDR = VerbosityPolicy(
    mode="tldr",
    show_prose=False,
    show_thinking=False,
    show_tool_lines=False,
    show_tool_diffs=False,
    show_task_list=False,
    # Rolling indicator on: a faint "🔧 Working…" heartbeat so a near-silent
    # session never looks dead. Prose/tools/diffs stay suppressed; this is the
    # only proof-of-life between commit milestones and user-actionable prompts.
    show_rolling_indicator=True,
    show_commit_milestones=True,
)

_POLICIES: dict[str, VerbosityPolicy] = {
    "full": _FULL,
    "light": _LIGHT,
    "tldr": _TLDR,
}

# Public constant re-exports so callers don't need to dual-import.
VALID_MODES = VALID_VERBOSITY_MODES
DEFAULT_MODE = DEFAULT_VERBOSITY


def policy_for(mode: str | None) -> VerbosityPolicy:
    """Return the policy for `mode`. Unknown / None modes return the default.

    Unknown modes don't raise — older DB rows or future modes degrade
    gracefully to the default (full = current behavior). Validation
    happens at write time in `state.upsert_verbosity`.
    """
    if mode is None:
        return _POLICIES[DEFAULT_VERBOSITY]
    return _POLICIES.get(mode, _POLICIES[DEFAULT_VERBOSITY])
