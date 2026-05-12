"""Tests for skill/command enumeration and frontmatter parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge import skills


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Path.home() at a tmp directory and pre-create .claude/."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _write_skill(home: Path, name: str, description: str) -> None:
    d = home / ".claude" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody\n")


def _write_command(home: Path, name: str, body: str) -> None:
    d = home / ".claude" / "commands"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(body)


# Enumeration


def test_list_commands_from_commands_dir(fake_home: Path) -> None:
    _write_command(fake_home, "finch", "---\nname: finch\ndescription: orchestrator\n---\n")
    _write_command(fake_home, "grip", "---\nname: grip\ndescription: convergence\n---\n")
    names = [s.name for s in skills.list_skills()]
    assert "finch" in names
    assert "grip" in names


def test_lists_skill_alongside_commands(fake_home: Path) -> None:
    _write_command(fake_home, "finch", "---\nname: finch\ndescription: orchestrator\n---\n")
    _write_skill(fake_home, "ask-discord", "ask via discord")
    names = [s.name for s in skills.list_skills()]
    assert names == sorted(names)
    assert {"finch", "ask-discord"}.issubset(set(names))


def test_skill_wins_over_command_on_name_collision(fake_home: Path) -> None:
    _write_command(fake_home, "finch", "---\nname: finch\ndescription: from-command\n---\n")
    _write_skill(fake_home, "finch", "from-skill")
    out = {s.name: s for s in skills.list_skills()}
    assert out["finch"].source == "user"
    assert out["finch"].description == "from-skill"


def test_dotfile_commands_ignored(fake_home: Path) -> None:
    _write_command(fake_home, "finch", "---\nname: finch\ndescription: ok\n---\n")
    (fake_home / ".claude" / "commands" / ".DS_Store").write_text("noise")
    (fake_home / ".claude" / "commands" / ".hidden.md").write_text("---\nname: hidden\ndescription: x\n---\n")
    names = [s.name for s in skills.list_skills()]
    assert ".hidden" not in names
    assert ".DS_Store" not in names
    assert "finch" in names


def test_subdirectories_in_commands_ignored(fake_home: Path) -> None:
    _write_command(fake_home, "finch", "---\nname: finch\ndescription: ok\n---\n")
    (fake_home / ".claude" / "commands" / "references").mkdir()
    (fake_home / ".claude" / "commands" / "references" / "thing.md").write_text("nope")
    names = [s.name for s in skills.list_skills()]
    assert "finch" in names
    assert "references" not in names
    assert "thing" not in names


def test_missing_commands_dir_is_fine(fake_home: Path) -> None:
    _write_skill(fake_home, "ask-discord", "ask via discord")
    names = [s.name for s in skills.list_skills()]
    assert names == ["ask-discord"]


# Frontmatter parsing


def test_inline_description(fake_home: Path) -> None:
    _write_command(fake_home, "x", "---\nname: x\ndescription: simple inline\n---\n")
    s = next(s for s in skills.list_skills() if s.name == "x")
    assert s.description == "simple inline"


def test_quoted_inline_description(fake_home: Path) -> None:
    _write_command(fake_home, "x", '---\nname: x\ndescription: "quoted text"\n---\n')
    s = next(s for s in skills.list_skills() if s.name == "x")
    assert s.description == "quoted text"


def test_folded_block_scalar(fake_home: Path) -> None:
    body = (
        "---\n"
        "name: finch\n"
        "description: >\n"
        "  Framework orchestrator.\n"
        "  Use when stuck.\n"
        "---\n"
    )
    _write_command(fake_home, "finch", body)
    s = next(s for s in skills.list_skills() if s.name == "finch")
    assert s.description == "Framework orchestrator. Use when stuck."


def test_literal_block_scalar(fake_home: Path) -> None:
    body = (
        "---\n"
        "name: x\n"
        "description: |\n"
        "  line one\n"
        "  line two\n"
        "---\n"
    )
    _write_command(fake_home, "x", body)
    s = next(s for s in skills.list_skills() if s.name == "x")
    assert s.description == "line one\nline two"


def test_block_scalar_terminated_by_non_indent(fake_home: Path) -> None:
    body = (
        "---\n"
        "name: x\n"
        "description: >\n"
        "  first line\n"
        "  second line\n"
        "name_after: stops_here\n"
        "---\n"
    )
    _write_command(fake_home, "x", body)
    s = next(s for s in skills.list_skills() if s.name == "x")
    assert s.description == "first line second line"


def test_no_frontmatter_returns_none_description(fake_home: Path) -> None:
    _write_command(fake_home, "x", "just body, no fences\n")
    s = next(s for s in skills.list_skills() if s.name == "x")
    assert s.description is None
