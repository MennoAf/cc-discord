"""Tests for the /auto project-name resolver helpers in bridge.commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge import commands


@pytest.fixture
def projects_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a sample projects.json to tmp and point PROJECTS_PATH at it."""
    p = tmp_path / "projects.json"
    p.write_text(json.dumps({
        "_comment": "ignored",
        "boiler": "/abs/boiler_room",
        "Cleanroom": "/abs/AIO_cleanroom",
        "wick": "/abs/Wick",
        "_other_skip": "/abs/skip",
        "bad_value": 42,
    }))
    monkeypatch.setattr(commands, "PROJECTS_PATH", p)
    return p


def test_load_projects_skips_underscore_and_non_string(projects_file: Path) -> None:
    out = commands._load_projects()
    assert "boiler" in out
    assert "cleanroom" in out  # lowercased
    assert "wick" in out
    assert "_comment" not in out
    assert "_other_skip" not in out
    assert "bad_value" not in out


def test_load_projects_lowercases_keys(projects_file: Path) -> None:
    out = commands._load_projects()
    assert out["cleanroom"] == "/abs/AIO_cleanroom"


def test_load_projects_missing_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, "PROJECTS_PATH", tmp_path / "nope.json")
    assert commands._load_projects() == {}


def test_load_projects_invalid_json_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    monkeypatch.setattr(commands, "PROJECTS_PATH", p)
    assert commands._load_projects() == {}


def test_resolve_project_case_insensitive(projects_file: Path) -> None:
    assert commands._resolve_project("BOILER") == "/abs/boiler_room"
    assert commands._resolve_project("cleanroom") == "/abs/AIO_cleanroom"
    assert commands._resolve_project("Wick") == "/abs/Wick"


def test_resolve_project_unknown_returns_none(projects_file: Path) -> None:
    assert commands._resolve_project("nonexistent") is None
