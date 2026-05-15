"""Tests for projects.py — BRIDGE_PROJECT_ROOTS parsing and enumeration."""

from __future__ import annotations

from pathlib import Path

import pytest

from bridge.projects import (
    Project,
    enumerate_projects,
    load_projects_from_env,
    parse_project_roots,
)


class TestParseProjectRoots:
    def test_none_returns_empty(self) -> None:
        assert parse_project_roots(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert parse_project_roots("") == []

    def test_single_existing_dir(self, tmp_path: Path) -> None:
        assert parse_project_roots(str(tmp_path)) == [tmp_path.resolve()]

    def test_multiple_dirs_colon_separated(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        result = parse_project_roots(f"{a}:{b}")
        assert result == [a.resolve(), b.resolve()]

    def test_skips_nonexistent(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        existing = tmp_path / "ok"
        existing.mkdir()
        missing = tmp_path / "nope"
        result = parse_project_roots(f"{existing}:{missing}")
        assert result == [existing.resolve()]
        assert "not an existing directory" in caplog.text

    def test_skips_files(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("nope")
        assert parse_project_roots(str(f)) == []

    def test_trailing_colon_ignored(self, tmp_path: Path) -> None:
        assert parse_project_roots(f"{tmp_path}:") == [tmp_path.resolve()]

    def test_leading_colon_ignored(self, tmp_path: Path) -> None:
        assert parse_project_roots(f":{tmp_path}") == [tmp_path.resolve()]

    def test_whitespace_around_entries(self, tmp_path: Path) -> None:
        assert parse_project_roots(f"  {tmp_path}  ") == [tmp_path.resolve()]

    def test_dedupes_repeated_path(self, tmp_path: Path) -> None:
        assert parse_project_roots(f"{tmp_path}:{tmp_path}") == [tmp_path.resolve()]

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        sub = tmp_path / "sub"
        sub.mkdir()
        assert parse_project_roots("~/sub") == [sub.resolve()]


class TestEnumerateProjects:
    def test_empty_roots_returns_empty(self) -> None:
        assert enumerate_projects([]) == []

    def test_lists_immediate_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        roots = [tmp_path]
        result = enumerate_projects(roots)
        names = [p.name for p in result]
        assert names == ["alpha", "beta"]

    def test_skips_files(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "notes.txt").write_text("ignore me")
        result = enumerate_projects([tmp_path])
        assert [p.name for p in result] == ["alpha"]

    def test_skips_dotfiles(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".git").mkdir()
        result = enumerate_projects([tmp_path])
        assert [p.name for p in result] == ["alpha"]

    def test_does_not_recurse(self, tmp_path: Path) -> None:
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / "inner").mkdir()
        result = enumerate_projects([tmp_path])
        assert [p.name for p in result] == ["outer"]
        assert all(p.name != "inner" for p in result)

    def test_root_label_is_parent_folder_basename(self, tmp_path: Path) -> None:
        work = tmp_path / "Work"
        work.mkdir()
        (work / "proj_a").mkdir()
        result = enumerate_projects([work])
        assert result[0].root_label == "Work"
        assert result[0].name == "proj_a"
        assert result[0].path == (work / "proj_a").resolve()

    def test_sort_is_case_insensitive_and_deterministic(self, tmp_path: Path) -> None:
        (tmp_path / "Zeta").mkdir()
        (tmp_path / "alpha").mkdir()
        (tmp_path / "Mango").mkdir()
        result = enumerate_projects([tmp_path])
        assert [p.name for p in result] == ["alpha", "Mango", "Zeta"]

    def test_multiple_roots_sorted_by_root_label_then_name(self, tmp_path: Path) -> None:
        work = tmp_path / "Work"
        personal = tmp_path / "Personal"
        work.mkdir()
        personal.mkdir()
        (work / "wproj").mkdir()
        (personal / "pproj").mkdir()
        # Pass them in non-sorted order on purpose.
        result = enumerate_projects([work, personal])
        labels_and_names = [(p.root_label, p.name) for p in result]
        assert labels_and_names == [("Personal", "pproj"), ("Work", "wproj")]

    def test_dedupes_by_absolute_path(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        # Symlink b/shared → a/shared so the same absolute path appears twice.
        (root_a / "shared").mkdir()
        (root_b / "shared").symlink_to(root_a / "shared")
        result = enumerate_projects([root_a, root_b])
        assert len(result) == 1
        assert result[0].path == (root_a / "shared").resolve()

    def test_unreadable_root_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a permission error on iterdir by pointing at a path that
        # exists per the input contract but fails on read.
        root = tmp_path / "broken"
        root.mkdir()

        original_iterdir = Path.iterdir

        def fake_iterdir(self: Path):  # noqa: ANN202
            if self == root:
                raise PermissionError("denied")
            return original_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)
        result = enumerate_projects([root])
        assert result == []
        assert "cannot read" in caplog.text


class TestLoadProjectsFromEnv:
    def test_unset_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRIDGE_PROJECT_ROOTS", raising=False)
        assert load_projects_from_env() == []

    def test_reads_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "proj").mkdir()
        monkeypatch.setenv("BRIDGE_PROJECT_ROOTS", str(tmp_path))
        result = load_projects_from_env()
        assert [(p.name, p.root_label) for p in result] == [
            ("proj", tmp_path.name)
        ]


class TestProjectDataclass:
    def test_is_hashable_and_frozen(self, tmp_path: Path) -> None:
        p = Project(path=tmp_path, name="a", root_label="Work")
        # frozen → assignment raises
        with pytest.raises(Exception):
            p.name = "b"  # type: ignore[misc]
        # hashable → can put in a set
        assert {p} == {p}
