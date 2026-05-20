"""Project roots: enumerate spawnable Claude project directories for /spawn.

`BRIDGE_PROJECT_ROOTS` is a colon-separated list of parent folders. Each parent
contributes its immediate subdirectories as spawnable projects (one level deep).
The list is enumerated once at server startup and cached in memory.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Project:
    """A spawnable project: an immediate subfolder of a configured root."""

    path: Path
    name: str
    root_label: str


def parse_project_roots(raw: str | None) -> list[Path]:
    """Parse BRIDGE_PROJECT_ROOTS into a list of validated absolute Paths.

    Splits on `:`, expands `~`, resolves to absolute, drops entries that aren't
    existing directories (logged as warnings). Returns an empty list when raw
    is None or empty so callers don't have to special-case unconfigured.
    """
    if not raw:
        return []
    roots: list[Path] = []
    seen: set[Path] = set()
    for entry in raw.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        path = Path(entry).expanduser()
        try:
            path = path.resolve()
        except OSError as e:
            logger.warning("BRIDGE_PROJECT_ROOTS: cannot resolve %r (%s)", entry, e)
            continue
        if not path.is_dir():
            logger.warning(
                "BRIDGE_PROJECT_ROOTS: %s is not an existing directory, skipping",
                path,
            )
            continue
        if path in seen:
            continue
        seen.add(path)
        roots.append(path)
    return roots


def enumerate_projects(roots: list[Path]) -> list[Project]:
    """Scan immediate subfolders of each root, return a sorted Project list.

    Dotfile-prefixed entries are skipped. Duplicate absolute paths are
    deduplicated (first-root-wins). Sort key is (root_label, name) for
    deterministic autocomplete ordering.
    """
    projects: list[Project] = []
    seen_paths: set[Path] = set()
    for root in roots:
        root_label = root.name or str(root)
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError as e:
            logger.warning("BRIDGE_PROJECT_ROOTS: cannot read %s (%s)", root, e)
            continue
        for child in entries:
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            resolved = child.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            projects.append(
                Project(path=resolved, name=child.name, root_label=root_label)
            )
    projects.sort(key=lambda p: (p.root_label.lower(), p.name.lower()))
    return projects


def load_projects_from_env() -> list[Project]:
    """Convenience: read BRIDGE_PROJECT_ROOTS and enumerate in one call."""
    roots = parse_project_roots(os.environ.get("BRIDGE_PROJECT_ROOTS"))
    projects = enumerate_projects(roots)
    logger.info(
        "Loaded %d project(s) from %d root(s)", len(projects), len(roots)
    )
    return projects
