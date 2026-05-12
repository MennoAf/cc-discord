"""Skill enumeration for the `/skill` Discord slash command.

Walks three sources, sorted by name, deduped (skills win over commands on collision):
  1. User-level skills at `~/.claude/skills/<name>/SKILL.md`
  2. Legacy slash commands at `~/.claude/commands/<name>.md`
  3. Enabled-plugin skills listed in `~/.claude/plugins/installed_plugins.json`,
     scoped by the enabled flag in `~/.claude/settings.json`.

Reads each file's YAML frontmatter for the description; supports inline
(`description: text`) and block scalar (`description: >` / `|`) forms.

Resolution is best-effort and forgiving: missing files / unparseable
frontmatter just produce a skill with `description=None`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    name: str  # `<skill>` for user, `<plugin>:<skill>` for plugin skills
    description: str | None
    source: str  # "user" or "plugin:<plugin-id>"


def list_skills() -> list[Skill]:
    """Enumerate user skills + legacy commands + enabled-plugin skills. Sorted by name."""
    home = Path.home()
    by_name: dict[str, Skill] = {}

    # 1. Legacy slash commands at ~/.claude/commands/<name>.md. Added first so
    #    user-level skills below can overwrite on name collision (skill wins).
    commands_dir = home / ".claude" / "commands"
    if commands_dir.is_dir():
        for f in sorted(commands_dir.iterdir()):
            if not f.is_file() or f.suffix != ".md" or f.name.startswith("."):
                continue
            name = f.stem
            desc = _read_description(f)
            by_name[name] = Skill(name=name, description=desc, source="command")

    # 2. User-level skills at ~/.claude/skills/<name>/SKILL.md.
    user_skills_dir = home / ".claude" / "skills"
    if user_skills_dir.is_dir():
        for d in sorted(user_skills_dir.iterdir()):
            if not d.is_dir():
                continue
            desc = _read_description(d / "SKILL.md")
            by_name[d.name] = Skill(name=d.name, description=desc, source="user")

    # 3. Plugin skills (namespaced as <plugin>:<skill>, never collide with the above).
    for plugin_id, install_path in _enabled_plugin_paths().items():
        skills_root = install_path / "skills"
        if not skills_root.is_dir():
            continue
        prefix = plugin_id.split("@", 1)[0]
        for d in sorted(skills_root.iterdir()):
            if not d.is_dir():
                continue
            desc = _read_description(d / "SKILL.md")
            qualified = f"{prefix}:{d.name}"
            by_name[qualified] = Skill(
                name=qualified,
                description=desc,
                source=f"plugin:{plugin_id}",
            )

    return sorted(by_name.values(), key=lambda s: s.name)


def _enabled_plugin_paths() -> dict[str, Path]:
    """Return {plugin_id: install_path} for every enabled plugin.

    Reads `enabledPlugins` from `~/.claude/settings.json` and joins against
    `~/.claude/plugins/installed_plugins.json` for the paths.
    """
    home = Path.home()
    settings_path = home / ".claude" / "settings.json"
    installed_path = home / ".claude" / "plugins" / "installed_plugins.json"

    enabled: dict[str, bool] = {}
    try:
        settings = json.loads(settings_path.read_text())
        if isinstance(settings, dict):
            ep = settings.get("enabledPlugins")
            if isinstance(ep, dict):
                enabled = {k: bool(v) for k, v in ep.items()}
    except (OSError, ValueError):
        pass

    installed: dict[str, list[dict]] = {}
    try:
        data = json.loads(installed_path.read_text())
        if isinstance(data, dict) and isinstance(data.get("plugins"), dict):
            installed = data["plugins"]
    except (OSError, ValueError):
        pass

    paths: dict[str, Path] = {}
    for plugin_id, is_enabled in enabled.items():
        if not is_enabled:
            continue
        entries = installed.get(plugin_id) or []
        if not entries:
            continue
        # Take the first install record's path (each plugin_id has 1 install).
        install_path = entries[0].get("installPath")
        if isinstance(install_path, str):
            paths[plugin_id] = Path(install_path)
    return paths


def _read_description(skill_md: Path) -> str | None:
    """Pull the `description:` value out of a YAML frontmatter.

    Handles three forms:
      description: inline text
      description: >          (folded, joined with spaces)
        line one
        line two
      description: |          (literal, joined with newlines)
        line one
        line two
    """
    try:
        text = skill_md.read_text()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None

    fm_lines = text[3:end].splitlines()
    for i, line in enumerate(fm_lines):
        stripped = line.strip()
        if not stripped.startswith("description:"):
            continue
        value = stripped[len("description:") :].strip()

        # Block scalar: > (folded) or | (literal).
        if value in (">", "|"):
            joiner = " " if value == ">" else "\n"
            collected: list[str] = []
            for follow in fm_lines[i + 1 :]:
                if follow.strip() == "":
                    collected.append("")
                    continue
                # Block scalar continues while the line is indented (any whitespace).
                if follow.startswith((" ", "\t")):
                    collected.append(follow.strip())
                else:
                    break
            joined = joiner.join(s for s in collected if s).strip()
            return joined or None

        # Inline form. Strip surrounding quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None
