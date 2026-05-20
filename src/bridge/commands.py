"""discord.app_commands tree for task lifecycle control.

Registered guild-scoped (instant sync). Bot must finish on_ready before sync runs.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord import app_commands

from bridge import skills, usage
from bridge.bot import Bot, BotMissingPermission
from bridge.projects import Project
from bridge.tasks import (
    SPAWN_BIND_TIMEOUT_SECS,
    Task,
    TaskNotFound,
    TaskRegistry,
    TaskRestartError,
    TaskSpawnError,
)

logger = logging.getLogger(__name__)


class _NotInTaskThread(Exception):
    """Raised when a thread-context command is used outside a task thread."""

    pass


def build_tree(
    bot: Bot,
    registry: TaskRegistry,
    projects: list[Project] | None = None,
) -> app_commands.CommandTree:
    """Construct and return the CommandTree (not yet synced; caller decides when).

    `projects` is the cached list of spawnable projects enumerated from
    BRIDGE_PROJECT_ROOTS at startup. When None or empty, /spawn still
    registers but reports that no roots are configured.
    """
    tree = app_commands.CommandTree(bot.client)
    projects_list: list[Project] = list(projects or [])
    # Key used as the slash command choice value: `{root_label}/{name}`.
    # Bounded well under Discord's 100-char value limit and unique enough
    # to disambiguate same-named folders across roots.
    projects_by_key: dict[str, Project] = {
        f"{p.root_label}/{p.name}": p for p in projects_list
    }

    @tree.command(name="start", description="Start a new Claude task in a fresh thread")
    @app_commands.describe(
        cwd="Working directory the task should run in (must exist)",
        prompt="Optional first message to send after the task is bound",
    )
    async def start(
        interaction: discord.Interaction,
        cwd: str,
        prompt: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = await registry.spawn_task(cwd=cwd, prompt=None)  # prompt handled below
        except TaskSpawnError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return

        # If a prompt was provided, wait for SessionStart to bind, then write it.
        if prompt:
            try:
                await _wait_for_session_bind(
                    registry, task.task_id, timeout=SPAWN_BIND_TIMEOUT_SECS
                )
                await registry.write_initial_prompt(task.task_id, prompt)
            except asyncio.TimeoutError:
                logger.warning(
                    "task %s did not bind within %.0fs",
                    task.task_id, SPAWN_BIND_TIMEOUT_SECS,
                )

        thread_url = f"https://discord.com/channels/{interaction.guild_id}/{task.thread_id}"
        await interaction.followup.send(
            f"✅ Started task `{task.task_id[:8]}` → <#{task.thread_id}> ({thread_url})",
            ephemeral=True,
        )

    async def _project_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        cur = current.lower()
        out: list[app_commands.Choice[str]] = []
        for key, proj in projects_by_key.items():
            if cur and cur not in proj.name.lower() and cur not in proj.root_label.lower():
                continue
            label = f"{proj.name} — {proj.root_label}"
            out.append(
                app_commands.Choice(name=label[:100], value=key[:100])
            )
            if len(out) >= 25:
                break
        return out

    @tree.command(
        name="spawn",
        description="Spawn a Claude task in a configured project folder (see BRIDGE_PROJECT_ROOTS)",
    )
    @app_commands.describe(
        project="Project folder (autocomplete shows immediate subfolders of BRIDGE_PROJECT_ROOTS)",
        prompt="Optional first message to send after the task is bound",
    )
    @app_commands.autocomplete(project=_project_autocomplete)
    async def spawn(
        interaction: discord.Interaction,
        project: str,
        prompt: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        if not projects_by_key:
            await interaction.followup.send(
                "❌ No project roots configured. Set `BRIDGE_PROJECT_ROOTS` "
                "(colon-separated parent paths) and restart the daemon.",
                ephemeral=True,
            )
            return
        proj = projects_by_key.get(project)
        if proj is None:
            await interaction.followup.send(
                f"❌ Unknown project `{project}`. Pick one from the autocomplete list.",
                ephemeral=True,
            )
            return
        try:
            task = await registry.spawn_task(cwd=str(proj.path), prompt=None)
        except TaskSpawnError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return

        if prompt:
            try:
                await _wait_for_session_bind(
                    registry, task.task_id, timeout=SPAWN_BIND_TIMEOUT_SECS
                )
                await registry.write_initial_prompt(task.task_id, prompt)
            except asyncio.TimeoutError:
                logger.warning(
                    "task %s did not bind within %.0fs",
                    task.task_id, SPAWN_BIND_TIMEOUT_SECS,
                )

        thread_url = f"https://discord.com/channels/{interaction.guild_id}/{task.thread_id}"
        await interaction.followup.send(
            f"✅ Started task `{task.task_id[:8]}` in `{proj.name}` "
            f"({proj.root_label}) → <#{task.thread_id}> ({thread_url})",
            ephemeral=True,
        )

    @tree.command(name="list", description="List active tasks")
    async def list_cmd(interaction: discord.Interaction) -> None:
        tasks = await registry.list_tasks()
        if not tasks:
            await interaction.response.send_message("No active tasks.", ephemeral=True)
            return
        lines = ["**Active tasks:**"]
        for t in tasks:
            cwd_leaf = Path(t.cwd).name or "/"
            ago = _humanize_age(t.last_activity)
            lines.append(
                f"- `{t.task_id[:8]}` · {cwd_leaf} · {t.status} · {ago} · <#{t.thread_id}>"
            )
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @tree.command(name="stop", description="Gracefully stop a task")
    @app_commands.describe(thread="Thread to stop (defaults to invocation thread)")
    async def stop(
        interaction: discord.Interaction,
        thread: discord.Thread | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, thread)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        try:
            stopped = await registry.stop_task(task.task_id)
        except TaskNotFound:
            await interaction.followup.send("❌ Task not found", ephemeral=True)
            return
        if stopped:
            await interaction.followup.send(f"✅ Stopped `{task.task_id[:8]}`", ephemeral=True)
        else:
            await interaction.followup.send(
                f"⚠️ Stop timed out for `{task.task_id[:8]}`. Use `/kill` to force.",
                ephemeral=True,
            )

    @tree.command(name="kill", description="Immediately kill a task (close its pane)")
    @app_commands.describe(thread="Thread to kill (defaults to invocation thread)")
    async def kill(
        interaction: discord.Interaction,
        thread: discord.Thread | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, thread)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        try:
            await registry.kill_task(task.task_id)
        except TaskNotFound:
            await interaction.followup.send("❌ Task not found", ephemeral=True)
            return
        await interaction.followup.send(f"💥 Killed `{task.task_id[:8]}`", ephemeral=True)

    @tree.command(name="restart", description="Restart a task with --resume")
    @app_commands.describe(thread="Thread to restart (defaults to invocation thread)")
    async def restart(
        interaction: discord.Interaction,
        thread: discord.Thread | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, thread)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        try:
            await registry.restart_task(task.task_id)
        except (TaskNotFound, TaskRestartError) as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        await interaction.followup.send(f"🔄 Restarted `{task.task_id[:8]}`", ephemeral=True)

    async def _skill_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        cur = current.lower()
        out: list[app_commands.Choice[str]] = []
        for s in skills.list_skills():
            if cur and cur not in s.name.lower() and not (
                s.description and cur in s.description.lower()
            ):
                continue
            label = s.name
            if s.description:
                label = f"{s.name} — {s.description}"
            # Discord limits both the displayed name and submitted value to 100 chars.
            out.append(
                app_commands.Choice(name=label[:100], value=s.name[:100])
            )
            if len(out) >= 25:
                break
        return out

    @tree.command(name="skill", description="Invoke a Claude Code skill in the task's session")
    @app_commands.describe(
        name="Skill name (autocomplete shows available skills + their descriptions)",
        args="Optional arguments to pass after the skill name",
    )
    @app_commands.autocomplete(name=_skill_autocomplete)
    async def skill_cmd(
        interaction: discord.Interaction,
        name: str,
        args: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, None)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        try:
            await registry.invoke_skill(task.task_id, name, args)
        except (TaskNotFound, TaskSpawnError) as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        rendered = f"/{name}" + (f" {args}" if args else "")
        await interaction.followup.send(
            f"✅ Sent `{rendered}` to `{task.task_id[:8]}`", ephemeral=True
        )

    @tree.command(
        name="rename",
        description="Rename the task's thread (omit name to auto-generate via claude -p)",
    )
    @app_commands.describe(name="New thread name; omit to auto-generate")
    async def rename_cmd(
        interaction: discord.Interaction,
        name: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, None)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return

        if name is None:
            try:
                generated = await registry.generate_thread_name(task.task_id)
            except Exception as e:
                await interaction.followup.send(
                    f"❌ Generation failed: {e}", ephemeral=True
                )
                return
            if not generated:
                await interaction.followup.send(
                    "❌ Couldn't auto-generate (no transcript yet, or claude -p errored). Pass a name explicitly.",
                    ephemeral=True,
                )
                return
            name = generated

        # Discord thread names: 1–100 chars, no newlines.
        cleaned = " ".join(name.split())[:100]
        if not cleaned:
            await interaction.followup.send("❌ Empty name.", ephemeral=True)
            return
        try:
            await bot.rename_thread(task.thread_id, cleaned)
        except Exception as e:
            await interaction.followup.send(
                f"❌ Rename failed: {e}", ephemeral=True
            )
            return
        await interaction.followup.send(f"✏️ Renamed to `{cleaned}`", ephemeral=True)

    @tree.command(name="stats", description="Show model / token / cost stats for a task")
    @app_commands.describe(thread="Thread to inspect (defaults to invocation thread)")
    async def stats_cmd(
        interaction: discord.Interaction,
        thread: discord.Thread | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, thread)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        if not task.current_transcript_path:
            await interaction.followup.send(
                "❌ Task has no transcript yet — wait for the first turn.",
                ephemeral=True,
            )
            return
        stats = usage.compute_stats(Path(task.current_transcript_path))
        if stats is None:
            await interaction.followup.send(
                "❌ No usage data in transcript yet.", ephemeral=True
            )
            return
        await interaction.followup.send(usage.format_summary(stats), ephemeral=True)

    @tree.command(
        name="tasks",
        description="Show claude's current session task list (mirrored by the bridge)",
    )
    @app_commands.describe(thread="Thread to inspect (defaults to invocation thread)")
    async def tasks_cmd(
        interaction: discord.Interaction,
        thread: discord.Thread | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            task = _resolve_task(registry, interaction, thread)
        except _NotInTaskThread as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        if not task.task_list_state:
            await interaction.followup.send(
                "ℹ No tasks tracked yet — claude hasn't called TaskCreate "
                "in this session (or the daemon was restarted since the last call).",
                ephemeral=True,
            )
            return
        embed = registry._render_task_list_embed(task)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @tree.command(
        name="pin",
        description="Create a Discord channel bound to a project; messages auto-spawn a Claude session",
    )
    @app_commands.describe(
        name="Optional channel name (default: cwd basename, normalized)",
        project=(
            "Project to bind (autocomplete) — required when /pin runs outside "
            "a task thread; ignored if inside one (the thread's cwd is inherited)"
        ),
    )
    @app_commands.autocomplete(project=_project_autocomplete)
    async def pin_cmd(
        interaction: discord.Interaction,
        name: str | None = None,
        project: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        # cwd source: inherit from task thread if invoked inside one; otherwise
        # require an explicit project pick.
        existing_task = registry.get_by_thread_id(interaction.channel_id or 0)
        if existing_task is not None:
            cwd = existing_task.cwd
            source = f"inherited from thread <#{existing_task.thread_id}>"
        else:
            if not projects_by_key:
                await interaction.followup.send(
                    "❌ Not inside a task thread, and no project roots configured. "
                    "Set `BRIDGE_PROJECT_ROOTS` and restart, or run `/pin` inside an "
                    "existing task thread.",
                    ephemeral=True,
                )
                return
            if not project:
                await interaction.followup.send(
                    "❌ Outside a task thread — pass `project:` (autocomplete shows "
                    "subfolders of BRIDGE_PROJECT_ROOTS).",
                    ephemeral=True,
                )
                return
            proj = projects_by_key.get(project)
            if proj is None:
                await interaction.followup.send(
                    f"❌ Unknown project `{project}`. Pick one from the autocomplete list.",
                    ephemeral=True,
                )
                return
            cwd = str(proj.path)
            source = f"project `{proj.name}` ({proj.root_label})"

        channel_name = _sanitize_channel_name(name or Path(cwd).name)
        try:
            channel_id = await bot.create_channel(channel_name)
        except BotMissingPermission as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        except Exception as e:
            await interaction.followup.send(
                f"❌ Channel creation failed: {e}", ephemeral=True
            )
            return

        try:
            await registry.pin_channel(channel_id, cwd)
        except ValueError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return

        await interaction.followup.send(
            f"📌 Pinned <#{channel_id}> → `{cwd}` ({source}). "
            "Send a message in that channel to wake a Claude session.",
            ephemeral=True,
        )

    @tree.command(
        name="unpin",
        description="Remove the pin binding from the current channel (channel itself is not deleted)",
    )
    async def unpin_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        channel_id = interaction.channel_id or 0
        removed = await registry.unpin_channel(channel_id)
        if removed:
            await interaction.followup.send(
                f"📍 Unpinned <#{channel_id}>. Future messages here won't auto-spawn.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "ℹ This channel isn't pinned.", ephemeral=True
            )

    return tree


def _resolve_task(
    registry: TaskRegistry, interaction: discord.Interaction, override: discord.Thread | None
) -> Task:
    """Resolve the task context from interaction or override thread.

    Raises _NotInTaskThread if no task is bound to the thread.
    """
    target_id = override.id if override else interaction.channel_id
    task = registry.get_by_thread_id(target_id)
    if task is None:
        raise _NotInTaskThread(
            "This command must run in a task thread (or pass `thread:` arg)."
        )
    return task


_CHANNEL_NAME_INVALID = re.compile(r"[^a-z0-9_-]+")
_CHANNEL_NAME_COLLAPSE = re.compile(r"-+")


def _sanitize_channel_name(name: str) -> str:
    """Coerce a string into a Discord text-channel-name-safe form.

    Discord text channels are 1–100 chars, lowercase letters/digits/`-`/`_`.
    Discord auto-normalizes on create, but doing it client-side surfaces a
    helpful error earlier and keeps the visible channel name stable.
    Returns a non-empty string; falls back to `cc-pin` if normalization
    collapses to empty.
    """
    cleaned = _CHANNEL_NAME_INVALID.sub("-", name.lower())
    cleaned = _CHANNEL_NAME_COLLAPSE.sub("-", cleaned).strip("-")
    return cleaned[:100] or "cc-pin"


def _humanize_age(epoch: int) -> str:
    """Format an epoch timestamp as a human-readable age string."""
    delta = datetime.now(timezone.utc).timestamp() - epoch
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


async def _wait_for_session_bind(
    registry: TaskRegistry, task_id: str, *, timeout: float
) -> None:
    """Poll until task.current_claude_session_id is set or timeout.

    Raises asyncio.TimeoutError if the session doesn't bind within timeout seconds.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        task = registry.get_by_task_id(task_id)
        if task is not None and task.current_claude_session_id is not None:
            return
        await asyncio.sleep(0.1)
    raise asyncio.TimeoutError()
