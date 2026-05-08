"""Async wrapper around the zellij CLI for managing per-task tabs in a session.

zellij 0.43+ removed `list-panes`, `--pane-id` targeting, and `send-keys`. We
target each task by tab name (e.g. `cc-aa429dc4`), focus the tab via
`go-to-tab-name` before issuing `write-chars`, and submit Enter via raw
`action write 13`.
"""

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Session name is configurable via env var so users can colocate bridge tasks
# with their existing zellij session (e.g. the one their SSH RemoteCommand
# attaches to). Defaults to `meow` to match Hailey's typical setup.
SESSION_NAME = os.environ.get("BRIDGE_ZELLIJ_SESSION", "meow")


def _session_already_exists(stderr: str) -> bool:
    """Whether stderr from `zellij attach --create-background` indicates the
    session is already alive (zellij ≥ 0.43 reports this as a non-zero exit)."""
    s = stderr.lower()
    return "already exists" in s or "already running" in s


def _running_inside_target_session() -> bool:
    """Whether the current process is running inside the configured session.

    zellij sets `ZELLIJ_SESSION_NAME` for processes spawned inside a session,
    and refuses `zellij attach --create-background <name>` when <name> matches
    the current session (panic at commands.rs: "trying to attach to the current
    session"). When colocated, the session is alive by definition — skip the
    attach call entirely.
    """
    return os.environ.get("ZELLIJ_SESSION_NAME") == SESSION_NAME


class ZellijError(Exception):
    """Base for all zellij-wrapper errors."""


class ZellijSessionMissing(ZellijError):
    """Raised when the bridge session can't be created or attached."""


class ZellijSpawnError(ZellijError):
    """Raised when a new task tab can't be created or its claude pane can't run."""


class ZellijManager:
    """Async-friendly wrapper around the zellij CLI.

    Each bridge task is one named tab in the configured session. Tabs are named
    `cc-<task_id_prefix>` and identified by that name throughout the API
    surface (the legacy `pane_id` parameter names now carry tab names).
    """

    def __init__(self, executable: str = "zellij") -> None:
        self._executable = executable
        self._session_lock = asyncio.Lock()

    async def _run_unlocked(
        self,
        *argv: str,
        env: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, str, str]:
        """Run a subprocess command with timeout. Caller must hold _session_lock
        if serialization is required."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                raise ZellijError(f"Command timed out: {' '.join(argv)}")

            stdout, stderr = await proc.communicate()
            return (proc.returncode, stdout.decode(), stderr.decode())
        except ZellijError:
            raise
        except Exception as e:
            raise ZellijError(f"Subprocess error: {e}") from e

    async def _run(
        self,
        *argv: str,
        env: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, str, str]:
        """Run a subprocess command, acquiring _session_lock for serialization."""
        async with self._session_lock:
            return await self._run_unlocked(*argv, env=env, timeout=timeout)

    async def ensure_session_alive(self) -> None:
        """Idempotently ensure the configured zellij session exists.

        zellij ≥ 0.43 returns non-zero with stderr "Session already exists" when
        the session is already running; treat that as success. If we're running
        inside the target session itself, skip the attach (zellij panics on
        self-attach).
        """
        if _running_inside_target_session():
            return
        returncode, _, stderr = await self._run(
            self._executable, "attach", "--create-background", SESSION_NAME
        )
        if returncode != 0 and not _session_already_exists(stderr):
            raise ZellijSessionMissing(f"Failed to create/attach session: {stderr}")

    async def list_panes(self) -> list[dict]:
        """List bridge-owned task tabs in the session.

        Returns a list of dicts shaped `{"id": tab_name, "exited": False}` for
        each tab whose name starts with `cc-`. The "id" field carries the tab
        name (preserving the historical `pane_id` contract). "exited" is always
        False because zellij 0.43 has no API to detect a dead pane in an
        existing tab — we infer death only from the tab disappearing entirely.
        """
        returncode, stdout, stderr = await self._run(
            self._executable, "--session", SESSION_NAME, "action", "query-tab-names"
        )
        if returncode != 0:
            raise ZellijError(f"query-tab-names failed: {stderr}")

        names = [line.strip() for line in stdout.splitlines() if line.strip()]
        return [{"id": n, "exited": False} for n in names if n.startswith("cc-")]

    async def write_to_pane(self, pane_id: str, text: str) -> None:
        """Type text into the task tab named `pane_id`.

        Single-line text is typed with `write-chars`. Multi-line text is
        wrapped in bracketed-paste markers (ESC[200~ ... ESC[201~) so Claude's
        TUI treats embedded newlines as content rather than Enter; a bare CR
        between segments would otherwise submit each line as its own prompt.
        Inside the paste block we use LF (byte 10) between segments. A
        trailing newline on the input always submits the buffered prompt via
        a final `action write 13` (CR).
        """
        submit = text.endswith("\n")
        body = text[:-1] if submit else text
        segments = body.split("\n")
        multiline = len(segments) > 1

        async with self._session_lock:
            rc, _, stderr = await self._run_unlocked(
                self._executable, "--session", SESSION_NAME,
                "action", "go-to-tab-name", pane_id,
            )
            if rc != 0:
                raise ZellijError(f"go-to-tab-name {pane_id!r} failed: {stderr}")

            if multiline:
                # ESC [ 2 0 0 ~  — begin bracketed paste
                await self._action_write_bytes(27, 91, 50, 48, 48, 126)
                for i, segment in enumerate(segments):
                    if segment:
                        # `--` terminates flag parsing; otherwise zellij
                        # silently drops segments that start with `-`
                        # (markdown bullets, diff lines, CLI flag-y prose).
                        rc, _, stderr = await self._run_unlocked(
                            self._executable, "--session", SESSION_NAME,
                            "action", "write-chars", "--", segment,
                        )
                        if rc != 0:
                            raise ZellijError(f"write-chars failed: {stderr}")
                    if i < len(segments) - 1:
                        await self._action_write_bytes(10)
                # ESC [ 2 0 1 ~  — end bracketed paste
                await self._action_write_bytes(27, 91, 50, 48, 49, 126)
            elif body:
                rc, _, stderr = await self._run_unlocked(
                    self._executable, "--session", SESSION_NAME,
                    "action", "write-chars", "--", body,
                )
                if rc != 0:
                    raise ZellijError(f"write-chars failed: {stderr}")

            if submit:
                await self._action_write_bytes(13)

    async def _action_write_bytes(self, *byte_vals: int) -> None:
        """Send raw bytes to the focused pane via `zellij action write`.

        `action write` accepts space-separated decimal byte values and emits
        them as a single contiguous write to the pane's stdin.
        """
        rc, _, stderr = await self._run_unlocked(
            self._executable, "--session", SESSION_NAME,
            "action", "write", *(str(b) for b in byte_vals),
        )
        if rc != 0:
            raise ZellijError(f"write {byte_vals!r} failed: {stderr}")

    async def close_pane(self, pane_id: str) -> None:
        """Close the task tab named `pane_id`. Idempotent — missing tab is a no-op."""
        async with self._session_lock:
            rc, _, stderr = await self._run_unlocked(
                self._executable, "--session", SESSION_NAME,
                "action", "go-to-tab-name", pane_id,
            )
            if rc != 0:
                logger.info("close_pane %s: tab not found, treating as already closed", pane_id)
                return
            rc, _, stderr = await self._run_unlocked(
                self._executable, "--session", SESSION_NAME,
                "action", "close-tab",
            )
            if rc != 0:
                logger.info("close-tab %s: %s", pane_id, stderr)

    async def spawn_task(
        self,
        cwd: str,
        env: dict[str, str],
        pane_name: str,
        *,
        extra_argv: list[str] | None = None,
    ) -> str:
        """Spawn a new task tab named `pane_name` running claude in `cwd`.

        Acquires the session lock for the entire operation so concurrent spawns
        don't race on tab focus. Sequence:
        1. Ensure session is alive (or skip if colocated).
        2. `new-tab --name <pane_name> --cwd <cwd>` — creates and focuses the tab.
        3. `run --close-on-exit --cwd <cwd> -- env K=V... claude [extra_argv...]`
           — opens a pane in the just-focused tab running claude.

        The `env` dict is injected via the `env(1)` prefix because zellij is
        client-server: the env we set on the `zellij run` subprocess is invisible
        to the spawned process, which inherits the *server's* env. `env(1)` sets
        the vars at exec time, which works regardless of who started the server.

        Returns the tab name (which is `pane_name`).

        Raises ZellijSpawnError on any failure.
        """
        async with self._session_lock:
            if not _running_inside_target_session():
                rc, _, stderr = await self._run_unlocked(
                    self._executable, "attach", "--create-background", SESSION_NAME
                )
                if rc != 0 and not _session_already_exists(stderr):
                    raise ZellijSpawnError(f"Failed to create/attach session: {stderr}")

            rc, _, stderr = await self._run_unlocked(
                self._executable, "--session", SESSION_NAME,
                "action", "new-tab", "--name", pane_name, "--cwd", cwd,
            )
            if rc != 0:
                raise ZellijSpawnError(f"new-tab {pane_name!r} failed: {stderr}")

            run_argv = [
                self._executable, "--session", SESSION_NAME, "run",
                "--close-on-exit", "--cwd", cwd, "--",
                "env",
                *(f"{k}={v}" for k, v in env.items()),
                "claude",
            ]
            if extra_argv:
                run_argv.extend(extra_argv)
            rc, _, stderr = await self._run_unlocked(*run_argv)
            if rc != 0:
                raise ZellijSpawnError(f"run claude failed: {stderr}")

            return pane_name
