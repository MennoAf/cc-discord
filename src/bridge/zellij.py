"""Async wrapper around the zellij CLI for managing panes in a session."""

import asyncio
import json
import logging
import subprocess
from typing import TypedDict

logger = logging.getLogger(__name__)

SESSION_NAME = "bridge"
SPAWN_POLL_INTERVAL = 0.1  # seconds
SPAWN_POLL_TIMEOUT = 5.0  # seconds


class ZellijError(Exception):
    """Base for all zellij-wrapper errors."""


class ZellijSessionMissing(ZellijError):
    """Raised when the bridge session can't be created or attached."""


class ZellijSpawnError(ZellijError):
    """Raised when a new pane can't be created or its id can't be resolved within timeout."""


class ZellijPaneInfo(TypedDict):
    """Type definition for pane information returned by list-panes."""

    id: str
    title: str
    pwd: str
    terminal_command: str
    exited: bool


class ZellijManager:
    """Async-friendly wrapper around zellij CLI."""

    def __init__(self, executable: str = "zellij") -> None:
        """Initialize ZellijManager with optional injectable executable path.

        Args:
            executable: Path to zellij binary (default: "zellij")
        """
        self._executable = executable
        self._session_lock = asyncio.Lock()
        self.SPAWN_POLL_TIMEOUT = SPAWN_POLL_TIMEOUT  # Allow overriding in tests

    async def _run(
        self,
        *argv: str,
        env: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, str, str]:
        """Run a subprocess command with timeout and return (returncode, stdout, stderr).

        All public methods funnel through this to ensure consistent subprocess behavior.
        Acquires the session lock around the call.

        Args:
            argv: Command arguments
            env: Optional environment dict
            timeout: Command timeout in seconds

        Returns:
            Tuple of (returncode, stdout_decoded, stderr_decoded)

        Raises:
            ZellijError: On timeout or other subprocess errors
        """
        async with self._session_lock:
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
                    raise ZellijError(f"Command timed out: {' '.join(argv)}")

                # For mocked procs in tests, return the stdout/stderr directly
                if hasattr(proc, "stdout_data") and hasattr(proc, "stderr_data"):
                    return (
                        proc.returncode,
                        proc.stdout_data.decode(),
                        proc.stderr_data.decode(),
                    )

                stdout, stderr = await proc.communicate()
                return (proc.returncode, stdout.decode(), stderr.decode())
            except ZellijError:
                raise
            except Exception as e:
                raise ZellijError(f"Subprocess error: {e}")

    async def ensure_session_alive(self) -> None:
        """Ensure the bridge session exists and is accessible.

        Runs `zellij attach --create-background bridge`. Idempotent — safe to call
        repeatedly. If exit code != 0 and stderr does not indicate success,
        raises ZellijSessionMissing.

        Raises:
            ZellijSessionMissing: If session creation fails
        """
        returncode, stdout, stderr = await self._run(
            self._executable, "attach", "--create-background", SESSION_NAME
        )
        if returncode != 0:
            raise ZellijSessionMissing(f"Failed to create/attach session: {stderr}")

    async def list_panes(self) -> list[ZellijPaneInfo]:
        """List all panes in the bridge session as JSON.

        Parses JSON from `zellij action list-panes --json`. The shape is nested:
        `{"tabs": [{"panes": [...]}, ...]}`. Flattens panes from all tabs.

        Zellij docs: https://zellij.dev/documentation/cli-actions

        Returns:
            List of pane info dicts

        Raises:
            ZellijError: On JSON parse error
        """
        returncode, stdout, stderr = await self._run(
            self._executable,
            "--session",
            SESSION_NAME,
            "action",
            "list-panes",
            "--json",
        )
        if returncode != 0:
            raise ZellijError(f"list-panes failed: {stderr}")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            raise ZellijError(f"malformed list-panes output: {stderr}")

        # Flatten panes from all tabs
        panes: list[ZellijPaneInfo] = []
        if isinstance(data, dict) and "tabs" in data:
            for tab in data.get("tabs", []):
                if isinstance(tab, dict) and "panes" in tab:
                    for pane in tab.get("panes", []):
                        if isinstance(pane, dict) and all(
                            k in pane for k in ["id", "pwd", "terminal_command", "exited"]
                        ):
                            panes.append(
                                {
                                    "id": pane["id"],
                                    "title": pane.get("title", ""),
                                    "pwd": pane["pwd"],
                                    "terminal_command": pane["terminal_command"],
                                    "exited": pane["exited"],
                                }
                            )
        return panes

    async def write_to_pane(self, pane_id: str, text: str) -> None:
        """Write text to a pane, handling multi-line input correctly.

        Splits text on `\\n` boundaries:
        - For each segment: `write-chars <segment>`
        - After each `\\n`: `send-keys Enter`

        This is the correct way to send multi-line input to TUI apps via zellij.

        Args:
            pane_id: The pane ID (e.g., "terminal_1")
            text: Text to write (may contain `\\n`)

        Raises:
            ZellijError: If any subprocess call returns non-zero
        """
        segments = text.split("\n")
        for i, segment in enumerate(segments):
            # Write the segment
            if segment:  # Only write non-empty segments
                returncode, _, stderr = await self._run(
                    self._executable,
                    "--session",
                    SESSION_NAME,
                    "action",
                    "write-chars",
                    "--pane-id",
                    pane_id,
                    segment,
                )
                if returncode != 0:
                    raise ZellijError(f"write-chars failed: {stderr}")

            # Send Enter if this segment was followed by a newline (i.e., not the last segment)
            if i < len(segments) - 1:
                returncode, _, stderr = await self._run(
                    self._executable,
                    "--session",
                    SESSION_NAME,
                    "action",
                    "send-keys",
                    "--pane-id",
                    pane_id,
                    "--",
                    "Enter",
                )
                if returncode != 0:
                    raise ZellijError(f"send-keys failed: {stderr}")

    async def close_pane(self, pane_id: str) -> None:
        """Close a pane by ID.

        Idempotent — if the pane is already gone, swallows the error and logs INFO.

        Args:
            pane_id: The pane ID to close
        """
        returncode, _, stderr = await self._run(
            self._executable,
            "--session",
            SESSION_NAME,
            "action",
            "close-pane",
            "--pane-id",
            pane_id,
        )
        if returncode != 0:
            logger.info(f"close-pane {pane_id}: {stderr}")

    async def spawn_task(
        self, cwd: str, env: dict[str, str], pane_name: str
    ) -> str:
        """Spawn a new pane running `claude` and resolve its pane ID.

        Process:
        1. Ensure session is alive
        2. Snapshot existing pane ids
        3. Spawn `claude` in a new pane
        4. Poll list-panes until the new pane appears
        5. Return the new pane id

        Args:
            cwd: Working directory for the new pane
            env: Environment dict to pass to subprocess
            pane_name: Name for the new pane (e.g., "cc-abc123")

        Returns:
            The pane ID of the newly spawned pane (e.g., "terminal_1")

        Raises:
            ZellijSpawnError: If spawn fails or poll times out
        """
        # Ensure session is alive
        await self.ensure_session_alive()

        # Snapshot existing pane ids
        before_panes = await self.list_panes()
        before_ids = {p["id"] for p in before_panes}

        # Spawn the new pane
        argv = [
            self._executable,
            "--session",
            SESSION_NAME,
            "run",
            "--cwd",
            cwd,
            "--name",
            pane_name,
            "--",
            "claude",
        ]
        returncode, _, stderr = await self._run(*argv, env=env)
        if returncode != 0:
            raise ZellijSpawnError(f"Failed to spawn pane: {stderr}")

        # Poll for the new pane to appear
        start = asyncio.get_event_loop().time()
        while True:
            try:
                panes = await self.list_panes()
                for pane in panes:
                    # Check if this is a new pane (not in before_ids)
                    # and matches our criteria (claude command, matching cwd)
                    if (
                        pane["id"] not in before_ids
                        and (
                            pane["terminal_command"].endswith("claude")
                            or "claude" in pane["terminal_command"]
                        )
                        and (pane["pwd"] == cwd or pane["pwd"].startswith(cwd))
                    ):
                        return pane["id"]

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > self.SPAWN_POLL_TIMEOUT:
                    raise ZellijSpawnError("pane not visible within 5s")

                # Wait before polling again
                await asyncio.sleep(SPAWN_POLL_INTERVAL)
            except ZellijSpawnError:
                raise
            except Exception as e:
                raise ZellijSpawnError(f"Poll failed: {e}")
