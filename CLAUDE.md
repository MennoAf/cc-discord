# claude-discord-bridge

Localhost HTTP bridge between Claude Code sessions and Discord. Single-process Python daemon — `aiohttp` server and `discord.py` client share one asyncio event loop.

Freshness: 2026-05-08

## Repo location and tooling

This repo lives at `/home/discord/claude-discord-bridge`, **outside** the `/home/discord/discord` monorepo. `clyde`, `clint`, the monorepo's pre-commit hooks, and Buildkite CI do not apply here. Don't import from or symlink into the monorepo.

Python is pinned to 3.12 via `uv` (`.python-version`). The system `python3` is 3.10 — always invoke through uv:

- Tests: `uv run pytest` (not `pytest`)
- Run daemon in foreground: `scripts/run-foreground.sh` (uses `uv run`)

## Gotchas

- **`MESSAGE_CONTENT` privileged intent.** `bot.py` sets `intents.message_content = True` because reply routing reads message text. The bot user in the Discord Developer Portal must have this intent enabled, or `on_message` payloads arrive empty. The `init` wizard prints a reminder; agents adding new gateway features should not forget it.
- **Hooks must always exit 0.** `hooks/notify-stop.py` and `hooks/notify-notification.py` wrap `main()` in `try/except: pass; finally: sys.exit(0)` on purpose — a Claude Code Stop/Notification hook that fails non-zero degrades the user's session. Preserve that contract when editing.
- **FIFO ordering for `/v1/ask` is enforced by `AskLockMap`, not `Listener`.** `Listener.register()` raises `RuntimeError` if a thread already has a pending ask — this is an invariant guard, not the queueing mechanism. The per-thread `asyncio.Lock` in `server.AskLockMap` must be acquired *before* posting the question and *released* after `unregister`. Any new `/v1/ask`-style endpoint must follow the same lock-then-register pattern.
- **Single event loop, shared by aiohttp + discord.py.** Long blocking work (sync DB calls, `time.sleep`, `requests`) inside any handler starves both the HTTP server and the Discord gateway. Use the async equivalents (`aiosqlite`, `asyncio.sleep`, `aiohttp` client).
- **`SKILL.md` in `skills/` is symlinked into `~/.claude/skills/ask-discord/SKILL.md`.** Edit the file in this repo; the live skill follows. Don't duplicate.
- **`cli doctor` checks settings.json hook paths against `bridge.__file__`.** If you `uv tool install .` the bridge into `~/.local/bin`, `bridge.__file__` resolves into the uv tool venv, not this repo. The doctor's hook-path check expects `<repo>/hooks/notify-*.py` paths in `~/.claude/settings.json` to match wherever the package is currently importing from. Run `doctor` from the same install you registered hooks against.

## Deployment paths

The systemd unit at `packaging/claude-discord-bridge.service` hardcodes `%h/.local/bin/claude-discord-bridge` — it assumes `uv tool install .`, not `uv run`. The two install paths are not interchangeable.

`systemctl --user` is **not** available on the coder workstation by default ("Operation not permitted"). The verified-working path is `scripts/run-foreground.sh` under tmux/nohup. To use real systemd, the user must first run `loginctl enable-linger $USER`.

## Architecture quick reference

- `src/bridge/server.py` — aiohttp app, endpoints `/v1/notify`, `/v1/ask`, `/v1/health`. `AskLockMap` lives here.
- `src/bridge/bot.py` — `discord.py` wrapper. `_chunk()` and `_extract_images()` are lifted verbatim from `/home/discord/victrola/src/discord_bot/bot.py` — keep them in sync if upstream changes.
- `src/bridge/threads.py` — `ThreadRegistry` owns session_id→thread_id mapping with 404 recovery. Single global lock is intentional (per-session contention is rare).
- `src/bridge/listener.py` — sliding-window coalescing for `/v1/ask` replies. `GRACE_SECS = 3.0` default; tests override.
- `src/bridge/state.py` — aiosqlite, WAL mode, `sessions` table.
- `src/bridge/secrets.py` — 0600 JSON at `~/.config/claude-discord-bridge/secrets.json`.
- `hooks/` — Claude Code Stop/Notification hooks. Posts to `BRIDGE_URL` (default `http://127.0.0.1:8787`); falls back to a Discord webhook URL at `~/.claude/discord-notify-webhook` if the bridge is down.
- `skills/` — `/ask-discord` skill. `SKILL.md` is symlinked into `~/.claude/skills/ask-discord/`.
