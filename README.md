# claude-discord-bridge

Localhost HTTP bridge between Claude Code sessions and Discord. Long turns ping your phone, permission prompts surface in a thread, and Claude can `/ask-discord <question>` when it's blocked and you're away from the keyboard.

## What it does

Runs as a small Python daemon (`aiohttp` + `discord.py`) on `127.0.0.1:8787`. Three things hang off it:

- **Stop hook** — `~/.claude/hooks/notify-long-task.sh` is replaced by a Python script that pings Discord when a Claude turn took >10 minutes. Result lands in a per-session thread instead of the channel root.
- **Notification hook** — Permission prompts and idle states surface as `⏸ awaiting input` in the same thread.
- **`/ask-discord` skill** — Claude calls this when blocked on a decision; the question lands in the thread, the daemon waits up to 15 minutes for your reply, and Claude continues with whatever you say.

A separate webhook URL at `~/.claude/discord-notify-webhook` is used as a fallback when the daemon isn't running, so you don't lose pings if you forgot to start it.

## Prereqs

- Python 3.12 managed by [uv](https://github.com/astral-sh/uv) (the repo pins it via `.python-version`).
- A Discord application with a bot, message-content intent enabled, invited to a guild you control, with permission to view + send messages + create public threads in one channel.
- [Claude Code](https://docs.claude.com/claude-code) installed.

## Setup

### 1. Discord bot

1. https://discord.com/developers/applications → **New Application** → **Bot** tab → **Reset Token**, copy it.
2. **Privileged Gateway Intents** → enable **Message Content Intent**. Save.
3. **OAuth2 → URL Generator** → scopes: `bot` → bot permissions: `View Channels`, `Send Messages`, `Create Public Threads`, `Send Messages in Threads`, `Read Message History` → open the generated URL → invite the bot to your server.
4. In the Discord client: User Settings → Advanced → enable **Developer Mode** → right-click the target channel → **Copy Channel ID**.

### 2. Bridge daemon

```bash
git clone https://github.com/haileyok/cc-discord.git claude-discord-bridge
cd claude-discord-bridge
uv sync
uv run claude-discord-bridge init
```

`init` prompts for the bot token and channel ID, writes `~/.config/claude-discord-bridge/secrets.json` at mode `0600`, validates the token by connecting to Discord (15s timeout), and posts a confirmation message to your channel. If the token's wrong it exits 2 and leaves the secrets file so you can fix and retry.

### 3. Wire Claude Code hooks

The Stop and Notification hooks are referenced by absolute path from `~/.claude/settings.json`. Open it and add (or merge with existing `hooks`):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "python3 /home/<you>/claude-discord-bridge/hooks/notify-stop.py", "async": true }
        ]
      }
    ],
    "Notification": [
      {
        "hooks": [
          { "type": "command", "command": "python3 /home/<you>/claude-discord-bridge/hooks/notify-notification.py", "async": true }
        ]
      }
    ]
  }
}
```

Replace `/home/<you>/claude-discord-bridge` with the actual repo path. Validate with `python3 -m json.tool ~/.claude/settings.json > /dev/null`.

If you already use `~/.claude/hooks/notify-long-task.sh` (or any other Stop hook), keep it on disk as rollback insurance — both hooks can coexist; this one just supersedes it.

### 4. Install the `/ask-discord` skill

Claude Code discovers skills under `~/.claude/skills/<name>/SKILL.md`. The bridge ships the source-of-truth markdown in the repo; symlink it into place:

```bash
mkdir -p ~/.claude/skills/ask-discord
ln -sfn "$(pwd)/skills/SKILL.md" ~/.claude/skills/ask-discord/SKILL.md
```

After symlinking, run `/reload-plugins` in a Claude Code session — `/ask-discord` will appear in the slash-command picker.

### 5. Webhook fallback (optional)

Create a Discord channel webhook (channel settings → Integrations → Webhooks → New Webhook → copy URL), then write the URL to `~/.claude/discord-notify-webhook`:

```bash
echo 'https://discord.com/api/webhooks/...' > ~/.claude/discord-notify-webhook
chmod 0600 ~/.claude/discord-notify-webhook
```

When the daemon's down, the Stop and Notification hooks fall back to this webhook (channel root instead of a thread), so you still get pinged.

### 6. Start the daemon

**Foreground (simplest):**
```bash
uv run claude-discord-bridge serve
```
Wait for `Bot ready as <name>, watching #<channel>`. Use `tmux` or `nohup` if you want it to outlive the shell.

**systemd user unit (survives reboots):**
```bash
uv tool install .                      # places `claude-discord-bridge` at ~/.local/bin/
bash scripts/install-systemd-user.sh   # copies the unit file into ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now claude-discord-bridge
```
If `systemctl --user` errors with `Operation not permitted`, run `sudo loginctl enable-linger $USER` first.

### 7. Verify

```bash
uv run claude-discord-bridge doctor
```
You should see `[ok]` for each of: secrets file present, secrets file mode `0600`, daemon health, settings.json hooks, skill symlink. `[fail]` lines tell you what to fix; `[warn]` lines are non-blocking.

## Usage

Once the daemon is running, the four surfaces work without further intervention:

| Surface | Trigger |
|---|---|
| Long turn ping | Run any Claude Code turn that takes >10 minutes |
| Permission prompt ping | Run any Claude Code action that needs your approval |
| `/ask-discord` from inside Claude | Ask Claude to use `/ask-discord` when it's blocked |
| Manual `POST /v1/notify` | `curl -X POST http://127.0.0.1:8787/v1/notify -H 'Content-Type: application/json' -d '{"session_id":"...","cwd":"...","message":"..."}'` |
| Manual `POST /v1/ask` | Same, but `/v1/ask` with a `question` field; blocks for the reply (default 15 min, capped at 60) |
| Manual `GET /v1/health` | `curl http://127.0.0.1:8787/v1/health` |

Threads are named `cc · <cwd-leaf> · <session-prefix>`. Same `session_id` always routes to the same thread; different sessions get different threads. Mappings persist in SQLite at `~/.local/state/claude-discord-bridge/state.db` and survive daemon restarts. Archived/deleted threads recreate transparently.

## Architecture

Single-process Python daemon. `aiohttp.web.AppRunner` and `discord.py` share one asyncio event loop. Per-session thread mapping lives in SQLite. Reply routing uses a per-thread `asyncio.Lock` (FIFO) plus a sliding 3-second coalescing window so multi-message replies fold into one response.

| File | Role |
|---|---|
| `src/bridge/server.py` | aiohttp app, `/v1/notify`, `/v1/ask`, `/v1/health` |
| `src/bridge/bot.py` | discord.py wrapper, chunked send, on_message dispatch |
| `src/bridge/threads.py` | session_id → thread_id with create-on-miss + recreate-on-404 |
| `src/bridge/listener.py` | Pending-ask state, sliding coalescing window, future lifecycle |
| `src/bridge/state.py` | aiosqlite, sessions table |
| `src/bridge/secrets.py` | 0600 JSON loader/writer |
| `src/bridge/cli.py` | click CLI: `init`, `serve`, `doctor` |
| `hooks/notify-stop.py` | Stop hook (long-turn ping) |
| `hooks/notify-notification.py` | Notification hook (permission/idle ping) |
| `skills/ask_discord.py` | `/ask-discord` script body |
| `skills/SKILL.md` | Skill instructions for Claude (symlinked into `~/.claude/skills/ask-discord/`) |

See `CLAUDE.md` for gotchas and tooling notes; the implementation plan lives in the parent worktree at `docs/implementation-plans/2026-05-07-claude-discord-bridge/`.

## Development

```bash
uv run pytest -q       # full test suite (163 tests)
uv run pytest -q tests/test_<module>.py
```

Tests use a `FakeBot` and in-memory SQLite, so the suite never hits real Discord. End-to-end smoke against a real bot is documented in `docs/test-plans/`.

## License

TBD.
