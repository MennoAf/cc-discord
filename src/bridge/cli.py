"""CLI entrypoints for the bridge daemon."""

import asyncio
import logging
import os
import stat
import sys
from pathlib import Path

import click

from bridge.bot import Bot
from bridge.secrets import SECRETS_FILE, SecretsError, load_secrets, write_secrets, Secrets
from bridge.server import serve as serve_server

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Claude Code <-> Discord bridge daemon."""
    pass


@cli.command()
def init() -> None:
    """Interactive bootstrap: collect bot token and channel ID, write secrets file.

    Prompts for:
    - DISCORD_BOT_TOKEN (hidden input)
    - DISCORD_CHANNEL_ID (validated as positive integer)

    Writes to ~/.config/claude-discord-bridge/secrets.json (mode 0600).
    """
    click.echo("Welcome to the Claude Code <-> Discord bridge.")
    click.echo()
    click.echo(
        "This wizard will set up the bridge to post messages to Discord. "
        "You'll need:"
    )
    click.echo("  - A Discord bot token (from Discord Developer Portal)")
    click.echo("  - A Discord channel ID (from a text channel you own)")
    click.echo()

    # Resolve secrets path from env var for testability
    secrets_path = Path(os.environ.get("BRIDGE_SECRETS_PATH", str(SECRETS_FILE)))

    # Check if file already exists
    if secrets_path.exists():
        if not click.confirm(
            f"Secrets file already exists at {secrets_path}. Overwrite?",
            abort=True,
        ):
            return

    # Prompt for bot token (hidden)
    bot_token = click.prompt(
        "DISCORD_BOT_TOKEN", hide_input=True, confirmation_prompt=False
    )

    # Prompt for channel ID (with validation)
    while True:
        channel_id_input = click.prompt("DISCORD_CHANNEL_ID")
        try:
            channel_id = int(channel_id_input)
            if channel_id <= 0:
                click.echo("Channel ID must be a positive integer.")
                continue
            break
        except ValueError:
            click.echo("Channel ID must be a positive integer.")

    # Print reminders before writing
    click.echo()
    click.echo("Important reminders:")
    click.echo(
        "1. You must enable Privileged Gateway Intent: Message Content for the bot at"
    )
    click.echo("   Discord Developer Portal > Applications > [your app] > Bot")
    click.echo()
    click.echo(
        "2. The bot must be a member of the guild containing channel ID {0}".format(
            channel_id
        )
    )
    click.echo("   and have these permissions:")
    click.echo("   - View Channel")
    click.echo("   - Send Messages")
    click.echo("   - Create Public Threads")
    click.echo()

    # Write secrets
    secrets = Secrets(bot_token=bot_token, channel_id=channel_id)
    write_secrets(secrets, path=secrets_path)

    # Verify mode
    perms = stat.S_IMODE(secrets_path.stat().st_mode)
    if perms != 0o600:
        # Delete the dangling permissive file before failing
        secrets_path.unlink()
        click.echo(
            f"Error: file mode is {oct(perms)}, expected 0o600. "
            f"Secrets file deleted. Please check your filesystem settings.",
            err=True
        )
        sys.exit(1)

    click.echo()
    click.echo(
        f"Wrote secrets to {secrets_path} (mode 0600). "
        "Start the daemon with:"
    )
    click.echo("  claude-discord-bridge serve")
    click.echo()
    click.echo(
        "Or use the systemd unit at:"
    )
    click.echo("  packaging/claude-discord-bridge.service")


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=8787,
    type=int,
    help="Port to bind to (default: 8787)",
)
def serve(host: str, port: int) -> None:
    """Run the bridge daemon.

    Loads secrets from ~/.config/claude-discord-bridge/secrets.json and starts
    the HTTP server + Discord bot.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve secrets path from env var for testability
    secrets_path = Path(os.environ.get("BRIDGE_SECRETS_PATH", str(SECRETS_FILE)))

    try:
        secrets = load_secrets(path=secrets_path)
    except SecretsError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    asyncio.run(serve_server(secrets, host=host, port=port))


def main() -> None:
    """Entry point for the CLI (referenced by pyproject.toml [project.scripts])."""
    cli()
