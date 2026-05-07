import json
import logging
import stat
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SECRETS_DIR = Path.home() / ".config" / "claude-discord-bridge"
SECRETS_FILE = SECRETS_DIR / "secrets.json"
REQUIRED_KEYS = ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID")


class SecretsError(RuntimeError):
    """Raised when secrets file is invalid or missing."""

    pass


@dataclass(frozen=True)
class Secrets:
    bot_token: str
    channel_id: int


def load_secrets(path: Path = SECRETS_FILE) -> Secrets:
    """Load secrets from JSON file.

    Reads the secrets file, validates required keys are present and non-empty,
    and coerces DISCORD_CHANNEL_ID to int.

    Raises SecretsError if the file is missing, unreadable, malformed JSON,
    missing keys, or has non-int channel ID. Error messages point users at
    'claude-discord-bridge init'.
    """
    if not path.exists():
        raise SecretsError(
            f"Secrets file not found at {path}. "
            f"Run 'claude-discord-bridge init' to create it."
        )

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SecretsError(
            f"Secrets file at {path} contains invalid JSON: {e}. "
            f"Run 'claude-discord-bridge init' to recreate it."
        ) from e
    except OSError as e:
        raise SecretsError(
            f"Cannot read secrets file at {path}: {e}. "
            f"Run 'claude-discord-bridge init' to recreate it."
        ) from e

    # Validate required keys are present and non-empty
    bot_token = data.get("DISCORD_BOT_TOKEN", "").strip()
    if not bot_token:
        raise SecretsError(
            f"DISCORD_BOT_TOKEN missing or empty in {path}. "
            f"Run 'claude-discord-bridge init' to set it."
        )

    channel_id_val = data.get("DISCORD_CHANNEL_ID")
    if channel_id_val is None or channel_id_val == "":
        raise SecretsError(
            f"DISCORD_CHANNEL_ID missing or empty in {path}. "
            f"Run 'claude-discord-bridge init' to set it."
        )

    try:
        channel_id = int(channel_id_val)
    except (ValueError, TypeError):
        raise SecretsError(
            f"DISCORD_CHANNEL_ID must be a number; got {channel_id_val!r} in {path}. "
            f"Run 'claude-discord-bridge init' to fix it."
        )

    logger.info("loaded secrets from %s", path)
    return Secrets(bot_token=bot_token, channel_id=channel_id)


def write_secrets(secrets: Secrets, path: Path = SECRETS_FILE) -> None:
    """Write secrets to a 0600 JSON file.

    Creates parent directories with 0700 mode. Writes JSON file with 0600 mode.
    """
    # Create parent directory with 0700 mode
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)

    # Write JSON
    data = {
        "DISCORD_BOT_TOKEN": secrets.bot_token,
        "DISCORD_CHANNEL_ID": secrets.channel_id,
    }
    path.write_text(json.dumps(data, indent=2))

    # Set file mode to 0600
    path.chmod(0o600)


def secrets_file_perms(path: Path = SECRETS_FILE) -> int | None:
    """Return the file's mode bits, or None if the file doesn't exist."""
    if not path.exists():
        return None
    return stat.S_IMODE(path.stat().st_mode)
