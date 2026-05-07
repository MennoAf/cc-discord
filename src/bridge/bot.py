"""Discord bot client wrapper with chunked send."""

import asyncio
import base64
import contextlib
import logging
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Lifted verbatim from /home/discord/victrola/src/discord_bot/bot.py:144-158.
MAX_CHUNK = 1900


def _chunk(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split text into <=limit-char chunks, breaking on newlines when possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:  # no good break point — hard split
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


# Lifted verbatim from /home/discord/victrola/src/discord_bot/bot.py:37-69.
# Per-image cap. Discord allows up to 10MB on free; we match that. Very
# large images will be quietly skipped with a log line rather than crash
# the agent call.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def _extract_images(
    message: discord.Message,
) -> list[dict[str, str]]:
    """Pull image attachments off a Discord message as base64 blobs.

    Non-image attachments and oversized images are ignored (logged).
    Returned dicts are shaped for Agent.chat(..., images=...).
    """
    images: list[dict[str, str]] = []
    for att in message.attachments:
        content_type = (att.content_type or "").lower()
        if not content_type.startswith("image/"):
            continue
        if att.size and att.size > MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image %s (%d bytes, max %d)",
                att.filename,
                att.size,
                MAX_IMAGE_BYTES,
            )
            continue
        try:
            data = await att.read()
        except Exception:
            logger.exception("Failed to read attachment %s", att.filename)
            continue
        images.append(
            {
                "media_type": content_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return images


class BotNotReady(RuntimeError):
    """Raised when attempting operations on a bot that hasn't finished handshake."""

    pass


class Bot:
    """Wraps a discord.py Client with the operations the bridge needs.

    State machine: instances start `not connected`. After `await start()` the
    Gateway handshake runs in the background; `is_ready` flips to True when
    the bot has fully connected and resolved the configured channel.
    """

    def __init__(self, token: str, channel_id: int) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required for Phase 3 reply routing
        self._client = discord.Client(intents=intents)
        self._token = token
        self._channel_id = channel_id
        self._channel: discord.TextChannel | None = None
        self._ready = asyncio.Event()
        # discord.py registers event handlers by method name.
        self._client.event(self.on_ready)

    @property
    def channel_id(self) -> int:
        return self._channel_id

    async def on_ready(self) -> None:
        """Called when the bot finishes the gateway handshake."""
        ch = self._client.get_channel(self._channel_id) or await self._client.fetch_channel(
            self._channel_id
        )
        if not isinstance(ch, discord.TextChannel):
            raise RuntimeError(
                f"Configured DISCORD_CHANNEL_ID={self._channel_id} "
                f"is not a TextChannel (got {type(ch).__name__})."
            )
        self._channel = ch
        self._ready.set()
        logger.info("Bot ready as %s, watching #%s", self._client.user, ch.name)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._client.is_closed()

    async def start(self) -> None:
        """Schedules the gateway handshake. Returns once the task is running.
        Caller must `await close()` to shut down cleanly."""
        self._task = asyncio.create_task(self._client.start(self._token))

    async def close(self) -> None:
        await self._client.close()
        if hasattr(self, "_task"):
            with contextlib.suppress(Exception):
                await self._task

    async def post(self, message: str, *, thread_id: int | None = None) -> list[int]:
        """Post `message` to the configured channel (or thread within it).

        Chunks per `_chunk()`. Returns the list of created message IDs.
        Raises `BotNotReady` if the bot isn't connected yet.
        """
        if not self.is_ready or self._channel is None:
            raise BotNotReady("bot not connected to Discord")
        target: discord.abc.Messageable = self._channel
        if thread_id is not None:
            target = await self._client.fetch_channel(thread_id)
        ids: list[int] = []
        for chunk in _chunk(message):
            msg = await target.send(chunk)
            ids.append(msg.id)
        return ids
