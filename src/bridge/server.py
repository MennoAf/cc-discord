"""aiohttp web server for /v1/notify, /v1/health, and /v1/ask endpoints."""

import asyncio
import json
import logging
import signal
import time
from datetime import datetime, timezone

from aiohttp import web

from bridge.bot import Bot, BotNotReady
from bridge.listener import Listener, _PendingAsk
from bridge.secrets import Secrets
from bridge.threads import ThreadRegistry
from bridge import state

logger = logging.getLogger(__name__)


def _clamp_timeout(secs: float) -> float:
    """Clamp timeout value to valid range [5, 3600] seconds.

    Args:
        secs: timeout in seconds

    Returns:
        clamped timeout in seconds

    Raises:
        ValueError: if secs cannot be converted to float
    """
    timeout = float(secs)
    return max(5.0, min(3600.0, timeout))


def _format_question(question: str, cwd: str) -> str:
    """Format a question with header and working directory.

    Args:
        question: the user's question text
        cwd: the working directory (can be empty)

    Returns:
        formatted question string
    """
    if cwd:
        return f"❓ asks\n{question}\n\n(cwd: {cwd})"
    return f"❓ asks\n{question}"


class AskLockMap:
    """Per-thread asyncio.Lock factory for FIFO serialization of /v1/ask calls."""

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def get(self, thread_id: int) -> asyncio.Lock:
        """Get or create a lock for a thread_id.

        Args:
            thread_id: Discord thread ID

        Returns:
            asyncio.Lock for the thread
        """
        async with self._guard:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[thread_id] = lock
            return lock


# Typed AppKey definitions to avoid NotAppKeyWarning
BOT_KEY: web.AppKey[Bot] = web.AppKey("bot", Bot)
THREADS_KEY: web.AppKey[ThreadRegistry] = web.AppKey("threads", ThreadRegistry)
LISTENER_KEY: web.AppKey[Listener] = web.AppKey("listener", Listener)
ASK_LOCKS_KEY: web.AppKey[AskLockMap] = web.AppKey("ask_locks", AskLockMap)

STARTED_AT_KEY: web.AppKey[float] = web.AppKey("started_at", float)


async def _handle_notify(request: web.Request) -> web.Response:
    """Handle POST /v1/notify.

    Body (JSON): { "session_id": str, "cwd": str, "message": str, "title"?: str, "level"?: str }
    """
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return web.Response(
            status=400,
            text=json.dumps({"error": "invalid json"}),
            content_type="application/json",
        )

    # Validate required fields
    required = ["session_id", "cwd", "message"]
    if not all(key in body for key in required):
        missing = [key for key in required if key not in body]
        return web.Response(
            status=400,
            text=json.dumps({"error": f"missing required fields: {', '.join(missing)}"}),
            content_type="application/json",
        )

    bot: Bot = request.app[BOT_KEY]
    registry: ThreadRegistry = request.app[THREADS_KEY]
    message = body["message"]

    # Check if bot is ready before routing to thread
    if not bot.is_ready:
        return web.Response(
            status=503,
            text=json.dumps({"error": "bot_not_connected"}),
            content_type="application/json",
        )

    try:
        thread_id = await registry.get_or_create_thread(body["session_id"], body["cwd"])
        message_ids = await bot.post(message, thread_id=thread_id)
        return web.Response(
            status=200,
            text=json.dumps(
                {"thread_id": thread_id, "message_id": message_ids[0] if message_ids else None}
            ),
            content_type="application/json",
        )
    except BotNotReady:
        return web.Response(
            status=503,
            text=json.dumps({"error": "bot_not_connected"}),
            content_type="application/json",
        )
    except Exception:
        logger.exception("notify failed")
        return web.Response(
            status=500,
            text=json.dumps({"error": "internal"}),
            content_type="application/json",
        )


async def _handle_ask(request: web.Request) -> web.Response:
    """Handle POST /v1/ask.

    Body (JSON): {
        "session_id": str,
        "cwd": str,
        "question": str,
        "timeout_secs"?: number (default 900, clamped to [5, 3600])
    }

    Response on success (200):
        { "reply": str, "replied_at": str (ISO8601) }

    Response on timeout (408):
        { "error": "timeout" }

    Response on bot not connected (503):
        { "error": "bot_not_connected" }
    """
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return web.Response(
            status=400,
            text=json.dumps({"error": "invalid json"}),
            content_type="application/json",
        )

    # Validate required fields
    required = ["session_id", "cwd", "question"]
    if not all(key in body for key in required):
        missing = [key for key in required if key not in body]
        return web.Response(
            status=400,
            text=json.dumps({"error": f"missing required fields: {', '.join(missing)}"}),
            content_type="application/json",
        )

    # Validate timeout_secs early if provided
    if "timeout_secs" in body:
        try:
            _clamp_timeout(body["timeout_secs"])
        except ValueError:
            return web.Response(
                status=400,
                text=json.dumps({"error": "invalid timeout_secs"}),
                content_type="application/json",
            )

    try:
        bot: Bot = request.app[BOT_KEY]
        registry: ThreadRegistry = request.app[THREADS_KEY]
        listener: Listener = request.app[LISTENER_KEY]
        locks: AskLockMap = request.app[ASK_LOCKS_KEY]

        # Check if bot is ready
        if not bot.is_ready:
            return web.Response(
                status=503,
                text=json.dumps({"error": "bot_not_connected"}),
                content_type="application/json",
            )

        try:
            thread_id = await registry.get_or_create_thread(body["session_id"], body["cwd"])
        except BotNotReady:
            return web.Response(
                status=503,
                text=json.dumps({"error": "bot_not_connected"}),
                content_type="application/json",
            )

        # Acquire per-thread lock for FIFO serialization
        lock = await locks.get(thread_id)
        async with lock:
            # Post the question AFTER acquiring the lock
            ask_text = _format_question(body["question"], body["cwd"])
            try:
                await bot.post(ask_text, thread_id=thread_id)
            except BotNotReady:
                return web.Response(
                    status=503,
                    text=json.dumps({"error": "bot_not_connected"}),
                    content_type="application/json",
                )

            # Register pending ask and wait for reply
            asked_at = datetime.now(timezone.utc)
            ask = _PendingAsk(asked_at)
            await listener.register(thread_id, ask)
            try:
                # Parse and clamp timeout
                timeout = _clamp_timeout(body.get("timeout_secs", 900))
                result = await asyncio.wait_for(ask.future, timeout=timeout)
            except asyncio.TimeoutError:
                return web.Response(
                    status=408,
                    text=json.dumps({"error": "timeout"}),
                    content_type="application/json",
                )
            finally:
                await listener.unregister(thread_id, ask)

        return web.Response(
            status=200,
            text=json.dumps(
                {"reply": result.reply, "replied_at": result.replied_at},
            ),
            content_type="application/json",
        )
    except BotNotReady:
        return web.Response(
            status=503,
            text=json.dumps({"error": "bot_not_connected"}),
            content_type="application/json",
        )
    except asyncio.TimeoutError:
        return web.Response(
            status=408,
            text=json.dumps({"error": "timeout"}),
            content_type="application/json",
        )
    except Exception:
        logger.exception("ask failed")
        return web.Response(
            status=500,
            text=json.dumps({"error": "internal"}),
            content_type="application/json",
        )


async def _handle_health(request: web.Request) -> web.Response:
    """Handle GET /v1/health."""
    bot: Bot = request.app[BOT_KEY]
    started_at: float = request.app[STARTED_AT_KEY]
    uptime_secs = int(time.monotonic() - started_at)

    response = {
        "bot_connected": bot.is_ready,
        "channel_id": bot.channel_id,
        "uptime_secs": uptime_secs,
    }
    return web.Response(
        status=200,
        text=json.dumps(response),
        content_type="application/json",
    )


async def build_app(bot: Bot, *, started_at: float | None = None) -> web.Application:
    """Build and configure the aiohttp Application."""
    app = web.Application()
    app[BOT_KEY] = bot
    app[STARTED_AT_KEY] = started_at if started_at is not None else time.monotonic()
    app.router.add_post("/v1/notify", _handle_notify)
    app.router.add_post("/v1/ask", _handle_ask)
    app.router.add_get("/v1/health", _handle_health)
    return app


async def serve(secrets: Secrets, *, host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the bridge server with bot integration and signal handling.

    Binds to host:port, starts the Discord bot, and runs until SIGTERM/SIGINT.
    Opens the database for session persistence and instantiates ThreadRegistry.
    """
    listener = Listener()
    bot = Bot(secrets.bot_token, secrets.channel_id, on_message=listener.deliver)
    conn = await state.open_db()
    registry = ThreadRegistry(bot, conn)
    app = await build_app(bot)
    app[THREADS_KEY] = registry
    app[LISTENER_KEY] = listener
    app[ASK_LOCKS_KEY] = AskLockMap()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    await bot.start()
    logger.info("listening on http://%s:%d", host, port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        await bot.close()
        await runner.cleanup()
        await state.close_db(conn)
