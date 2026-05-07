"""aiohttp web server for /v1/notify and /v1/health endpoints."""

import asyncio
import json
import logging
import signal
import time

from aiohttp import web

from bridge.bot import Bot, BotNotReady
from bridge.secrets import Secrets
from bridge.threads import ThreadRegistry
from bridge import state

logger = logging.getLogger(__name__)


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

    bot: Bot = request.app["bot"]
    registry: ThreadRegistry = request.app["threads"]
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


async def _handle_health(request: web.Request) -> web.Response:
    """Handle GET /v1/health."""
    bot: Bot = request.app["bot"]
    started_at: float = request.app["started_at"]
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
    app["bot"] = bot
    app["started_at"] = started_at if started_at is not None else time.monotonic()
    app.router.add_post("/v1/notify", _handle_notify)
    app.router.add_get("/v1/health", _handle_health)
    return app


async def serve(secrets: Secrets, *, host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the bridge server with bot integration and signal handling.

    Binds to host:port, starts the Discord bot, and runs until SIGTERM/SIGINT.
    Opens the database for session persistence and instantiates ThreadRegistry.
    """
    bot = Bot(secrets.bot_token, secrets.channel_id)
    conn = await state.open_db()
    registry = ThreadRegistry(bot, conn)
    app = await build_app(bot)
    app["threads"] = registry
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
