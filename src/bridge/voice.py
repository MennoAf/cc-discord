"""Voice-memo transcription via Wispr Flow API.

Wispr Flow's `POST /transcribe` accepts only base64-encoded 16kHz mono PCM WAV
(<= 25MB, <= 6 minutes), so we convert the original Discord upload through
ffmpeg before posting. Auth: bearer JWT in `WISPR_FLOW_API_TOKEN`.

Docs: https://api-docs.wisprflow.ai/client_api_transcribe
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


WISPR_TRANSCRIBE_URL = "https://api.wisprflow.ai/transcribe"

_AUDIO_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".ogg",
    ".oga",
    ".wav",
    ".webm",
    ".flac",
    ".opus",
    ".aac",
    ".mp4",  # voice memos sometimes ship as .mp4 audio-only
}


def is_audio_path(path: Path) -> bool:
    """Heuristic: does this filename look like audio? Used to route attachments
    to the transcription path."""
    return path.suffix.lower() in _AUDIO_SUFFIXES


async def transcribe(audio_path: Path, *, timeout: float = 60.0) -> str | None:
    """Transcribe an audio file via Wispr Flow.

    Returns the transcription string on success, or None on:
      - missing `WISPR_FLOW_API_TOKEN`
      - ffmpeg conversion failure (binary missing or unreadable input)
      - HTTP error from Wispr (non-2xx, malformed JSON, missing `text`)
      - timeout
    Failures are logged; callers should treat None as "no transcript".
    """
    token = os.environ.get("WISPR_FLOW_API_TOKEN")
    if not token:
        logger.warning(
            "voice attachment received but WISPR_FLOW_API_TOKEN is not set"
        )
        return None

    wav_path = await _convert_to_pcm_wav(audio_path)
    if wav_path is None:
        return None
    try:
        audio_bytes = wav_path.read_bytes()
    except OSError:
        logger.exception("failed to read converted wav %s", wav_path)
        return None
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {"audio": audio_b64}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                WISPR_TRANSCRIBE_URL,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status >= 400:
                    detail = (await resp.text())[:500]
                    logger.warning(
                        "Wispr transcribe HTTP %d: %s", resp.status, detail
                    )
                    return None
                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    logger.exception("Wispr transcribe returned non-JSON body")
                    return None
    except asyncio.TimeoutError:
        logger.warning("Wispr transcribe timed out after %ss", timeout)
        return None
    except aiohttp.ClientError:
        logger.exception("Wispr transcribe HTTP client error")
        return None

    text = data.get("text") if isinstance(data, dict) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    logger.warning("Wispr transcribe returned no `text`; payload=%r", data)
    return None


async def _convert_to_pcm_wav(src: Path) -> Path | None:
    """Run ffmpeg to produce a 16kHz mono PCM WAV alongside `src`. Returns the
    wav path or None on failure (binary missing, conversion error)."""
    out = src.with_name(src.stem + ".wispr.wav")
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(out),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not installed; cannot convert %s", src)
        return None
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg conversion failed (rc=%d): %s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return None
    return out
