"""Tests for the discord bot wrapper."""

import pytest

from bridge.bot import MAX_CHUNK, _chunk, _extract_images, Bot, BotNotReady


class Test_chunk:
    """Unit tests for the _chunk() function."""

    def test_single_char(self):
        """Single character returns a single chunk."""
        result = _chunk("a")
        assert result == ["a"]

    def test_under_limit(self):
        """Text under limit returns single chunk."""
        text = "a" * 1899
        result = _chunk(text)
        assert result == [text]
        assert len(result) == 1

    def test_exact_limit(self):
        """Text exactly at limit returns single chunk."""
        text = "a" * MAX_CHUNK
        result = _chunk(text)
        assert result == [text]
        assert len(result) == 1

    def test_over_limit_splits(self):
        """Text over limit is split into multiple chunks."""
        text = "a" * 5000
        result = _chunk(text)
        assert len(result) > 1
        # Verify all chunks are under limit
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK
        # Verify concatenation equals original (except stripped newlines)
        reconstructed = "".join(result)
        assert reconstructed == text

    def test_splits_on_newlines(self):
        """Prefers breaking on newlines over hard split."""
        # Create a string with lots of newlines, exceeding limit
        text = "line1\n" * 1000  # ~6000 chars
        result = _chunk(text)

        # Should have multiple chunks
        assert len(result) > 1

        # Each chunk should be under or at the limit
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK

        # Verify content is preserved (though newlines may be stripped between chunks)
        reconstructed = "".join(result)
        # After stripping, we may have fewer newlines, so just check content
        assert "line1" in reconstructed
        assert len(reconstructed) <= len(text)

    def test_hard_split_no_newline(self):
        """Falls back to hard split when no good newline break exists."""
        # One very long line with no breaks
        text = "a" * 2500
        result = _chunk(text)

        # Should split at MAX_CHUNK boundary
        assert result[0] == "a" * MAX_CHUNK
        assert result[1] == "a" * (2500 - MAX_CHUNK)

        # Verify reconstruction
        reconstructed = "".join(result)
        assert reconstructed == text

    def test_hard_split_poor_newline(self):
        """Hard splits if only newlines in lower half of chunk."""
        # Force hard split by putting newline only in lower half
        text = "a" * (MAX_CHUNK // 4) + "\n" + "b" * (MAX_CHUNK + 100)
        result = _chunk(text)

        # First chunk should be hard-split at limit
        assert len(result[0]) == MAX_CHUNK

        # Verify reconstruction
        reconstructed = "".join(result)
        assert reconstructed == text

    def test_strips_leading_newlines(self):
        """Strips leading newlines when continuing after a chunk."""
        text = "first part" + "\n" * 10 + ("x" * 1900)
        result = _chunk(text)

        # Should have 2 chunks
        assert len(result) == 2

        # Second chunk should not have leading newlines (key behavior of lstrip)
        assert not result[1].startswith("\n")

        # All chunks except the final should be at or under the limit
        for chunk in result[:-1]:
            assert len(chunk) <= MAX_CHUNK

        # Final chunk should be under the limit
        assert len(result[-1]) <= MAX_CHUNK

        # Verify reconstruction preserves content
        reconstructed = "".join(result)
        assert "first part" in reconstructed
        assert "x" in reconstructed

    def test_custom_limit(self):
        """Custom limit parameter is respected."""
        text = "a" * 500
        result = _chunk(text, limit=100)

        # Should be split at custom limit
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 100

    def test_empty_string(self):
        """Empty string returns single empty chunk."""
        result = _chunk("")
        assert result == [""]


class TestBot:
    """Unit tests for the Bot class."""

    def test_bot_not_ready_exception(self):
        """BotNotReady is a RuntimeError subclass."""
        exc = BotNotReady("test")
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "test"

    def test_bot_init(self):
        """Bot initializes with token and channel_id."""
        bot = Bot("test_token", 12345)
        assert bot.channel_id == 12345

    def test_bot_not_ready_initially(self):
        """Bot is not ready immediately after creation."""
        bot = Bot("test_token", 12345)
        assert not bot.is_ready

    @pytest.mark.asyncio
    async def test_bot_post_not_ready_raises(self):
        """Bot.post() raises BotNotReady if bot is not connected."""
        bot = Bot("test_token", 12345)
        with pytest.raises(BotNotReady, match="not connected"):
            await bot.post("test message")

    @pytest.mark.asyncio
    async def test_bot_close_without_start(self):
        """Bot.close() works even if start() was never called."""
        bot = Bot("test_token", 12345)
        # Should not raise
        await bot.close()

    @pytest.mark.asyncio
    async def test_bot_create_thread_not_ready_raises(self):
        """Bot.create_thread() raises BotNotReady if bot is not connected."""
        bot = Bot("test_token", 12345)
        with pytest.raises(BotNotReady, match="not connected"):
            await bot.create_thread("test thread")
