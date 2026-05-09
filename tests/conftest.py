"""Shared test fixtures."""

import aiosqlite
import pytest

from bridge.state import init_schema


async def init_db_schema(conn: aiosqlite.Connection) -> None:
    """Initialize database schema. Delegates to bridge.state.init_schema so
    the test schema and production schema can't drift."""
    await init_schema(conn)


@pytest.fixture
async def in_memory_db():
    """Create an in-memory SQLite database with full schema for testing.

    Schema matches production state.open_db to prevent schema drift.
    """
    conn = await aiosqlite.connect(":memory:")
    await init_db_schema(conn)
    yield conn
    await conn.close()
