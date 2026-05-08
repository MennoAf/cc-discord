"""Shared test fixtures."""

import aiosqlite
import pytest


async def init_db_schema(conn: aiosqlite.Connection) -> None:
    """Initialize database schema (same as state.open_db does)."""
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            thread_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            thread_id INTEGER NOT NULL,
            zellij_pane_id TEXT,
            cwd TEXT NOT NULL,
            status TEXT NOT NULL,
            current_claude_session_id TEXT,
            current_transcript_path TEXT,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_thread_id ON tasks(thread_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(current_claude_session_id)"
    )
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_log (
            request_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            tool_input_json TEXT NOT NULL,
            decision TEXT NOT NULL,
            decision_reason TEXT NOT NULL,
            decided_at INTEGER NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_approval_log_task_id ON approval_log(task_id)"
    )
    await conn.commit()


@pytest.fixture
async def in_memory_db():
    """Create an in-memory SQLite database with full schema for testing.

    Schema matches production state.open_db to prevent schema drift.
    """
    conn = await aiosqlite.connect(":memory:")
    await init_db_schema(conn)
    yield conn
    await conn.close()
