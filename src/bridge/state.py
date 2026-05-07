import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

DEFAULT_DB_PATH = Path.home() / ".local" / "state" / "claude-discord-bridge" / "state.db"


@dataclass(frozen=True)
class SessionRow:
    """A session row from the database."""

    session_id: str
    cwd: str
    thread_id: int
    created_at: int
    last_activity: int


async def open_db(path: Path = DEFAULT_DB_PATH) -> aiosqlite.Connection:
    """Open SQLite database, initializing schema if needed.

    Creates parent directories if missing. Sets WAL mode for concurrent access.
    Returns a ready-to-use aiosqlite.Connection.
    """
    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open connection
    conn = await aiosqlite.connect(path)

    # Set WAL mode
    await conn.execute("PRAGMA journal_mode=WAL")

    # Initialize schema if it doesn't exist
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            thread_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL
        )
    """)
    await conn.commit()

    return conn


async def close_db(conn: aiosqlite.Connection) -> None:
    """Close the database connection."""
    await conn.close()


async def get_session(conn: aiosqlite.Connection, session_id: str) -> SessionRow | None:
    """Retrieve a session row by session_id. Returns None if not found."""
    cursor = await conn.execute(
        "SELECT session_id, cwd, thread_id, created_at, last_activity FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return SessionRow(
        session_id=row[0],
        cwd=row[1],
        thread_id=row[2],
        created_at=row[3],
        last_activity=row[4],
    )


async def upsert_session(
    conn: aiosqlite.Connection,
    session_id: str,
    cwd: str,
    thread_id: int,
    *,
    now: int | None = None,
) -> None:
    """Insert or update a session. Bumps last_activity; preserves created_at on conflict."""
    now_val = now or int(time.time())
    await conn.execute(
        """
        INSERT INTO sessions (session_id, cwd, thread_id, created_at, last_activity)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            thread_id=excluded.thread_id,
            last_activity=excluded.last_activity
        """,
        (session_id, cwd, thread_id, now_val, now_val),
    )
    await conn.commit()


async def delete_session(conn: aiosqlite.Connection, session_id: str) -> None:
    """Delete a session row by session_id."""
    await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    await conn.commit()
