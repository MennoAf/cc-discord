from pathlib import Path

import aiosqlite

DEFAULT_DB_PATH = Path.home() / ".local" / "state" / "claude-discord-bridge" / "state.db"


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
