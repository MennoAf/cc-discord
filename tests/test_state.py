from pathlib import Path

import aiosqlite
import pytest

from bridge.state import open_db, close_db, DEFAULT_DB_PATH


@pytest.mark.asyncio
class TestOpenDB:
    """open_db creates and initializes the database."""

    async def test_opens_connection(self, tmp_path: Path) -> None:
        """open_db returns a usable aiosqlite connection."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            # Verify it's a valid connection by executing a query
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result == (1,)
        finally:
            await conn.close()

    async def test_creates_parent_dir(self, tmp_path: Path) -> None:
        """open_db creates parent directory if missing."""
        db_path = tmp_path / "subdir1" / "subdir2" / "state.db"
        conn = await open_db(db_path)
        try:
            assert db_path.exists()
        finally:
            await conn.close()

    async def test_sets_wal_mode(self, tmp_path: Path) -> None:
        """After open, PRAGMA journal_mode reports wal."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            cursor = await conn.execute("PRAGMA journal_mode")
            mode = await cursor.fetchone()
            assert mode[0].lower() == "wal"
        finally:
            await conn.close()

    async def test_creates_sessions_table(self, tmp_path: Path) -> None:
        """Schema verification: sessions table exists."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "sessions"
        finally:
            await conn.close()

    async def test_idempotent_multiple_opens(self, tmp_path: Path) -> None:
        """Calling open_db twice on the same path is idempotent."""
        db_path = tmp_path / "state.db"
        conn1 = await open_db(db_path)
        conn2 = await open_db(db_path)
        try:
            # Both should work and be valid
            cursor1 = await conn1.execute("SELECT 1")
            result1 = await cursor1.fetchone()
            assert result1 == (1,)

            cursor2 = await conn2.execute("SELECT 1")
            result2 = await cursor2.fetchone()
            assert result2 == (1,)
        finally:
            await conn1.close()
            await conn2.close()

    async def test_sessions_table_schema(self, tmp_path: Path) -> None:
        """Sessions table has the correct columns."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            cursor = await conn.execute("PRAGMA table_info(sessions)")
            columns = await cursor.fetchall()
            col_names = {col[1] for col in columns}
            assert col_names == {
                "session_id",
                "cwd",
                "thread_id",
                "created_at",
                "last_activity",
            }
        finally:
            await conn.close()


@pytest.mark.asyncio
class TestCloseDB:
    """close_db properly closes the connection."""

    async def test_closes_connection(self, tmp_path: Path) -> None:
        """close_db actually closes (subsequent query raises)."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        await close_db(conn)

        # After closing, queries should fail
        with pytest.raises((aiosqlite.ProgrammingError, ValueError)):
            await conn.execute("SELECT 1")
