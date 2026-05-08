from pathlib import Path

import aiosqlite
import pytest

import time

from bridge.state import open_db, close_db, SessionRow, get_session, upsert_session, delete_session


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


@pytest.mark.asyncio
class TestSessionRow:
    """SessionRow dataclass."""

    async def test_session_row_creation(self) -> None:
        """SessionRow can be instantiated with all fields."""
        row = SessionRow(
            session_id="sess-123",
            cwd="/tmp/test",
            thread_id=999,
            created_at=1000,
            last_activity=2000,
        )
        assert row.session_id == "sess-123"
        assert row.cwd == "/tmp/test"
        assert row.thread_id == 999
        assert row.created_at == 1000
        assert row.last_activity == 2000

    async def test_session_row_is_frozen(self) -> None:
        """SessionRow is immutable (frozen)."""
        row = SessionRow(
            session_id="sess-123",
            cwd="/tmp/test",
            thread_id=999,
            created_at=1000,
            last_activity=2000,
        )
        with pytest.raises(AttributeError):
            row.thread_id = 123


@pytest.mark.asyncio
class TestGetSession:
    """get_session retrieves a session row by session_id."""

    async def test_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        """get_session returns None when session_id doesn't exist."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            result = await get_session(conn, "unknown")
            assert result is None
        finally:
            await conn.close()

    async def test_retrieves_inserted_session(self, tmp_path: Path) -> None:
        """get_session returns the inserted row with all fields intact."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            # Insert a session
            now = int(time.time())
            await conn.execute(
                "INSERT INTO sessions (session_id, cwd, thread_id, created_at, last_activity) VALUES (?, ?, ?, ?, ?)",
                ("sess-123", "/tmp/test", 999, now, now),
            )
            await conn.commit()

            # Retrieve it
            row = await get_session(conn, "sess-123")
            assert row is not None
            assert row.session_id == "sess-123"
            assert row.cwd == "/tmp/test"
            assert row.thread_id == 999
            assert row.created_at == now
            assert row.last_activity == now
        finally:
            await conn.close()


@pytest.mark.asyncio
class TestUpsertSession:
    """upsert_session inserts or updates a session row."""

    async def test_inserts_new_session(self, tmp_path: Path) -> None:
        """upsert_session inserts a new session when it doesn't exist."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            now = int(time.time())
            await upsert_session(conn, "sess-123", "/tmp/test", 999, now=now)

            row = await get_session(conn, "sess-123")
            assert row is not None
            assert row.session_id == "sess-123"
            assert row.cwd == "/tmp/test"
            assert row.thread_id == 999
            assert row.created_at == now
            assert row.last_activity == now
        finally:
            await conn.close()

    async def test_upsert_bumps_last_activity(self, tmp_path: Path) -> None:
        """upsert_session updates thread_id and bumps last_activity, preserves created_at."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            t0 = int(time.time())
            await upsert_session(conn, "sess-123", "/tmp/test", 999, now=t0)

            row1 = await get_session(conn, "sess-123")
            assert row1 is not None
            assert row1.created_at == t0
            assert row1.last_activity == t0
            assert row1.thread_id == 999

            # Re-upsert with different thread_id and later timestamp
            t1 = t0 + 100
            await upsert_session(conn, "sess-123", "/tmp/test", 888, now=t1)

            row2 = await get_session(conn, "sess-123")
            assert row2 is not None
            assert row2.session_id == "sess-123"
            assert row2.cwd == "/tmp/test"
            assert row2.thread_id == 888  # Updated
            assert row2.created_at == t0  # Preserved
            assert row2.last_activity == t1  # Bumped
        finally:
            await conn.close()

    async def test_upsert_uses_current_time_if_now_not_provided(self, tmp_path: Path) -> None:
        """upsert_session defaults to current time if now is None."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            before = int(time.time())
            await upsert_session(conn, "sess-123", "/tmp/test", 999)
            after = int(time.time())

            row = await get_session(conn, "sess-123")
            assert row is not None
            # last_activity should be between before and after
            assert before <= row.last_activity <= after + 1
        finally:
            await conn.close()


@pytest.mark.asyncio
class TestDeleteSession:
    """delete_session removes a session row."""

    async def test_deletes_existing_session(self, tmp_path: Path) -> None:
        """delete_session removes the row; subsequent get_session returns None."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            now = int(time.time())
            await upsert_session(conn, "sess-123", "/tmp/test", 999, now=now)

            row = await get_session(conn, "sess-123")
            assert row is not None

            await delete_session(conn, "sess-123")

            row = await get_session(conn, "sess-123")
            assert row is None
        finally:
            await conn.close()

    async def test_delete_nonexistent_is_safe(self, tmp_path: Path) -> None:
        """delete_session on unknown session_id doesn't raise."""
        db_path = tmp_path / "state.db"
        conn = await open_db(db_path)
        try:
            # Should not raise
            await delete_session(conn, "unknown")
        finally:
            await conn.close()


@pytest.mark.asyncio
class TestSessionPersistence:
    """Session mapping survives database restart (AC2.3)."""

    async def test_persistence_across_restart(self, tmp_path: Path) -> None:
        """Open DB, upsert, close; reopen the same path, get_session still returns the row."""
        db_path = tmp_path / "state.db"
        now = int(time.time())

        # First connection: insert
        conn1 = await open_db(db_path)
        try:
            await upsert_session(conn1, "sess-123", "/tmp/test", 999, now=now)
        finally:
            await conn1.close()

        # Second connection: retrieve
        conn2 = await open_db(db_path)
        try:
            row = await get_session(conn2, "sess-123")
            assert row is not None
            assert row.session_id == "sess-123"
            assert row.cwd == "/tmp/test"
            assert row.thread_id == 999
            assert row.created_at == now
            assert row.last_activity == now
        finally:
            await conn2.close()
