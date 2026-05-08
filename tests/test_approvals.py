"""Tests for ApprovalRouter in src/bridge/approvals.py."""

from __future__ import annotations

import asyncio

import pytest

from bridge import state
from tests.fakes import FakeBot


@pytest.mark.asyncio
async def test_approval_router_allow_via_reaction(tmp_path):
    """request_permission returns ("allow", "approved via reaction") after ✅ reaction."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    # Create a test task first
    await state.upsert_task(
        conn, "task-1", 1001, "/tmp", "running"
    )

    router = ApprovalRouter(bot, conn, timeout=10.0)

    async def trigger_reaction():
        """Simulate user clicking ✅ after a short delay."""
        await asyncio.sleep(0.1)
        # Find the message_id that was posted
        assert len(bot.get_post_calls()) > 0
        # For now we'll manually trigger the reaction by message_id
        # The message_id gets set after post, so we need to retrieve it from the router state
        # We'll resolve it manually by accessing the router's internal state
        pending_list = list(router._by_request_id.values())
        if pending_list:
            pending = pending_list[0]
            if pending.message_id:
                await router.resolve_by_reaction(pending.message_id, "✅", False)

    task = asyncio.create_task(trigger_reaction())

    decision, reason = await router.request_permission(
        request_id="req-1",
        task_id="task-1",
        thread_id=1001,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    await task
    assert decision == "allow"
    assert reason == "approved via reaction"
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_deny_via_reaction(tmp_path):
    """request_permission returns ("deny", "denied via reaction") after ❌ reaction."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-2", 1002, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    async def trigger_reaction():
        await asyncio.sleep(0.1)
        pending_list = list(router._by_request_id.values())
        if pending_list:
            pending = pending_list[0]
            if pending.message_id:
                await router.resolve_by_reaction(pending.message_id, "❌", False)

    task = asyncio.create_task(trigger_reaction())

    decision, reason = await router.request_permission(
        request_id="req-2",
        task_id="task-2",
        thread_id=1002,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    await task
    assert decision == "deny"
    assert reason == "denied via reaction"
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_deny_via_text(tmp_path):
    """request_permission returns ("deny", text) after resolve_by_text with reason."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-3", 1003, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    async def trigger_text():
        await asyncio.sleep(0.1)
        await router.resolve_by_text(1003, "use a different approach", False)

    task = asyncio.create_task(trigger_text())

    decision, reason = await router.request_permission(
        request_id="req-3",
        task_id="task-3",
        thread_id=1003,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    await task
    assert decision == "deny"
    assert reason == "use a different approach"
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_timeout(tmp_path):
    """request_permission returns ("deny", "approval timed out") after timeout."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-4", 1004, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=0.1)

    decision, reason = await router.request_permission(
        request_id="req-4",
        task_id="task-4",
        thread_id=1004,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    assert decision == "deny"
    assert reason == "approval timed out"
    # Verify that a timeout notice was posted
    assert any("Denied (timeout)" in call["content"] for call in bot.get_post_calls())
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_logs_decision(tmp_path):
    """request_permission logs the decision to approval_log table."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-5", 1005, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    async def trigger_reaction():
        await asyncio.sleep(0.1)
        pending_list = list(router._by_request_id.values())
        if pending_list:
            pending = pending_list[0]
            if pending.message_id:
                await router.resolve_by_reaction(pending.message_id, "✅", False)

    task = asyncio.create_task(trigger_reaction())

    await router.request_permission(
        request_id="req-5",
        task_id="task-5",
        thread_id=1005,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    await task

    # Verify approval was logged
    approvals = await state.list_approvals_for_task(conn, "task-5")
    assert len(approvals) == 1
    assert approvals[0].request_id == "req-5"
    assert approvals[0].decision == "allow"
    assert approvals[0].tool_name == "Bash"
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_filters_bot_reactions(tmp_path):
    """resolve_by_reaction returns False and doesn't resolve if user_is_bot=True."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-6", 1006, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    async def trigger_reactions():
        await asyncio.sleep(0.05)
        pending_list = list(router._by_request_id.values())
        if pending_list:
            pending = pending_list[0]
            if pending.message_id:
                # First, bot reacts (should be ignored)
                result1 = await router.resolve_by_reaction(pending.message_id, "✅", True)
                assert result1 is False
                # Then, user reacts (should resolve)
                result2 = await router.resolve_by_reaction(pending.message_id, "✅", False)
                assert result2 is True

    task = asyncio.create_task(trigger_reactions())

    decision, reason = await router.request_permission(
        request_id="req-6",
        task_id="task-6",
        thread_id=1006,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    await task
    assert decision == "allow"
    await state.close_db(conn)


@pytest.mark.asyncio
async def test_approval_router_concurrent_approvals(tmp_path):
    """Two concurrent approvals to the same thread complete independently."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-7", 1007, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    results = []

    async def first_approval():
        # First request will timeout or wait
        decision, reason = await router.request_permission(
            request_id="req-7a",
            task_id="task-7",
            thread_id=1007,
            tool_name="Bash",
            tool_input={"cmd": "ls"},
        )
        results.append(("first", decision, reason))

    async def second_approval():
        await asyncio.sleep(0.2)
        # This comes after first is posted
        decision, reason = await router.request_permission(
            request_id="req-7b",
            task_id="task-7",
            thread_id=1007,
            tool_name="Bash",
            tool_input={"cmd": "pwd"},
        )
        results.append(("second", decision, reason))

    async def trigger_reactions():
        await asyncio.sleep(0.05)
        # Resolve first one with ✅
        pending_list = list(router._by_request_id.values())
        if len(pending_list) >= 1:
            pending = [p for p in pending_list if p.request_id == "req-7a"][0]
            if pending.message_id:
                await router.resolve_by_reaction(pending.message_id, "✅", False)

        await asyncio.sleep(0.3)
        # Resolve second one with ❌
        pending_list = list(router._by_request_id.values())
        if len(pending_list) >= 1:
            pending = [p for p in pending_list if p.request_id == "req-7b"][0]
            if pending.message_id:
                await router.resolve_by_reaction(pending.message_id, "❌", False)

    reactions_task = asyncio.create_task(trigger_reactions())

    await asyncio.gather(first_approval(), second_approval())
    await reactions_task

    # Both should complete independently
    assert len(results) == 2
    # Find each result
    first_result = next((r for r in results if r[0] == "first"), None)
    second_result = next((r for r in results if r[0] == "second"), None)

    assert first_result is not None
    assert first_result[1] == "allow"  # decision
    assert second_result is not None
    assert second_result[1] == "deny"  # decision

    await state.close_db(conn)


@pytest.mark.asyncio
async def test_resolve_by_text_returns_false_on_empty_input(tmp_path):
    """resolve_by_text returns False and doesn't resolve pending approval on empty/whitespace input."""
    from bridge.approvals import ApprovalRouter

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    await state.upsert_task(conn, "task-8", 1008, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=0.5)

    async def spawn_request():
        """Spawn the pending request and keep it alive briefly."""
        # Start the request in a task so it doesn't block
        task = asyncio.create_task(
            router.request_permission(
                request_id="req-8",
                task_id="task-8",
                thread_id=1008,
                tool_name="Bash",
                tool_input={"cmd": "ls"},
            )
        )
        # Give it time to post
        await asyncio.sleep(0.05)
        return task

    request_task = await spawn_request()

    # Test empty string
    result = await router.resolve_by_text(1008, "", author_is_bot=False)
    assert result is False

    # Test whitespace-only string
    result = await router.resolve_by_text(1008, "   \n   ", author_is_bot=False)
    assert result is False

    # Verify the pending approval is still there (not resolved)
    pending_list = list(router._by_request_id.values())
    assert len(pending_list) == 1
    assert not pending_list[0].future.done()

    # Clean up: cancel the request task
    request_task.cancel()
    try:
        await request_task
    except asyncio.CancelledError:
        pass

    await state.close_db(conn)


@pytest.mark.asyncio
async def test_request_permission_add_reactions_failure(tmp_path):
    """request_permission returns ('deny', 'failed to add approval reactions...') when add_reactions raises."""
    from bridge.approvals import ApprovalRouter
    from tests.fakes import FakeBot

    db_path = tmp_path / "test.db"
    conn = await state.open_db(db_path)
    bot = FakeBot()

    # Extend FakeBot to raise on add_reactions
    async def failing_add_reactions(*args: any, **kwargs: any) -> None:
        raise RuntimeError("Bot permissions missing for add_reactions")

    bot.add_reactions = failing_add_reactions

    await state.upsert_task(conn, "task-9", 1009, "/tmp", "running")

    router = ApprovalRouter(bot, conn, timeout=10.0)

    decision, reason = await router.request_permission(
        request_id="req-9",
        task_id="task-9",
        thread_id=1009,
        tool_name="Bash",
        tool_input={"cmd": "ls"},
    )

    # Verify it returns deny with the expected reason
    assert decision == "deny"
    assert "failed to add approval reactions" in reason

    # Verify request_id is no longer in the router's request dict
    assert "req-9" not in router._by_request_id

    # Verify message_id is no longer in the router's message dict
    # (Since the add_reactions failed, we can't reliably check what message_id was posted,
    # but we can verify the dict is cleaned up)
    assert len(router._by_message_id) == 0

    await state.close_db(conn)
