"""A tripped respawn breaker stays parked until an explicit reset.

The old cooldown retry reopened unchanged loops and generated repeated
stranded-agent notifications. A tripped task now remains gated until its
status/assignment changes or an operator explicitly resets its tracker.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from roboco.runtime.orchestrator import AgentOrchestrator

_WEDGED_COUNT = 63  # a counter driven far past the trip threshold by a storm


def _new_orchestrator() -> AgentOrchestrator:
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pm_respawn_tracker = {}
    orch._bg_tasks = set()
    cast("Any", orch)._schedule_respawn_persist = lambda *_a, **_k: None
    return orch


def _quiet_audit() -> AsyncMock:
    audit = AsyncMock()
    audit.has_recent_tracing_gap = AsyncMock(return_value=False)
    return audit


@contextlib.contextmanager
def _patches() -> Any:
    with (
        patch(
            "roboco.services.audit.get_audit_service",
            return_value=_quiet_audit(),
        ),
        patch(
            "roboco.services.notification.NotificationService",
            return_value=AsyncMock(),
        ),
    ):
        yield


async def _trip(orch: AgentOrchestrator, slug: str, task: dict[str, Any]) -> None:
    """Drive same-status ticks until the breaker trips (count=threshold+1)."""
    for _ in range(orch._PM_RESPAWN_MAX_UNPRODUCTIVE + 1):
        await orch._pm_respawn_should_gate(slug, task)


@pytest.mark.asyncio
async def test_tripped_breaker_stays_gated_after_cooldown() -> None:
    """An unchanged tripped task remains parked after the old cooldown."""
    orch = _new_orchestrator()
    task_id = str(uuid4())
    task = {"id": task_id, "status": "pending"}
    with _patches():
        await _trip(orch, "pr-reviewer-1", task)
        assert await orch._pm_respawn_should_gate("pr-reviewer-1", task) is True
        key = ("pr-reviewer-1", task_id)
        assert (
            orch._pm_respawn_tracker[key]["count"] > orch._PM_RESPAWN_MAX_UNPRODUCTIVE
        )

        assert await orch._pm_respawn_should_gate("pr-reviewer-1", task) is True
        assert orch._pm_respawn_tracker[key]["count"] > orch._PM_RESPAWN_MAX_UNPRODUCTIVE
        assert orch._pm_respawn_tracker[key]["notified"] is True


@pytest.mark.asyncio
async def test_tripped_breaker_stays_gated_and_quiet() -> None:
    """A tripped task stays gated and its count remains frozen."""
    orch = _new_orchestrator()
    task_id = str(uuid4())
    task = {"id": task_id, "status": "pending"}
    with _patches():
        await _trip(orch, "be-dev-1", task)
        frozen = orch._pm_respawn_tracker[("be-dev-1", task_id)]["count"]
        # Two more ticks while within the cooldown: both gate, count frozen.
        assert await orch._pm_respawn_should_gate("be-dev-1", task) is True
        assert await orch._pm_respawn_should_gate("be-dev-1", task) is True
        assert orch._pm_respawn_tracker[("be-dev-1", task_id)]["count"] == frozen, (
            "count must freeze once tripped, not climb every dispatch tick"
        )


@pytest.mark.asyncio
async def test_status_change_is_the_reset_path() -> None:
    """A real task-state change still resets the parked breaker."""
    orch = _new_orchestrator()
    task_id = str(uuid4())
    task = {"id": task_id, "status": "pending"}
    with _patches():
        await _trip(orch, "be-dev-1", task)
        key = ("be-dev-1", task_id)
        task["status"] = "in_progress"
        assert await orch._pm_respawn_should_gate("be-dev-1", task) is False
        assert orch._pm_respawn_tracker[key]["count"] == 1
        assert orch._pm_respawn_tracker[key]["notified"] is False


@pytest.mark.asyncio
async def test_freshly_restored_row_trips_immediately_not_cooldown_reset() -> None:
    """Durability: a restored tripped row with a fresh (re-stamped) last_check
    still gates on the first tick — the cooldown must not disarm a fresh
    restore. (Restore re-stamps last_check to now in production.)"""
    orch = _new_orchestrator()
    task_id = str(uuid4())
    task = {"id": task_id, "status": "pending"}
    # Simulate a restored row: count past threshold, notified, last_check=now.
    orch._pm_respawn_tracker[("be-pm", task_id)] = {
        "count": _WEDGED_COUNT,
        "last_status": "pending",
        "last_check": datetime.now(UTC),
        "tracing_resets": 0,
        "notified": True,
    }
    with _patches():
        gated = await orch._pm_respawn_should_gate("be-pm", task)
    assert gated is True, "a freshly restored row must trip, not cooldown-reset"
    assert orch._pm_respawn_tracker[("be-pm", task_id)]["count"] == _WEDGED_COUNT
