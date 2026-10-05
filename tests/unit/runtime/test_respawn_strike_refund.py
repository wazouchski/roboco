"""A spawn skipped by the provider gate must not burn a respawn-breaker strike.

The breaker takes its strike before ``spawn_agent`` runs. With a one-slot local
cap every queued task was skipped each tick, so each one tripped the breaker
within ~3 ticks and was parked (live 2026-09-30). ``spawn_agent`` now refunds the
strike when it bails without launching a container.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from roboco.runtime.orchestrator import AgentOrchestrator

_TICKS = 25  # far more ticks than the trip threshold
_TRIPPED_COUNT = 4


def _new_orchestrator() -> AgentOrchestrator:
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pm_respawn_tracker = {}
    orch._bg_tasks = set()
    cast("Any", orch)._schedule_respawn_persist = lambda *_a, **_k: None
    return orch


@contextlib.contextmanager
def _patches() -> Any:
    audit = AsyncMock()
    audit.has_recent_tracing_gap = AsyncMock(return_value=False)
    with (
        patch("roboco.services.audit.get_audit_service", return_value=audit),
        patch(
            "roboco.services.notification.NotificationService",
            return_value=AsyncMock(),
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_cap_skipped_ticks_never_trip_the_breaker() -> None:
    orch = _new_orchestrator()
    task = {"id": str(uuid4()), "status": "pending"}
    with _patches():
        for _ in range(_TICKS):
            assert await orch._pm_respawn_should_gate("fe-dev-1", task) is False
            orch._refund_respawn_strike("fe-dev-1", task["id"])
    record = orch._pm_respawn_tracker[("fe-dev-1", task["id"])]
    assert record["count"] == 1
    assert not record.get("notified")


@pytest.mark.asyncio
async def test_unrefunded_ticks_still_trip_the_breaker() -> None:
    """Real (launched) respawns that don't advance still trip it."""
    orch = _new_orchestrator()
    task = {"id": str(uuid4()), "status": "pending"}
    with _patches():
        results = [
            await orch._pm_respawn_should_gate("fe-dev-1", task) for _ in range(_TICKS)
        ]
    assert results[-1] is True


def test_refund_leaves_a_tripped_record_alone() -> None:
    orch = _new_orchestrator()
    key = ("be-pm", "t1")
    orch._pm_respawn_tracker[key] = {"count": 4, "notified": True}
    orch._refund_respawn_strike("be-pm", "t1")
    assert orch._pm_respawn_tracker[key]["count"] == _TRIPPED_COUNT


def test_refund_ignores_missing_record_and_missing_task() -> None:
    orch = _new_orchestrator()
    orch._refund_respawn_strike("be-pm", "unknown")
    orch._refund_respawn_strike("be-pm", None)
    assert orch._pm_respawn_tracker == {}
