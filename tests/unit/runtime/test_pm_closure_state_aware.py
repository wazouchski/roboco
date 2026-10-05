"""Closure dispatch must be state-aware and respawn-gated.

A finished PM-owned container task looped for days: the closure prompt always
said ``submit_up`` (invalid unless the task is ``in_progress``), so a PM spawned
onto an ``awaiting_pm_review`` or ``claimed`` task was rejected (invalid_state),
escalated, the task went ``blocked``, ``main-pm`` flipped it back, and the
dispatcher respawned the PM again -- each cycle two Sonnet spawns.

- ``awaiting_pm_review``: spawn with the PM review prompt (complete()), not the
  closure prompt.
- ``claimed``: start the parent (-> in_progress) before the spawn.
- The respawn circuit breaker (``_pm_respawn_should_gate``) now guards this path.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from roboco.runtime.orchestrator import AgentOrchestrator


def _ready_orch() -> Any:
    with patch.object(AgentOrchestrator, "__init__", return_value=None):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._is_recently_paused = MagicMock(return_value=False)
    orch._fetch_all_descendants = AsyncMock(
        return_value=[{"id": "leaf", "status": "completed"}]
    )
    orch._all_descendants_terminal = MagicMock(return_value=True)
    orch._already_promoted_for_closure = MagicMock(return_value=False)
    orch._closure_pm_for_team = MagicMock(return_value="ux-pm")
    orch._is_agent_active = MagicMock(return_value=False)
    orch._closure_review_pm = AsyncMock(return_value="ux-pm")
    orch._build_pm_closure_prompt = MagicMock(return_value="CLOSURE")
    orch._build_pm_review_prompt = MagicMock(return_value="REVIEW")
    orch._task_git_context = MagicMock(return_value=None)
    orch._pm_respawn_should_gate = AsyncMock(return_value=False)
    orch._try_auto_submit = AsyncMock(return_value=(False, None))
    orch._auto_resume_paused_parent = AsyncMock()
    orch._auto_recover_blocked_parent = AsyncMock()
    orch._auto_start_claimed_parent = AsyncMock()
    orch.spawn_agent = AsyncMock()
    return orch


def _prompt_sent(orch: Any) -> str:
    return cast("str", cast("AsyncMock", orch.spawn_agent).await_args.kwargs["initial_prompt"])


@pytest.mark.asyncio
async def test_awaiting_pm_review_uses_review_prompt_not_closure_prompt() -> None:
    orch = _ready_orch()
    task = {"id": "t1", "status": "awaiting_pm_review", "team": "ux_ui"}

    await orch._maybe_spawn_pm_closure(AsyncMock(), task)

    assert _prompt_sent(orch) == "REVIEW"
    cast("MagicMock", orch._build_pm_closure_prompt).assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "paused", "claimed", "blocked"])
async def test_other_states_keep_the_closure_prompt(status: str) -> None:
    orch = _ready_orch()
    task = {"id": "t2", "status": status, "team": "ux_ui"}

    await orch._maybe_spawn_pm_closure(AsyncMock(), task)

    assert _prompt_sent(orch) == "CLOSURE"
    cast("MagicMock", orch._build_pm_review_prompt).assert_not_called()


@pytest.mark.asyncio
async def test_claimed_parent_is_started_before_spawn() -> None:
    orch = _ready_orch()
    client = AsyncMock()
    task = {"id": "t3", "status": "claimed", "team": "ux_ui"}

    await orch._maybe_spawn_pm_closure(client, task)

    cast("AsyncMock", orch._auto_start_claimed_parent).assert_awaited_once_with(
        client, "t3"
    )
    cast("AsyncMock", orch._auto_resume_paused_parent).assert_not_awaited()
    cast("AsyncMock", orch.spawn_agent).assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "paused", "blocked", "awaiting_pm_review"])
async def test_non_claimed_parent_is_not_started(status: str) -> None:
    orch = _ready_orch()
    await orch._maybe_spawn_pm_closure(
        AsyncMock(), {"id": "t4", "status": status, "team": "ux_ui"}
    )
    cast("AsyncMock", orch._auto_start_claimed_parent).assert_not_awaited()


@pytest.mark.asyncio
async def test_respawn_breaker_blocks_the_closure_spawn() -> None:
    orch = _ready_orch()
    orch._pm_respawn_should_gate = AsyncMock(return_value=True)

    await orch._maybe_spawn_pm_closure(
        AsyncMock(), {"id": "t5", "status": "awaiting_pm_review", "team": "ux_ui"}
    )

    cast("AsyncMock", orch._pm_respawn_should_gate).assert_awaited_once()
    cast("AsyncMock", orch.spawn_agent).assert_not_awaited()


@pytest.mark.asyncio
async def test_breaker_is_not_charged_when_auto_submit_handles_closure() -> None:
    """An auto-submit (no PM spawned) must not take a respawn strike."""
    orch = _ready_orch()
    orch._try_auto_submit = AsyncMock(return_value=(True, None))

    await orch._maybe_spawn_pm_closure(
        AsyncMock(), {"id": "t6", "status": "in_progress", "team": "ux_ui"}
    )

    cast("AsyncMock", orch._pm_respawn_should_gate).assert_not_awaited()
    cast("AsyncMock", orch.spawn_agent).assert_not_awaited()
