"""CoronerEngine opens at most one autopsy per (incident, kind).

The one-open-at-a-time gate only stopped overlap: once an autopsy completed, the
stranded sweep fired again for the same still-blocked task and wrote another
(six postmortems of one task in a day). A different kind is a new question and
still opens one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from roboco.services import coroner_engine
from roboco.services.coroner_engine import CoronerEngine


def _prior_autopsy(incident_id: object, kind: str) -> MagicMock:
    return MagicMock(
        orchestration_markers={
            "coroner_incident": {"incident_task_id": str(incident_id), "kind": kind}
        }
    )


def _engine_with_history(history: list[MagicMock]) -> CoronerEngine:
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(
            scalars=MagicMock(return_value=iter(history)),
        )
    )
    return CoronerEngine(session)


@pytest.mark.asyncio
async def test_already_autopsied_matches_same_incident_and_kind() -> None:
    incident = uuid4()
    engine = _engine_with_history([_prior_autopsy(incident, "stranded")])

    assert await engine._already_autopsied(incident, "stranded") is True


@pytest.mark.asyncio
async def test_a_different_kind_is_a_new_question() -> None:
    incident = uuid4()
    engine = _engine_with_history([_prior_autopsy(incident, "stranded")])

    assert await engine._already_autopsied(incident, "bounced") is False


@pytest.mark.asyncio
async def test_a_different_incident_is_not_matched() -> None:
    engine = _engine_with_history([_prior_autopsy(uuid4(), "stranded")])

    assert await engine._already_autopsied(uuid4(), "stranded") is False


@pytest.mark.asyncio
async def test_open_for_incident_skips_an_incident_already_autopsied() -> None:
    incident = uuid4()
    engine = _engine_with_history([_prior_autopsy(incident, "stranded")])
    task_svc = MagicMock()
    task_svc.list_open_coroner_cycles = AsyncMock(return_value=[])
    task_svc.get = AsyncMock()
    engine._originate = AsyncMock()  # type: ignore[method-assign]

    with (
        patch.object(coroner_engine, "program_armed", AsyncMock(return_value=True)),
        patch.object(coroner_engine, "get_task_service", return_value=task_svc),
        patch(
            "roboco.services.maintenance_pause.is_paused",
            AsyncMock(return_value=False),
        ),
    ):
        result = await engine.open_for_incident(incident, kind="stranded")

    assert result is None
    engine._originate.assert_not_awaited()
    task_svc.get.assert_not_awaited()
