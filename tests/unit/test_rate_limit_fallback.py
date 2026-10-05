"""Rate-limit handling: Codex 'usage limit' detection + local fallback routing.

1. Codex's real limit message ("You've hit your usage limit ... try again at
   11:41 PM") matched none of the detector's words, so a limited run was
   classified as a crash (breaker strikes) instead of exit 75 / park.
2. ``settings.rate_limit_fallback_model``: while a cloud provider is parked,
   spawns are routed to the local fallback model instead of waiting, and the
   park step does not register a wait record.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from roboco.config import settings
from roboco.llm.providers import codex_cli_sniff
from roboco.models.base import ModelProvider
from roboco.runtime.orchestrator import AgentOrchestrator
from roboco.services.llm import AgentRoute, ModelRoutingService

CODEX_MSG = (
    "You’ve hit your usage limit. Upgrade to Pro "
    "(https://chatgpt.com/explore/pro), visit "
    "https://chatgpt.com/codex/settings/usage to purchase more credits or try "
    "again at 11:41 PM."
)


def test_codex_usage_limit_message_is_a_rate_limit() -> None:
    assert codex_cli_sniff.is_rate_limited(CODEX_MSG)


def test_codex_unrelated_error_is_not_a_rate_limit() -> None:
    assert not codex_cli_sniff.is_rate_limited("sandbox failed to start")


def _route(ptype: ModelProvider, model: str = "gpt-5.6-luna") -> AgentRoute:
    return AgentRoute(
        provider_id=uuid4(),
        provider_type=ptype,
        base_url="https://x",
        auth_token="t",
        model_name=model,
    )


def _svc(local_route: AgentRoute | None) -> Any:
    svc = ModelRoutingService.__new__(ModelRoutingService)
    svc.session = MagicMock()
    svc.log = MagicMock()
    local = MagicMock(enabled=True)
    svc._find_local_provider = AsyncMock(return_value=local)  # type: ignore[method-assign]
    svc._local_route_or_none = AsyncMock(return_value=local_route)  # type: ignore[method-assign]
    return svc


def _tracker(limited: bool | Exception) -> Any:
    inst = MagicMock()
    if isinstance(limited, Exception):
        inst.is_rate_limited = AsyncMock(side_effect=limited)
    else:
        inst.is_rate_limited = AsyncMock(return_value=limited)
    return patch(
        "roboco.services.gateway.rate_limit_tracker.RateLimitStateTracker",
        return_value=inst,
    )


@pytest.fixture
def fallback_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_fallback_model", "qwen36-reap")


@pytest.mark.asyncio
async def test_off_by_default_leaves_route_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "rate_limit_fallback_model", "")
    route = _route(ModelProvider.OPENAI)
    with _tracker(True):
        assert await _svc(_route(ModelProvider.LOCAL, "qwen36-reap"))._rate_limit_fallback_route(route, "be-dev-2") is route


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_parked_cloud_provider_falls_back_to_local() -> None:
    local = _route(ModelProvider.LOCAL, "qwen36-reap")
    with _tracker(True):
        out = await _svc(local)._rate_limit_fallback_route(
            _route(ModelProvider.OPENAI), "be-dev-2"
        )
    assert out is local


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_not_parked_keeps_route() -> None:
    route = _route(ModelProvider.OPENAI)
    with _tracker(False):
        assert await _svc(_route(ModelProvider.LOCAL))._rate_limit_fallback_route(route, "x") is route


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_local_route_is_never_replaced() -> None:
    route = _route(ModelProvider.LOCAL, "qwen36-reap")
    with _tracker(True):
        assert await _svc(None)._rate_limit_fallback_route(route, "x") is route


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_unreachable_local_keeps_original_route() -> None:
    route = _route(ModelProvider.OPENAI)
    with _tracker(True):
        assert await _svc(None)._rate_limit_fallback_route(route, "x") is route


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_tracker_error_keeps_original_route() -> None:
    route = _route(ModelProvider.OPENAI)
    with _tracker(RuntimeError("redis down")):
        assert await _svc(_route(ModelProvider.LOCAL))._rate_limit_fallback_route(route, "x") is route


def _orch() -> Any:
    with patch.object(AgentOrchestrator, "__init__", return_value=None):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._finalize_spawn_session = AsyncMock()
    orch._make_tracker = MagicMock(return_value=MagicMock(activate=AsyncMock()))
    orch._waiting_records = {}
    orch._persist_waiting_record = AsyncMock()
    return orch


async def _park(orch: Any, provider: str) -> None:
    inst = MagicMock(current_task_id="t1", container_id="c", error_count=3)
    await orch._park_provider_unavailable(
        "be-dev-2", inst, provider=provider, retry_after=60.0, kind="rate_limited"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("fallback_on")
async def test_park_skips_wait_record_when_fallback_on() -> None:
    orch = _orch()
    await _park(orch, "openai")
    assert orch._waiting_records == {}
    orch._make_tracker.return_value.activate.assert_awaited_once()  # provider still parked


@pytest.mark.asyncio
async def test_park_keeps_wait_record_when_fallback_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "rate_limit_fallback_model", "")
    orch = _orch()
    await _park(orch, "openai")
    assert "be-dev-2" in orch._waiting_records
