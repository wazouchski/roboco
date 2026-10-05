"""upsert_assignment must refuse a Claude-looking name that is not in the catalog.

Regression guard for the 2026-09-30 incident: ``sonnet-5.5`` was assigned while the
deployed build did not know it, and the service silently filed it under the LOCAL
(self-hosted) provider instead of failing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from roboco.models.base import AssignmentScope
from roboco.services.llm import ModelRoutingService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["claude-sonnet-9-9", "sonnet-9.9", "Opus-Next", "haiku-7"]
)
async def test_unknown_claude_name_is_refused(name: str) -> None:
    svc = ModelRoutingService(MagicMock())
    with pytest.raises(ValueError, match="Unknown Claude model"):
        await svc.upsert_assignment(
            scope=AssignmentScope.AGENT_SLUG,
            scope_value="auditor",
            model_name=name,
        )
