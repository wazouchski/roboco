"""The dev prompt tells the agent to follow a verb response's `next` step.

be-dev-1 (Qwen) called give_me_work() four times in 13 minutes; each reply was
"call i_will_work_on(...) to start" and it never did. Say so in the prompt.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from roboco.runtime.orchestrator import AgentOrchestrator


@pytest.mark.asyncio
async def test_dev_prompt_says_to_follow_next_and_not_repeat_give_me_work() -> None:
    with patch.object(AgentOrchestrator, "__init__", return_value=None):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._open_findings_prompt_block = AsyncMock(return_value="")  # type: ignore[method-assign]
    p = await orch._build_dev_prompt({"id": "t1", "title": "T", "status": "needs_revision"})
    assert "`next` field" in p
    assert "Do not call give_me_work() again" in p
