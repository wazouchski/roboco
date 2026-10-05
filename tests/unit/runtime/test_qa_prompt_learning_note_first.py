"""The QA prompt must make the learning note a required step BEFORE pass/fail.

The gateway refuses pass()/fail() until a journal:learning entry exists
(tracing gate). The prompt used to list note(scope='learning') as an optional
step AFTER the decision, so a weaker local model followed it in order, got
refused, and looped on i_am_blocked (invalid in awaiting_qa) -- 13 calls in
23 minutes on ux-qa, 2026-10-03.
"""

from __future__ import annotations

from unittest.mock import patch

from roboco.runtime.orchestrator import AgentOrchestrator


def _prompt() -> str:
    with patch.object(AgentOrchestrator, "__init__", return_value=None):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
    return orch._build_qa_prompt({"id": "t1", "title": "T", "team": "ux_ui"})


def test_learning_note_comes_before_pass_and_fail() -> None:
    p = _prompt()
    assert p.index('note(scope="learning"') < p.index("PASS: pass(")
    assert p.index('note(scope="learning"') < p.index("FAIL: fail(")
    assert "REQUIRED BEFORE" in p


def test_prompt_forbids_i_am_blocked_during_qa() -> None:
    p = _prompt()
    assert "Do NOT call i_am_blocked" in p


def test_prompt_says_a_note_is_not_a_verdict() -> None:
    p = _prompt()
    assert "A note\n   is NOT a verdict" in p or "is NOT a verdict" in p
    assert "call pass() or fail()" in p
