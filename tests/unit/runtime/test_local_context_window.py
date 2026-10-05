"""Local (self-hosted) agents are told their model's real context window.

Regression guard for the 2026-10-01 incident: Claude Code does not recognise a
self-hosted model id, never compacted before the server's 64k window filled, and
the overflow ended agent sessions with exit code 1 (crash, restart, stranded).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from roboco.config import Settings, settings
from roboco.models.runtime import OrchestratorAgentConfig as AgentConfig
from roboco.runtime.orchestrator import AgentOrchestrator

_QWEN = "qwen36-reap"
_QWEN_WINDOW = 120_000
_FALLBACK_WINDOW = 56_000
_SPEC = f"{_QWEN}={_QWEN_WINDOW}, *={_FALLBACK_WINDOW}"
_ENV_NAME = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"


def _env_flags(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]


def _config(model: str, provider_type: str) -> AgentConfig:
    return AgentConfig(
        agent_id="be-dev-1",
        blueprint_path=Path("/tmp/blueprint.md"),
        model=model,
        provider_type=provider_type,
        provider_base_url="http://172.20.0.1:1234",
    )


def test_unset_means_no_window() -> None:
    assert Settings(local_context_windows="").local_compact_window_for(_QWEN) is None


def test_exact_match_beats_wildcard() -> None:
    s = Settings(local_context_windows=_SPEC)
    assert s.local_compact_window_for(_QWEN) == _QWEN_WINDOW


def test_wildcard_is_the_fallback() -> None:
    s = Settings(local_context_windows=_SPEC)
    assert s.local_compact_window_for("gpt-oss-20b") == _FALLBACK_WINDOW


@pytest.mark.parametrize("bad", ["qwen36-reap", "qwen36-reap=abc", "qwen36-reap=0", "=5"])
def test_malformed_entries_are_ignored(bad: str) -> None:
    assert Settings(local_context_windows=bad).local_compact_window_for(_QWEN) is None


def test_local_agent_gets_the_window_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "local_context_windows", _SPEC)
    cmd: list[str] = []
    AgentOrchestrator._append_provider_env(cmd, _config(_QWEN, "local"))
    assert f"{_ENV_NAME}={_QWEN_WINDOW}" in _env_flags(cmd)


def test_other_providers_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "local_context_windows", _SPEC)
    cmd: list[str] = []
    AgentOrchestrator._append_provider_env(cmd, _config("kimi-k2.7-code:cloud", "ollama_cloud"))
    assert not any(f.startswith(_ENV_NAME) for f in _env_flags(cmd))
