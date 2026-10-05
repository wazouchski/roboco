# Claude handoff — 2026-09-29 (updated 2026-09-30)

## Update 2026-09-30: why nothing was being worked on, and what changed

The CEO found the board idle. There were two separate causes; both are fixed locally and deployed, **and nothing is committed**. RoboCo is upstream software the CEO operates but does not own, and there is no private fork yet, so do not assume commits or PRs have a home. Offer a local branch or a patch file, and note that upstreaming needs a fork plus the signed CLA.

**Cause 1: respawn breaker parked every queued task.**
- `ROBOCO_LOCAL_MAX_CONCURRENT` now defaults to 1 (local edit in `roboco/config.py`), so `spawn_agent` skips queued spawns each tick with "Spawn skipped: local concurrency cap reached".
- `_pm_respawn_should_gate` takes a strike BEFORE `spawn_agent` runs, so cap-skips counted as unproductive respawns. Every queued leaf task tripped the breaker (count 4, threshold 3, `notified=true`) and the dispatcher then went silent, with no log line after the one-shot trip.
- Fix: `_refund_respawn_strike` in `roboco/runtime/engines/dispatch_breaker.py`, called from the first skip branch of `spawn_agent` in `roboco/runtime/engines/spawn_launch.py` (Protocol stub in `_types.py`). Test: `tests/unit/runtime/test_respawn_strike_refund.py`.
- Verified after deploy: 31 cap-skips in 3 minutes, every queued task's count stayed at 1.

**Cause 2: UX/UI cell `ux-pm` crash loop (parent `74f865c9`).**
- `dev_task_collision_edges` (`roboco/services/sequencing.py`) sorts siblings by (priority, sequence). A new sibling arrives at sequence 0 and sorts ahead of stamped siblings; after it is wave-stamped it ties with the older one, the tie flips the edge, and `add_dependency` refuses the reverse edge with "would close a dependency cycle". The 409 rolled back the whole delegate on every retry, so `ux-pm` looped and exited with code 1.
- Fix in `roboco/services/task.py`: `wire_sibling_collision_dag` skips a cycle-closing edge (existing order wins, logs a warning); new `raise_dependent_waves` lifts older held-back siblings above the new task after `stamp_wave_sequence` (called from `roboco/services/gateway/choreographer/_impl.py`). The same file turns a cycle `ConflictError` into an `invalid_state` envelope with a remediate hint. Tests are in `tests/unit/services/test_task.py` and `tests/unit/services/test_sequencing.py`.
- Live data: row `729a4ed3` was set to sequence 2 by hand.

**Deployed 2026-09-30 about 08:40 UTC.** `orchestrator` and `dispatcher` were rebuilt and recreated on one image, so the dispatcher rollout caveat below is resolved, and it includes the local respawn-breaker edit.

**Operational gotchas**
- The running stack comes from `docker-compose.registry.yml`, NOT the default `docker-compose.yml` (that file only defines `orchestrator-blue/green`). Image `roboco-orchestrator-local`, built from `docker/orchestrator.Dockerfile`; `orchestrator` and `dispatcher` share it. Rebuild with `docker compose -f docker-compose.registry.yml build orchestrator`, then `up -d --no-deps orchestrator dispatcher`.
- There is no API to reset the respawn breaker. The dispatcher flushes its in-memory tracker to `respawn_tracker` on shutdown, so resetting rows requires stop, delete, start, in that order, or the delete is undone.
- The local `.venv` has no pytest and `uv` is offline. Run unit tests in a throwaway container from the orchestrator image: `uv pip install pytest pytest-asyncio pytest-timeout hypothesis ruff` into `/app/.venv`, then `python -m pytest -o addopts=""` with the tree mounted read-only (set `RUFF_CACHE_DIR=/tmp/rc` for ruff).
- The API needs an `X-Agent-ID` header, so task listing through nginx returns 401. Read-only SQL against `roboco-postgres` (`psql -U roboco -d roboco`) is the quick way to inspect tasks.
- Known failures that pre-date these fixes (same failures without them): 8 runtime unit tests (`test_interactive_provider_guard` x6, `test_respawn_gate_cooldown::test_status_change_is_the_reset_path`, `test_stalled_marker::test_retrip_after_cooldown_is_one_shot_not_double_counted`) and 3 ruff ARG002 errors in `_pm_cooldown_gate`. They come from the uncommitted breaker/provider edits.

**Later on 2026-09-30: PM routing and local-model tuning (CEO-approved).**
- **PM roles on Claude:** `model_assignments` now has `role:cell_pm` and `role:main_pm` -> `anthropic` / `sonnet` (RoboCo maps `sonnet` to `claude-sonnet-5`). The global assignment stays `local` / `google/gemma-4-12b-qat`, so devs, QA and doc agents still run on the local model. Claude agents have no concurrency cap; only LOCAL (1) and KIMI (1) are capped. Claude runs on the mounted `~/.claude` subscription login, so PM sessions spend the CEO's Claude quota. **Rollback:** delete those two `role` rows. The CEO wants Claude kept on the PM roles until a local model is proven able to do the job.
- **Why local was slow:** LM Studio's saved default for Gemma (`~/.lmstudio/.internal/user-concrete-model-default-config/google/gemma-4-12b-qat.json`) had `offloadRatio: 0.5`, so 25 of 48 layers ran on the CPU (about 5 tokens/s, GPU at 14%). Raised to `1` (all layers on the RTX 3060 12 GB; 9.3 GB used at 64k context with q8 KV cache and flash attention). Backup of the old file is in the session scratchpad. `lms load google/gemma-4-12b-qat -y` now picks it up.
- **Benchmark** (script `bench.py` / `bench_anth.py` in the session scratchpad; 5 trials each; synthetic delegate-contract and claim-ordering tests, not the real agent contract): Gemma baseline 25 layers: 8.6k-token prompt 87 s, 25.7k 125 s, delegate call 140 s, claim 41 s. Gemma full GPU: 17 s, 30 s, 16 s, 5 s. `gpt-oss-20b` at 0.8 offload (20 of 24 layers, 10.3 GB): 16.5 s, 30.7 s, 8 s, 3 s via the OpenAI endpoint. Through the Anthropic `/v1/messages` endpoint, which is the path the agents use: Gemma 4.4 s delegate and 2.3 s claim, gpt-oss 7.1 s and 2.9 s. Every configuration passed 5/5, so these tests measure speed, not fidelity. Winner: Gemma 4 with full GPU offload (8x faster than before, no quality difference seen, fits fully in VRAM). No change to the global assignment was needed. Other local models (`qwen3-30b-a3b`, `devstral-small`, etc.) were not tested.
- **Open question:** whether the tuned local Gemma can handle the PM roles (the failures seen were `covers_parent_criteria` omitted 8 times and `tracing_gap`, under 26k+ token real prompts). Test that before moving the PM rows back.
- The maintenance pause used for the benchmark (dispatch, engines, board programs) was lifted at 11:13 UTC.

**Still open**
- Throughput is slow by design: one local agent at a time, and each PM or dev session takes minutes on the local 12B model. `ROBOCO_LOCAL_MAX_CONCURRENT=2` is the knob if the GPU can serve two sessions; the CEO chose 1, so leave it unless told.
- `ux-pm` containers exited with code 1 after about 3.5 minutes at 07:38 and 07:57; the cause was the delegate loop above, but confirm it no longer recurs.
- Breaker rows still tripped on purpose or left alone: `main-pm` on `539560ca` and `74f865c9`, `be-pm` on `305f4cda` (a blocked placeholder). Check them before assuming they are stuck.
- Two `awaiting_qa` tasks (`46781ab0` backend, `7569f960` UX) and ten pending tasks were waiting for the single slot at deploy time.
- The "Suggested first checks" list at the bottom is still valid, except item 3 (dispatcher is now rebuilt and recreated).

---

## User goals and operational context

- The CEO wants a clear picture of what agents are doing. Today’s discussion surfaced that a task can appear **In Progress** on Kanban while its detail says **active — planning**; task status and planning phase are separate signals and should be presented consistently.
- The CEO asked whether separate frontend/backend projects are useful, whether task state appears on Kanban, and why agents were producing repeated “stranded” notifications. Treat this as ongoing operational UX/reliability work; don’t assume every pending task means an agent is idle or working invisibly.
- The CEO asked to reduce dispatch to one task at a time. There are local edits to concurrency configuration and compose files; inspect the diff before changing or claiming fleet-wide behavior.
- The CEO reported updating Gemma’s context to 64K and K/V cache quantization to `8_0`. This is a user-made host/model configuration, not a repo change to repeat or overwrite.
- Blender is installed on the CEO’s PC with MCP, but likely absent from agent containers. The CEO explicitly chose to wait until an agent asks for it; do not install it proactively.
- The CEO asked about PR tasks and learned that review agents own normal PR review. A task blocked only to create/hold a folder may be a placeholder rather than a dependency blocking active work; verify its dependency links before interpreting the board.

## Shipped today: KB documents open from the panel

Search result cards and Browse-tab document rows now open a right-side drawer with the full indexed content, source, index category, and chunk count. Content is reconstructed from indexed chunks.

- API: `GET /api/v1/optimal/kb/{index_type}/document?source=...`
- UI and API wiring: `panel/src/components/knowledge-base/`, `panel/src/hooks/use-knowledge-base.ts`, `panel/src/lib/api/knowledge-base.ts`, and `panel/src/types/index.ts`.
- Backend retrieval: `roboco/api/routes/optimal.py`, `roboco/services/optimal.py`, `roboco/services/optimal_brain/indexes/base.py`, and `roboco/services/optimal_brain/vector_store.py`.
- Verification: Python compilation and `git diff --check` passed. Docker production build succeeded for both orchestrator and panel, including Next.js TypeScript checking. Orchestrator and panel were recreated; orchestrator reported healthy, panel running.
- The API route’s permission gate currently uses `KBAction.VIEW_STATS`; review whether this should instead use a dedicated document-read/search permission before broadening access.

## Agent incidents and code state

- There were repeated “Agent stranded” notifications for `fe-pm`, `be-pm`, and `main-pm`, plus reports of agents stuck on tasks. A local edit changes the PM respawn breaker to keep an exhausted task parked until a real task state/assignment change or explicit reset, avoiding cooldown-based retry/notification loops.
- Related edits are in `roboco/runtime/engines/dispatch_breaker.py` and `tests/unit/runtime/test_respawn_gate_cooldown.py`.
- Important deployment caveat (resolved 2026-09-30, see the update above; dispatcher is now on the new image): the rebuild/recreate command targeted only `orchestrator` and `panel`. `dispatcher` remained running on its prior container/image, so the respawn-breaker edit was **not confirmed deployed**. Inspect dispatcher status/image and plan a controlled dispatcher rollout before assuming stranded notifications are fixed.
- The CEO asked the system to pause briefly so the GPU could breathe and later explicitly said “resume.” Do not leave services paused based on that earlier request.
- Immediately after the last API restart, logs showed `fe-pm` trying to delegate under task `f4042843-5e3d-479d-a280-46e0d21f5c7b`; the child was rejected because `covers_parent_criteria` was omitted. The parent criteria IDs/text are in the API log. There was also a warning that automatic PM decision recording hit a closed transaction, though the gateway continued and rejected the delegate for the missing criteria. Check the task/agent’s current state before intervening.
- No Blender install was requested. Do not act on this unless an agent requests it.

## Workspace cautions

The working tree is intentionally dirty from accumulated work. Preserve all existing changes and inspect them before edits. Current modified areas include provider/model routing, task/agent panel views, concurrency/compose configuration, project access, the respawn breaker, and the KB reader. The KB drawer component is newly untracked.

Other untracked items present at handoff:

- `GET`
- `POST`
- `front/registry-active-upstreams.conf`

Their origin and purpose have not been established. Do not delete or overwrite them.

No commits were made today. Do not claim the broader collection of local changes is committed or fully deployed. The compose output also warned about agent containers as orphans; they were not removed.

## Suggested first checks

1. Inspect live Kanban/task detail and current status for the three PMs and task `f4042843-5e3d-479d-a280-46e0d21f5c7b`.
2. Verify whether `fe-pm` retried delegation with the required `covers_parent_criteria` and whether the closed-transaction warning repeats.
3. Check dispatcher logs and image/container state; the respawn gate fix is only local until dispatcher is rebuilt/recreated.
4. When convenient, click a KB Search result and a Browse row in the panel to confirm the drawer loads actual indexed document content.
