# Local Model Tuning Report: TurboQuant llama.cpp fork for RoboCo

**Date:** 2026-09-30
**Hardware:** RTX 3060 12GB, Ryzen 5 9600X (6 cores / 12 threads), 61GB RAM, Arch (Omarchy), CUDA 13.3
**Goal:** a snappy, accurate set of auto-switching local models for RoboCo.
**Related:** `~/AI/local-model-rubrick.md` (2026-09-21 LM Studio evaluation of 13 models).

> **Read this first.** Everything below is **speed and memory-fit data**. Accuracy has **not** been measured beyond one 2-turn tool-call smoke test per model. Almost all numbers are **single runs** (`-r 1`), so differences under about 1 to 2% are noise. Nothing here has been tested against RoboCo itself.

---

## 1. Setup

- **Source of the method:** Codacus tutorial (youtu.be/0AqpaFm11oI). Transcript and pinned comment were supplied by the user. Its recipe: REAP-pruned MoE models, `llama-bench` sweeps for threads and ubatch, `--cache-reuse 256`, TurboQuant KV compression (`turbo4` keys / `turbo2` values), a `models.ini` preset router, Pi agent, Tailscale.
- **Fork:** `TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`, cloned to `~/llama-cpp-turboquant`, built with `cmake -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. `cmake` had to be installed first (`sudo pacman -S cmake`).
- **Models:** read directly from LM Studio's folder (`~/.lmstudio/models/...`). No copies or conversions.
- **Local patch (one line):** `src/llama-kv-cache.cpp`, `llama_kv_cache::type_v()` now returns `layers[0].v ? layers[0].v->type : GGML_TYPE_COUNT`. Without it, `llama-bench` segfaults on MLA models (GLM-4.7-Flash) because MLA caches have no V tensor (`has_v = !is_mla`) and `type_v()` dereferenced null. Only `llama-bench` called it; `llama-server` was unaffected. Not reported upstream.
- **Fork-specific flags seen in `--help`:** `-ctk/-ctv` accept `turbo2|turbo3|turbo4`; `--moe-cache` (caches hottest CPU experts in spare VRAM) exists but was **not tested**.

### Models tested

| Model | File | Size | Notes |
|---|---|---|---|
| Qwen3.6-28B-REAP20-A3B | `barozp/...Q4_K_S.gguf` | 16.2GB | hybrid attention (small KV); the tutorial used Q4_K_M |
| GLM-4.7-Flash-REAP-23B-A3B | `unsloth/...IQ4_XS.gguf` | 12.6GB | MLA: no V cache, K and V types must match |
| gpt-oss-20b | `lmstudio-community/...MXFP4.gguf` | 12.1GB | MoE |
| Gemma 4 12B QAT | `lmstudio-community/...Q4_0.gguf` | 7.0GB | dense, fits fully in VRAM |

Role assignment: **Qwen is the coder** (user-confirmed). The transcript says only "one is fantastic for coding, the other is a generalist, faster but weaker on code" without naming which; GLM is therefore treated as the generalist.

---

## 2. Method notes and caveats

- Benchmarks: `llama-bench -ngl 99 -fa 1`, `-ub 1024 -b 4096` unless stated. "Empty context" = `-p 4096 -n 64`; "16k / 32k depth" = `-d 16384` / `-d 32768` with a short prompt.
- **64k fit** was determined by starting `llama-server -c 64000..65536` at each `--n-cpu-moe` (stepping by 2, so a true minimum can be 1 lower).
- **Flaw found in the fit method:** a server that *starts* can still fail on the first request. gpt-oss at `--n-cpu-moe 4` started (11.8GB used) then crashed at `cublasCreate_v2` on the first inference. Qwen and GLM passed real requests at their chosen minimums; gpt-oss was moved to 5.
- LM Studio must not hold a model while benchmarking. It held 8.8GB at the start and was ejected by the user.
- The KV-type comparison is only fair when each type is given its own minimum `--n-cpu-moe`, because the point of compression is freeing VRAM for experts.

---

## 3. Results by model

### 3.1 GLM-4.7-Flash REAP (MLA)

**Threads** (`--n-cpu-moe 9`, p512/n128, 3 reps): prefill / decode

| threads | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| prefill t/s | 1258 | 1281 | **1282** | 1278 | 1264 |
| decode t/s | 57.6 | 61.3 | 62.1 | 62.4 | 62.8 |

Flat. Used 4.

**`--n-cpu-moe` × ubatch** (4 threads, p4096, single runs): prefill / decode; "OOM" = context creation failed

| n-cpu-moe | ub 512 | ub 1024 | ub 2048 |
|---|---|---|---|
| 9 | 1068 / 61.8 | OOM | OOM |
| 11 | 1004 / 59.2 | **1364 / 58.6** | OOM |
| 12 | 974 / 57.6 | 1341 / 57.3 | OOM |
| 14 | 925 / 55.2 | 1284 / 54.9 | **1581 / 55.2** |

Each extra expert layer moved to CPU costs about 1 to 2 t/s decode; a bigger ubatch buys a large prefill gain (ub 256 → 752, 512 → 1066, 1024 → ~1364 at n-cpu-moe 11).

**KV type, `--n-cpu-moe 11`, empty context:** f16 1364 / 59.0 · q8_0 1361 / 58.2 · turbo4 934 / 56.7 · turbo3 930 / 57.0 · turbo2 934 / 57.4.
**Mixed K/V is refused** on GLM: "model does not support different K (turbo4) and V (f16/turbo2) cache types". Use the same type for both.

**32k depth, `--n-cpu-moe 17`, p512:** f16 249 / 37.6 · q8_0 248 / 28.3 · turbo4 116 / 17.8 · turbo2 116 / 21.4.

**64k context, minimum `--n-cpu-moe` that fits:** f16 **23**, turbo4 13, turbo2 **11**.

| KV @ min | empty ctx prefill / decode | 16k depth prefill / decode |
|---|---|---|
| f16 @ 23 | 1128 / 45.9 | **444 / 39.2** |
| turbo2 @ 11 | 933 / **57.4** | 227 / 32.1 |
| turbo4 @ 13 | 906 / 54.2 | 226 / 27.3 |

The tutorial's "KV compression gives more decode, costs prefill" holds at empty context (+25% decode, -17% prefill) but **reverses with depth**: at 16k f16 is ~2x faster at prefill and 22% faster at decode.

### 3.2 Gemma 4 12B QAT (dense, fully on GPU)

- **Threads 2/4/6:** prefill 1422 / 1420 / 1404, decode 41.8 flat. Irrelevant.
- **Ubatch 256/512/1024/2048:** prefill 1274 / 1347 / **1380** / 1364; decode ~41.8 flat.
- **KV, empty context:** f16 1386 / 41.8 · q8_0 1367 / 41.3 · turbo4/turbo4 1325 / 39.6 · turbo4/turbo2 1328 / 39.9.
- **KV, 16k depth:** f16 1059 / 37.2 · q8_0 1043 / 35.2 · turbo4/turbo2 1028 / 29.7.
- **Fit:** 64k with f16 (10.9GB incl. desktop). 128k: f16 **no**; q8_0 yes (10.5GB); turbo4/turbo2 yes (9.3GB). Mixed K/V types work (not MLA).

### 3.3 gpt-oss-20b (MoE)

- **Minimum `--n-cpu-moe` at 64k** (steps of 2): f16 6, q8_0 4, turbo4/turbo2 4. **`--n-cpu-moe 4` crashed on first real request** (no VRAM left for cuBLAS); the preset uses **5**.
- **Threads** (f16, n-cpu-moe 6, prefill): 2 → 183, 3 → 261, 4 → 335, 5 → 400, 6 → 446; decode 62 to 64. **Prefill depends on threads** (CPU experts). Extended (q8_0, n-cpu-moe 4, p1024): 6 → 571/68.1, **8 → 590/70.2**, 10 → 613/69.3, 12 → 637/**58.8**. Decode drops at 12 threads. Use 8 to 10.
- **KV at 64k:**

| KV | n-cpu-moe | empty ctx prefill / decode | 16k prefill / decode |
|---|---|---|---|
| f16 | 6 | 340 / 63.9 | 330 / **57.6** |
| **q8_0** | 4 | **478 / 70.9** | **460** / 51.9 |
| turbo4/turbo2 | 4 | 91 / 67.0 | did not finish (>10 min, killed) |

The turbo types are unusable on gpt-oss (prefill collapses to 91 t/s).

### 3.4 Qwen3.6-28B REAP20

- **Threads** (q8_0, n-cpu-moe 18, p1024): 2 → 294/60.8, 3 → 389/58.0, 4 → 467/60.5, **6 → 569/64.0**, 8 → 513/63.0, 10 → 546/59.4.
- **Minimum `--n-cpu-moe` at 64k** (steps of 2): f16 **20**, q8_0 **18**, turbo4/turbo2 **16 (server) / 17 (benchmark)**. The KV cache is small (hybrid model), so compression frees little.

| KV @ min | empty ctx prefill / decode | 16k prefill / decode |
|---|---|---|
| f16 @ 20 | 442 / 61.7 | 414 / **58.3** |
| q8_0 @ 18 | 476 / 63.7 | 462 / 47.1 |
| turbo4/turbo2 @ 17 | **497** / 61.1 | **466** / 39.0 |

KV sweeps ran at 4 threads; the best thread count is 6, so absolute numbers would be slightly higher. Qwen holds decode best at depth (61.7 → 58.3 at 16k).

### 3.5 Comparison (64k context, each model's best setting)

| Model | Setting | Empty ctx prefill / decode | 16k depth prefill / decode |
|---|---|---|---|
| Gemma 4 12B | f16, fully on GPU, 4 threads | 1386 / 41.8 | 1059 / 37.2 |
| GLM-4.7 REAP | f16, `--n-cpu-moe 23`, 4 threads | 1128 / 45.9 | 444 / 39.2 |
| gpt-oss-20b | q8_0, `--n-cpu-moe 4`*, 8 threads | 478 / 70.9 | 460 / 51.9 |
| Qwen3.6 REAP | f16, `--n-cpu-moe 20`, 6 threads | 442 / 61.7 | 414 / 58.3 |

\* crashed on first inference; run at 5 in the preset (not re-benchmarked at 5).

Cross-check against the 2026-09-21 rubric (LM Studio, 150-token generation, different method): gpt-oss-20b 45.3 tok/s and Qwen3-30B-A3B 28.5 tok/s there, versus 70.9 and ~43 (single short run at `--n-cpu-moe 30`) here. The fork plus tuned offload looks faster, but the methods differ, so treat it as indicative only.

---

## 4. Serving setup

`~/.config/llama-server/models.ini` (global: ctx 65536, batch 4096, ubatch 1024, `ngl 99`, flash-attn on, `no-mmap`, `jinja`, `cache-reuse 256`, metrics on; `mlock` omitted):

| Preset | n-cpu-moe | threads | KV |
|---|---|---|---|
| `qwen36-reap` | **22** (benchmarked minimum 20, +2 headroom) | 6 | f16 |
| `glm47-flash-reap` | **25** (benchmarked minimum 23, +2 headroom) | 4 | f16 |
| `gpt-oss-20b` | **6** (benchmarked minimum 4, +2 headroom) | 8 | q8_0 |
| `gemma4-12b` | 0 (dense, all GPU; no expert layers to add) | 4 | f16 |

**Final serving (see §10):** a systemd *user* service `llama-router` (`~/.config/systemd/user/llama-router.service`, `Restart=always`) runs `llama-server --models-preset ~/.config/llama-server/models.ini --models-max 1 --host 172.20.0.1 --port 1234`. It is bound **only to the RoboCo Docker bridge address**, so it is not exposed on the LAN. Router mode loads models on demand; measured load + first token 10s (Qwen), 13s (GLM), 17s (Gemma), 4s (gpt-oss, cached). All four presets were re-verified at the +2 headroom with a real ~10.7k-token prompt (no out-of-memory).

---

## 5. RoboCo findings (from its repo and docs)

- **Fleet path:** RoboCo has a `self_hosted` routing mode (`ModelProvider.LOCAL`) that runs agents on Claude Code with `ANTHROPIC_BASE_URL` injected from the provider row (same shape as its Ollama Cloud and Z.ai modes). The fork's `llama-server` implements the Anthropic `/v1/messages` format, and the smoke test below confirms it works through the router. **RoboCo itself has not been connected or run.**
- **Auxiliary path:** `ROBOCO_LOCAL_LLM_BASE_URL` / `ROBOCO_LOCAL_LLM_MODEL` is an OpenAI-compatible `/v1` endpoint restricted to localhost/private addresses, used for RAG answer synthesis, lesson distillation and X/video drafting (not for code-writing agents).
- **Requirements and risks:**
  - RoboCo's lifecycle needs reliable structured tool output; it **retired Haiku from QA/PM roles** for looping on malformed envelopes. Local models face the same bar.
  - Up to 25 agents may want different models concurrently; one 3060 serves one model at a time (`--models-max 1`), so swaps can thrash (10 to 17s each). Prefer few models in role groups or one workhorse with parallel slots (`-np`).
  - Agents run in Docker, so the server must bind to an address the containers can reach (currently `127.0.0.1` only).
  - RoboCo's self-hosted model discovery probes Ollama's `/api/tags`; compatibility with `llama-server`'s model listing is **unchecked**.

---

## 6. Tool-call smoke test (Anthropic `/v1/messages`, one sample per model)

Prompt: create `utils/slug.py` with `slugify`, then run `tests/test_slug.py`; two tools (`create_file`, `run_tests`); second turn supplied a tool result.

| Model | Turn 1 | Turn 2 | Load+turn 1 |
|---|---|---|---|
| Qwen3.6 REAP | valid `create_file` with `slugify` | `run_tests` (correct) | 10s |
| GLM-4.7 REAP | valid `create_file` with `slugify` | `run_tests` (correct) | 13s |
| Gemma 4 12B | valid `create_file` | wrote a test file instead of running tests | 17s |
| gpt-oss-20b | called `run_tests` first (out of order) | then `create_file` | 4s |

(The gpt-oss run was done after the `--n-cpu-moe` fix; my harness fed a `create_file` result to a `run_tests` call in turn 2, so its turn-2 result is not meaningful.) A single pass/fail sample is not a reliability measure.

---

## 6b. Harder multi-step tool-call eval (3 trials per task per model, objective grading)

Harness: Anthropic `/v1/messages` through the router, ~10.7k-token system prompt (120 filler rules + a 40-section handbook) with the decisive rules buried inside it; preset sampling settings; output cap 6,000 tokens per response. The grading was validated first (the buggy package fails 5 tests; a reference fix passes all).

- **T1 coding loop:** fix 5 bugs in a 2-file Python package using `list_files`/`read_file`/`write_file`/`run_tests` (real unittest execution, max 16 turns). Pass = tests pass AND `tests/` untouched (buried rule 41) AND final message is exactly one line starting `DONE:` (buried rule 67).
- **T2 QA envelope:** `pass_review` vs `fail_review` where the developer claims all tests pass but one test output contradicts it. Pass = `fail_review`, flagging **only** AC-3, schema valid. Also recorded: quoting the failing test name (buried rule 88).
- **T3 PM delegation:** split a 5-criterion parent task across backend/frontend/uxui with `covers_parent_criteria`. Pass = all criteria covered, every criterion mapped to the expected cell, zero invalid calls (buried rule 103).

| Model | T1 coding | T2 QA envelope | T3 delegation |
|---|---|---|---|
| **Qwen3.6 REAP** | **3/3** (avg 50s, 6.7 turns) | **3/3** (22s; quoted test 3/3) | **3/3** (31s) |
| **GLM-4.7 REAP** | **3/3** (avg 64s, 6.3 turns) | **3/3** (14s; quoted test 3/3) | **3/3** (23s) |
| **gpt-oss-20b** | **3/3** (avg 38s, 7.0 turns) | 3/3 (9s; quoted test **2/3**) | 2/3 (66s; one invalid delegate call) |
| **Gemma 4 12B** | **1/3** (avg 395s; 2 runs hit the 6k-token cap) | 3/3 (61s; quoted test 3/3) | 2/3 (37s; one trial mapped a criterion wrong, accuracy 0.8) |

Observations:
- **No tool-call errors (unknown tool / missing argument) and no policy violations (`tests/` modified) in any T1 trial**, and no model claimed `DONE:` while tests still failed.
- **Qwen and GLM are indistinguishable at this difficulty** (9/9 each). The eval is too easy to separate them on accuracy; speed is the only difference seen (GLM faster on T2/T3, Qwen faster on T1). The "Qwen is the coder" assignment is user-supplied and is **not confirmed or contradicted** by this data.
- **gpt-oss-20b** is fastest on the coding loop but shows two small reliability lapses (one invalid delegate call, one review finding that did not quote the test name despite the buried rule). With n=3 these could be noise.
- **Gemma 4 12B is by far the slowest and most verbose**: 747s for one successful coding trial, and two trials ran out of the 6,000-token cap mid-run (counted as failures). It passes the short structured tasks, but it does not fit "snappy" for multi-step agent loops at this output cap; a higher cap might turn those into passes at even greater latency.
- **Setup correction found by this run:** the first attempt returned HTTP 500 for GLM (and Gemma right after) because GLM hit `CUDA error: out of memory` on a real 10k-token prompt at `--n-cpu-moe 23`. It had fit at startup and on tiny prompts. Raised to **25**, after which it ran cleanly. This is the same kind of headroom failure as gpt-oss earlier (§2): **the benchmarked minimum `--n-cpu-moe` is not a safe serving value; add about 2 layers of headroom and verify with a realistic long prompt.** GLM's speed at 25 was not re-benchmarked (expect slightly lower decode than the 23 figures). The 9 invalid rows from that failed run were discarded and GLM/Gemma/gpt-oss were re-run.
- **Limits:** n=3 per cell (pass rates move in steps of 33%); one scenario per task; single-turn-family tasks only; RoboCo's real prompts, tool manifests and MCP gateway were not used.

---

## 7. Provisional roster (speed data plus smoke test plus §6b; accuracy only partly verified)

| Role group | Model |
|---|---|
| Developers and code-heavy roles | **Qwen3.6 REAP** (best decode at depth, passed both turns) |
| PMs, planning, Board, generalist roles | **GLM-4.7 REAP** (passed both turns; slower at depth) |
| Documenters (short, low-risk generation) | **gpt-oss-20b** (fastest on T1/T2, but see §10.7: it approved broken work in QA trials; **not for QA/review**) |
| Background local LLM (RAG/drafting, single-shot, not agent loops) | **Gemma 4 12B** (fastest prefill, but slow and verbose in multi-step loops: 1/3 on T1) |

Revised from the first draft after §6b: gpt-oss-20b moved up (its early out-of-order call was not reproduced in the larger eval) and Gemma moved out of agent-loop roles.

---

## 8. Not done / next steps

1. **A harder eval still**: §6b did not separate Qwen from GLM (9/9 each). Needs more trials, multi-file refactors, ambiguous requirements, longer contexts (30k+), and RoboCo's real prompts/manifests through the actual agent stack.
2. Test **`--moe-cache`** (fork feature, not in the tutorial).
3. Find the **f16-vs-turbo crossover depth** for GLM (only 0, 16k and 32k measured) and measure **64k-deep** speed (too slow to run; fit only).
4. Re-benchmark gpt-oss at `--n-cpu-moe 5`; ubatch sweeps were only run on GLM (and Gemma); Qwen and gpt-oss used ubatch 1024 without a sweep.
5. ~~Connect RoboCo's `self_hosted` mode~~ **Done, see §10.** Still to do: watch real agent runs for failures and tune per §10.6.
6. Report the `llama-bench` MLA null-deref upstream (draft only; not filed).
7. Optional: Pi coding agent and Tailscale from the tutorial (not started).

## 9. Artifacts

- Fork and build: `~/llama-cpp-turboquant/build/bin/` (`llama-server`, `llama-bench`).
- Presets: `~/.config/llama-server/models.ini`.
- Raw results and scripts (temporary scratchpad): `/tmp/claude-1000/-home-wazouchski/722f669a-5306-4843-a56e-98efb1e2c8b7/scratchpad/` (`glm-*.txt`, `qwen-suite.out`, `toolcall.py`, `fit2.sh`). May be cleaned up.
- Memory note: `~/.claude/projects/-home-wazouchski/memory/llama-turboquant-roboco-setup.md`.

---

## 10. RoboCo wiring (2026-09-30)

Live RoboCo (`~/AI/roboco`, Docker stack from `docker-compose.registry.yml`) now routes the local-model agents to the llama router. All changes were made through RoboCo's own service layer (`ModelRoutingService.upsert_assignment`) from inside the orchestrator container, not raw SQL and not the authenticated HTTP API.

### 10.1 Final routing (resolved per agent via `resolve_for_agent`, including the spawn-time reachability probe)

| Agents | Provider / model | Notes |
|---|---|---|
| be/fe/ux-dev-1/2 (6) | **self-hosted `qwen36-reap`** (Qwen3.6 REAP) | coder role |
| be/fe/ux-qa (3) | **self-hosted `glm47-flash-reap`** (GLM-4.7 REAP) | QA |
| be/fe/ux-doc (3) | **self-hosted `gpt-oss-20b`** | low-risk generation |
| intake-1, secretary-1 | **self-hosted `glm47-flash-reap`** | via the GLOBAL fallback, changed from gpt-oss at the user's request |
| ux-pr-reviewer, cell-pr-reviewer-2 | ~~self-hosted `glm47-flash-reap`~~ **now Anthropic `sonnet`** (see §11) | changed 2026-10-01 after the local reviewer could not satisfy the gateway's finding schema (§10.15) |
| main-pm, be-pm, fe-pm, ux-pm, be-pr-reviewer, fe-pr-reviewer, pr-reviewer-1 | Anthropic `sonnet` (Claude Sonnet 5) | **untouched** (agent-slug pins) |
| auditor, head-marketing, product-owner | Anthropic **`sonnet-5.5`** (`claude-sonnet-5-5`) | changed from Opus at the user's request; briefly on `sonnet` until the §10.5 fix was deployed |
| high-complexity developer tasks (`developer:high`) | Ollama Cloud `kimi-k2.7-code:cloud` | **left as is** (see §10.6) |

Rows changed: ROLE `developer`, `qa`, `pr_reviewer`, `documenter` and GLOBAL (all to the LOCAL provider), plus AGENT_SLUG `auditor`, `head-marketing`, `product-owner` (Opus to Sonnet). Not touched: the OpenRouter ROLE rows for `auditor`, `cell_pm`, `head_marketing`, `main_pm`, `product_owner` (shadowed by the slug pins), the `developer:high` row, and the provider base URL (`http://172.20.0.1:1234`, unchanged). End-to-end check from inside the orchestrator container: an Anthropic-format tool call to each of the three models using the exact model names above returned a correct `run_tests` tool call (5 to 8 s). **No real RoboCo agent run has been observed yet.**

### 10.2 Router plumbing and what it depends on

- **Port 1234, bridge-only.** From inside the containers the host firewall (nftables, root-only to inspect) allowed port 1234 (RoboCo's existing LM Studio slot) but **silently dropped 8080 and 11434**. The router therefore serves on `172.20.0.1:1234`, the RoboCo network's gateway address, and is **not listening on the LAN or any other interface**. It runs as the systemd user service `llama-router` (`Restart=always`).
- **The router must stay up.** `resolve_for_agent` probes `{base_url}/api/tags` and then `/v1/models` at every spawn; if the server is unreachable it silently falls back to the legacy **Anthropic** route. A dead router therefore means local agents quietly run on Claude and spend Claude quota. (The probe works against the router: `/api/tags` returns 404 but `/v1/models` returns 200.)
- **Keep LM Studio's server off** while the router runs (same port) and do not load models in LM Studio (they would compete for the 3060's VRAM).
- **Concurrency:** `ROBOCO_LOCAL_MAX_CONCURRENT=1` and router `--models-max 1`, so local agents run one at a time and each change of model costs a 10 to 17 s load.
- The auto-mode safeguard in the assistant environment blocked one step (moving the service to port 1234) as "unauthorized persistence" until the user explicitly approved it.

### 10.3 Saved presets and rollback

- llama-server presets: `~/.config/llama-server/models.ini` (+2 headroom: Qwen 22, GLM 25, gpt-oss 6; Gemma dense, unchanged).
- RoboCo routing presets (Settings, AI routing card, presets): **`pre-local-router-2026-09-30`** (20 rows, the state before any change) and **`local-llama-router-2026-09-30`** (21 rows, the state after). A third preset, **`local-llama-router+sonnet55-2026-10-01`** (21 rows), captures the final state with the Sonnet 5.5 pins. Applying the `pre-` preset restores everything, including the three Opus pins.
- Plain-text backup: `~/AI/roboco-routing-backup-2026-09-30.csv`.
- To stop using the router: apply the `pre-` preset and run `systemctl --user disable --now llama-router`.

### 10.4 Incident during the work: a silently misfiled assignment (repaired in about 20 s)

I first set the three Opus-pinned agents to `sonnet-5.5`. `upsert_assignment` looks a name up in the static model catalog and, if it is not found and a LOCAL provider is seeded, **files it under the LOCAL provider without any warning**. Because the running orchestrator does not know `sonnet-5.5`, all three rows were saved as `provider=local`. I caught this in the verification output and repaired the rows to `sonnet` immediately. No agent containers were running, so nothing spawned on a wrong route. (RoboCo's own routing code notes that unreachable local servers also degrade silently to Anthropic, so this kind of error would have been invisible in production.)

### 10.5 Adding `claude-sonnet-5-5` properly (proposal, then APPLIED 2026-10-01; see §10.9)

**Why it does not work today**
1. `sonnet-5.5` exists only as **uncommitted one-line local edits** in `roboco/models/runtime.py` (`MODEL_MAP`) and `roboco/models/llm_catalog.py` (`_ANTHROPIC_DISPLAY`), saved at 18:37 EDT. The running image was built at 04:35 EDT and the container created at 12:55 EDT. The stack (`docker-compose.registry.yml`) does not mount the source tree, so the deployed code does not contain them: its `MODEL_MAP` has only `opus`, `sonnet`, `haiku`.
2. The edit maps the alias to **`claude-sonnet-5.5` (with a dot)**. The model ID list available to the assistant gives Sonnet 5.5 as **`claude-sonnet-5-5` (hyphen)**, so the dotted ID would likely be rejected by the API. **Verify this against the real CLI before deploying.**
3. Unknown names are silently misfiled under LOCAL (§10.4).

**Proposed change set**
1. `roboco/models/runtime.py`: `"sonnet-5.5": "claude-sonnet-5-5"` (keep the `sonnet-5.5` alias).
2. `roboco/models/llm_catalog.py`: keep `("sonnet-5.5", "Claude Sonnet 5.5")`. The catalog entries are derived from `MODEL_MAP`, so both edits are needed for it to appear in the routing dropdown and be assignable.
3. `roboco/billing/pricing.py`: add an explicit row for `claude-sonnet-5-5` with the price from Anthropic's pricing page (the assistant does not know it and did not assume one). The table appears to use longest-prefix matching, so without a row the new ID would silently be billed at the Sonnet 5 rate ($2.00/$10.00), because `claude-sonnet-5-5` starts with `claude-sonnet-5`. Confirm the matching rule when editing.
4. Tests: extend `tests/unit/models/test_model_map.py` (alias resolves to the ID; the ID is priced by its own row, not the prefix) and add a catalog test that `sonnet-5.5` resolves to the Anthropic provider, not LOCAL.
5. Guard against the silent misfile: make `upsert_assignment` raise (or at least warn) when a name contains `claude`, `sonnet`, `opus` or `haiku` but is not in the catalog, instead of assigning it to LOCAL.
6. Verify the ID with one tiny call from an agent container (`claude --model claude-sonnet-5-5 -p "ok"`; it uses a little Claude quota) before any deploy.
7. Deploy (per the 2026-09-30 handoff): run the unit tests in a throwaway container, then `docker compose -f docker-compose.registry.yml build orchestrator` and `up -d --no-deps orchestrator dispatcher`, in a quiet window with no agents running. The rebuild also ships the other uncommitted local edits that are already in the tree (respawn breaker and similar).
8. Then upsert `sonnet-5.5` for the chosen agents through the service layer, check with `resolve_for_agent` that each shows `anthropic / sonnet-5.5`, and save a new RoboCo routing preset.

The repo is upstream software with no private fork yet (per the handoff), so keep this as a local branch or a patch file rather than assuming a commit or PR has a home.

### 10.6 `developer:high` left on Ollama Cloud; alternatives if it fails too often

Kept as `kimi-k2.7-code:cloud` per the user's decision. If failures pile up, candidates from RoboCo's own Ollama Cloud catalog, in suggested order: **`glm-5.3:cloud`** (GLM 5.3; RoboCo's own default and the same family as the local GLM), then **`minimax-m3:cloud`**, then `nemotron-3-ultra:cloud`. The most reliable option would be Claude Sonnet, at the cost of Pro-plan quota. None of these were tested.

### 10.7 Extra finding: gpt-oss approves broken work in QA

After the +2 headroom change, gpt-oss-20b failed 3 of 6 runs of the QA review task, and it was more reliable (3 of 3) in the main eval. In the two failures that were inspected it called `pass_review` with no flagged criteria, approving work whose test output showed a failure; the third failure was not inspected. A false approval is the worst failure a reviewer can have, so **gpt-oss is not used for QA or review**. GLM-4.7 REAP passed the same task in 11 of 11 runs (3 in the main eval plus 8 follow-ups). Small samples: this is a strong warning, not a precise rate.

### 10.8 Open risks and next checks

- **No real agent run yet.** Watch the first local dev, QA and documenter tasks: tool-call errors, stalls, cold model swaps, and any silent fallback to Anthropic (check `resolve_for_agent` or the spawn logs).
- Single-slot local execution makes the company slow by design (10 to 17 s per model swap, agents one at a time).
- Memory and headroom: the +2 layers were checked with a 10.7k-token prompt, but real agent prompts may be larger (RoboCo prompts reach 26k+ tokens per the handoff). If you see CUDA out-of-memory errors in `journalctl --user -u llama-router`, add 1 to 2 more layers for that model.

### 10.9 Sonnet 5.5 fix: what was actually applied (2026-10-01) and the deployment record

The user confirmed the model ID is `claude-sonnet-5-5` and supplied the Sonnet 5.5 prices ($2.00 input, $10.00 output, $0.20 cache read, $2.50 5-minute cache write, $4.00 1-hour cache write). The proposal in §10.5 was executed as follows (an extra CLI call to verify the ID was skipped because the user confirmed it; the first real Sonnet 5.5 agent run will be the live check).

**Source changes** (uncommitted, in `~/AI/roboco`; the repo is upstream with no private fork yet):
1. `roboco/models/runtime.py`: `"sonnet-5.5": "claude-sonnet-5-5"` (was the dotted `claude-sonnet-5.5`).
2. `roboco/billing/pricing.py`: new row `("claude-sonnet-5-5", 2.00, 10.00, 0.20, 2.50)`. Lookup is longest-substring, so this row now wins over the `claude-sonnet-5` row (which would otherwise have matched and billed 5.5 at Sonnet 5 rates). The table has a single cache-write column, so the $4.00 one-hour rate is **not modeled** (5-minute rate used). Observation, not changed: the existing Sonnet 5 row carries `0.50` as its cache-write rate, a "25% of input" convention, whereas the Sonnet 5.5 figure supplied is 125% of input; the Sonnet 5 row was not verified and was left alone.
3. `roboco/services/llm.py`: `upsert_assignment` now raises `ValueError("Unknown Claude model ...")` for a name containing `claude`, `sonnet`, `opus` or `haiku` that is not in the catalog (via `_CLAUDE_NAME_FRAGMENTS`), instead of silently filing it under the LOCAL provider. Explicit LOCAL assignments (`provider_type_override`) are unaffected. Verified live against the deployed system.
4. `roboco/models/llm_catalog.py`: the `sonnet-5.5` display entry from the earlier local edit is kept as is.

**Tests added** (8, all passing): `test_model_map.py` (alias resolves to the hyphenated ID; catalog entry routes to Anthropic), `test_pricing.py` (own price row; cache-write is $2.50, not the Sonnet 5 $0.50), and the new `tests/unit/services/test_llm_unknown_claude_guard.py` (4 parametrized names refused). In a throwaway container with the new source mounted read-only: 168 related tests ran, 162 passed, 6 failed. The 6 failures are the `test_interactive_provider_guard` cases that fail on database connection and were already listed as failing before these changes in the 2026-09-30 handoff.

**Deploy:** `docker compose -f docker-compose.registry.yml build orchestrator`, then the orchestrator and dispatcher were recreated on the new image (`MODEL_MAP` in the running container now contains `sonnet-5.5 -> claude-sonnet-5-5`). Assignments upserted for `auditor`, `head-marketing`, `product-owner`; `resolve_for_agent` shows `anthropic / sonnet-5.5` for all three, `main-pm` still `anthropic / sonnet`, and the local routes unchanged.

**Incident during the deploy (registry env):** the first recreate sent the dispatcher into a restart loop with `Failed to pull image ghcr.io/wazouchski/roboco-agent-base:latest ... denied`. `.env` gained `ROBOCO_REGISTRY=ghcr.io/wazouchski` at 15:11 on 2026-09-30, after the old containers were created, and recreating picked it up; the agent images that exist locally are under the default `ghcr.io/rennf93` namespace. Fix: both services were recreated with `ROBOCO_REGISTRY=ghcr.io/rennf93` set **for that command only** (`.env` untouched), which restored the previous behavior. **Consequence:** any future recreate that reads `.env` as is will hit the same loop; either keep passing `ROBOCO_REGISTRY=ghcr.io/rennf93`, or change/remove that `.env` line, or locally tag the needed agent images under `ghcr.io/wazouchski/` (only an openrouter image is tagged that way today). Nothing was spawning during this window.

**State when the work finished:** orchestrator and dispatcher healthy on the new image; `llama-router` active and reachable from the containers on `172.20.0.1:1234`; no agent containers running because the CEO's maintenance pause (engines, dispatch and board programs, set 2026-09-30 22:10 UTC, auto-expiring at 2026-10-01 02:10 UTC) is still on. Three tasks are queued (2 awaiting QA, 1 awaiting PR review). The first local agent runs will begin once the pause is lifted or expires.

### 10.10 First local runs and resetting parked (breaker-tripped) tasks (2026-10-01)

**After unpausing** the CEO maintenance pause: the first agent to start was the `auditor` on `sonnet-5.5` (exit 0 after 27 s), but the three tasks that use local models (`907d2958` awaiting QA on `be-qa`, `a83c7959` awaiting QA on `ux-qa`, `0a4af768` awaiting PR review on `ux-pr-reviewer`) never spawned: the dispatcher's **respawn breaker** was tripped for each (count 4 over the threshold of 3, `notified=true`), and the tasks carried a durable marker `stalled_reason = breaker_tripped`.

**Reset performed (with the user's approval):** stop the dispatcher so it flushes its in-memory tracker, delete exactly those three rows from `respawn_tracker` (by agent slug and task id), start the dispatcher (the other 34 breaker rows were untouched). Result: at 00:47:54 UTC the dispatcher spawned `ux-pr-reviewer` on `glm47-flash-reap`, the router loaded the model at 20:47:56 EDT and was proxying the agent's requests 3 s later, confirming the full local path (RoboCo to router to model). The `be-qa` task waits its turn (one local agent at a time; its breaker count stays at 1 while it waits). **Correction:** the `ux-qa` task was *not* just waiting; see §10.11. The `breaker_tripped` marker on the tasks is **not** a gate (the dispatcher spawned despite it); it clears itself on genuine forward progress.

**Why a plain re-spawn was not possible and how the breaker resets** (from `roboco/runtime/engines/dispatch_breaker.py`): the counter resets when the task's status changes to a status that agent has **not** yet seen on that task (and the stalled marker is cleared at the same time). A status that **was** already seen gets only a bounded number of "revisit" resets, after which strikes accrue again. There is also a tracing-gap reset. In the 2026-09-30 local edits the breaker keeps an exhausted task parked until a real state or assignment change, or an explicit reset.

**Dashboard (panel) findings, from reading the panel code; not exercised in a browser:**
- There is **no dedicated "reset breaker" button** and no API route for it (the handoff said the same).
- Overview, "Stalled / Needs You" panel (`stalled-needs-you-panel.tsx`): read-only list of tasks the breaker gave up on, each linking to its task page. The task page shows a red **stalled** chip with the reason.
- On the task page the header has a **status control**. In-band transitions run the real lifecycle action; any other status is a CEO "god-mode" **forced override** (`PATCH /tasks/{id}` with `force: true`, audited, no side effects such as PR merges). A status change is what the breaker reacts to, so this is the dashboard lever, but it moves the task through the workflow and should be used deliberately (for a QA task, forcing it elsewhere and back is a revisit that spends the bounded revisit-reset budget; the budget value was not looked up and this route was not tested).
- Pause/Resume exist on the task page (`resume` is paused to in_progress), but resuming changes the task to `in_progress`, not back to `awaiting_qa`, so it is not a clean breaker reset for a QA task.
- The only method verified to clear the count outright is the database procedure above (stop dispatcher, delete the `respawn_tracker` rows, start dispatcher; the order matters because the dispatcher flushes on shutdown).

### 10.11 Reset script, and why one reset did not stick

**Script:** `~/AI/roboco-tools/reset-breaker.sh` (outside the repo). It does the verified procedure in one command and touches **only** the `respawn_tracker` table:
- `reset-breaker.sh --list` shows every tripped breaker row with its task title and status (read-only).
- `reset-breaker.sh <task-id-prefix>...` resets all agents' rows for those tasks; `--all-tripped` resets every tripped row; `--yes` skips the confirmation.
- Order of operations: confirm, stop the dispatcher (it flushes its tracker to the database), delete the rows, start the dispatcher, wait for healthy.
- Safety: task-id prefixes are validated (4 to 36 hex characters or dashes), it prints exactly what it will delete and asks first, and a "no" answer aborts before anything is stopped. Tested: help text, bad input (rejected), `--list`, a prefix matching nothing, and the abort path. The destructive path itself was run by hand earlier (§10.10), not through the script.

**What the first reset showed:** only some tripped tasks are tripped for a reason a reset can fix.
- `ux-pr-reviewer` on `0a4af768`: reset worked; the agent ran on `glm47-flash-reap`.
- `be-qa` on `907d2958`: reset worked; it is waiting for the single local slot ("Spawn skipped: local concurrency cap reached"), and the breaker count is not incremented while it waits.
- `ux-qa` on `a83c7959`: **re-tripped within about 2 minutes.** The dispatcher tries to spawn it on every tick and is refused with "waiting on non-terminal dependency `be6971da`" (task "Asset Integrity Audit", currently `awaiting_documentation`). Each refused spawn counts as an unproductive respawn, so the breaker trips again (count 4, threshold 3). Resetting it is pointless until the dependency reaches a terminal status. `be6971da` is itself stalled because its documenter row (`ux-doc`) is tripped. So the chain is `a83c7959` (QA) blocked by `be6971da` (waiting on documentation by `ux-doc`).

**Other tripped rows at the time of `--list`** (not reset; none were requested): documenters `be-doc` on `305f4cda`, `ux-doc` on `be6971da`, `fe-doc` on `c5395dbf` and `f192fc70` (these run on local `gpt-oss-20b`); PMs on `46781ab0`, `74f865c9`, `539560ca` (Sonnet); board-program cycles `510135fe`, `5ec7be4c`, `b0071f75` (Sonnet 5.5 agents). Resetting the four documenter rows would exercise `gpt-oss-20b` and is the way to start unblocking the `be6971da` chain, but whether those tasks tripped for a dependency reason like the one above, or for an older local-model problem, was not investigated.

### 10.12 First real local runs, the runaway-reasoning incident, and thinking-budget tuning (2026-10-01)

**First card moved by a local model.** After the documenter reset, the dispatcher gave the single local slot to `be-qa` (GLM) on task `907d2958` ("Standardize root Makefile ..."). At 01:48:31 UTC the task went `awaiting_qa` to `awaiting_documentation` (QA passed), about 16 minutes after the spawn. In the same period an `auditor` on `sonnet-5.5` also ran. The four documenter tasks and `ux-pr-reviewer` were queued behind the local concurrency cap (each logged "Spawn skipped: local concurrency cap reached", which does not add a breaker strike).

**Incident: runaway reasoning.** `ux-pr-reviewer` (GLM, task `0a4af768`) spent over 15 minutes inside one response: the agent's "thinking" estimate passed 50,000 tokens and kept growing at about 60 tokens/s. The router logs only the start of a streaming request, so it looked idle. Because only one local agent runs at a time, this blocked every other local card. My earlier evals used small tasks, where these models think briefly, so they did not show it.

**Fix (approved):** `models.ini` now caps thinking: `reasoning-budget = 8192` for `qwen36-reap` and `glm47-flash-reap`, and `reasoning-effort = low` for `gpt-oss-20b` (a backup is at `models.ini.bak-2026-10-01`). The router was restarted and the flags are visible in `/v1/models`. Check: a deliberately thinking-heavy prompt ended after 8,290 output tokens (8,192 thinking plus the answer) with a clean `end_turn`. Side effect seen in that test: the truncated reasoning made GLM ignore a "give ONE sentence" instruction and answer with code, so a cap can degrade answers. The reviewer survived the restart (it retried) and resumed with bounded passes, but a full review is still slow (about 2 minutes per step, 36+ minutes in with no review output when last checked).

**Per-request control** (via the Anthropic `thinking` field on `/v1/messages`): `{"type":"enabled","budget_tokens":N}` **is honored** (512 gave 581 output tokens); `{"type":"disabled"}` is **not** honored (the model kept thinking). Turning thinking off needs a server-side setting (`--reasoning off`, or chat-template kwargs); RoboCo's agents do not set a per-request budget themselves, so the preset value is what applies.

**Budget eval (GLM, QA-review task, per-request budgets): inconclusive and stopped early.** It ran concurrently with live agents, so timings are meaningless (29 to 235 s per call for the same work) and 2 of 7 calls failed with HTTP 500 when the router swapped models for the next queued agent. Valid trials: budget 1024: 3 of 3 correct (`fail_review` flagging only AC-3, test name quoted); budget 4096: 2 of 2 correct; 8192 was not reached. Small, contended sample: no sign that a lower budget hurts the QA verdict, but also no speed measurement. **A proper comparison needs exclusive use of the GPU** (a maintenance pause, or an empty queue); the router's `--models-max 1` means any concurrent eval can also force a model swap that disrupts a live agent.

**Documenter reset (done):** the four documenter rows were reset with the script (`305f4cda:be-doc`, `be6971da:ux-doc`, `c5395dbf:fe-doc`, `f192fc70:fe-doc`). Two defects in `reset-breaker.sh` were found and fixed along the way: a multi-argument call built invalid SQL (separator bug; it failed before anything was stopped), and the script gained a `<task-prefix>:<agent>` form so a reset can target one agent's row instead of every agent on the task. **Stopping the dispatcher also stops any running agent** ("Agent stopped" is logged), so run the reset when no agent is mid-task.

**State at the end of this session's work:** router `llama-router` active with the thinking caps; dispatcher healthy; `be-qa` finishing, then the queued local tasks run one at a time (documenters on `gpt-oss-20b`, `ux-pr-reviewer` on GLM). `a83c7959` (`ux-qa`) is still blocked by its dependency chain (`be6971da`, which waits on documentation).

**Suggested next steps:** (1) let the documenter queue run and watch `gpt-oss-20b` on real work; (2) when the GPU can be used exclusively, compare thinking budgets (for example 2048, 4096, 8192) on GLM and Qwen across the QA, coding and delegation tasks, then set the per-model budget from the data (a moderate 4096 for GLM looks plausible from the small sample but is not established); (3) if reviews stay too slow, consider server-side thinking off for the review roles or moving PR review to a Sonnet pin.

### 10.13 Exclusive-GPU thinking-budget sweep (2026-10-01)

**Setup.** With the user's authorization (they were away) I paused the `dispatch` and `engines` maintenance scopes through RoboCo's pause service (stamped `claude-code (user-authorized)`, 3-hour auto-expiry), waited for the running agent to finish on its own (nothing was killed), and ran `budget_sweep.py` against the router with the GPU to itself. The sweep refuses to start, and aborts, if any `roboco-agent*` container is running. Per-request thinking budgets via the Anthropic `thinking.budget_tokens` field (§10.12), models `glm47-flash-reap` and `qwen36-reap` at budgets 2048 / 4096 / 8192, the three §6b tasks (3 coding loops, 4 QA reviews, 3 delegations per setting): **60 trials, 0 server errors**, about 35 minutes. Afterwards the two paused scopes were resumed and the dispatcher immediately started spawning again. Only the thinking budget varied; this was a per-request override, so no restarts were needed.

| Model | Budget | T1 coding | T2 QA review | T3 delegation |
|---|---|---|---|---|
| GLM-4.7 REAP | 2048 | 2/3 (99 s) | 4/4 (12 s) | 3/3 (22 s) |
| GLM-4.7 REAP | 4096 | 3/3 (62 s) | 4/4 (12 s) | 2/3 (21 s) |
| GLM-4.7 REAP | 8192 | 2/3 (48 s) | 4/4 (15 s) | 3/3 (24 s) |
| Qwen3.6 REAP | 2048 | 2/3 (46 s) | 4/4 (20 s) | 3/3 (71 s) |
| Qwen3.6 REAP | 4096 | 2/3 (32 s) | 4/4 (22 s) | 2/3 (50 s) |
| Qwen3.6 REAP | 8192 | 3/3 (28 s) | 4/4 (16 s) | 3/3 (31 s) |

(Average wall time per trial in brackets.) Aggregated over both models and all tasks: budget 2048 passed 18/20 (avg 42 s, 1,512 output tokens), 4096 passed 17/20 (32 s, 1,390 tokens), 8192 passed 19/20 (26 s, 1,092 tokens).

**Findings.**
- **No QA false approvals at any budget** (16/16 `fail_review` flagging only AC-3, with the failing test named).
- **A lower budget was not faster on these tasks.** Time and output tokens *fall* as the budget rises. Likely cause (not investigated): truncated thinking leads to longer answers or extra turns. The largest effect was Qwen on delegation: 71 s at 2048 against 31 s at 8192.
- **No clear accuracy effect.** The 6 failures are spread across budgets with n=3 per cell, so differences are noise. All 4 coding failures were *only* the buried one-line `DONE:` format rule (the code fix itself was correct each time); both delegation failures were one criterion mapped to the wrong cell.
- **Decision: no change.** Keep `reasoning-budget = 8192` for both models. The budget is a safety cap against runaway thinking (§10.12), not a speed lever on tasks like these. Caveat: these tasks rarely think long (an average of 400 to 3,400 output tokens per trial), so this sweep does not show what the cap does on a real PR review, which is where the runaway happened.
- gpt-oss-20b (server-side `reasoning-effort`, not a per-request budget) and thinking fully off were not swept.

**Open: PR review on `ux-pr-reviewer` (GLM) did not advance the card.** After the cap, the agent ran 01:54 to 01:59 UTC and exited gracefully (exit 0), but task `0a4af768` stayed `awaiting_pr_review` (it carries the old `breaker_tripped` marker) and the dispatcher spawned it again at 02:34. The dispatcher also keeps logging `get_pr_ci_status check-runs non-2xx` (HTTP 403) for the `atomic-frontier` project, which may mean the reviewer cannot read PR CI status and so cannot reach a verdict; that is a guess, not verified. If the reviewer keeps ending without a verdict, each respawn counts a strike and the breaker will trip again after about 3 more.

### 10.14 Root cause of the stalled PR review: the project's GitHub token cannot read check-runs (2026-10-01)

**Finding (verified, read-only).** The PR-review gate in RoboCo refuses `pr_pass` unless CI on the PR's head commit is green. It reads GitHub check-runs (`GitService.get_pr_ci_status`). For project `atomic-frontier` (repo `wazouchski/atomic-frontier`, PR #9, task `0a4af768`) GitHub answers that request with **HTTP 403, "Resource not accessible by personal access token"**, and its response header `x-accepted-github-permissions: checks=read` names the missing permission. The project's stored token is a **fine-grained personal access token** (valid until 2026-12-28 21:31 UTC; the token itself was never printed). The check ran through RoboCo's own code in the orchestrator container, using the stored token read-only; only the status, message and permission headers were read.

**Why it looks like a model problem but is not.** `_fetch_check_runs` turns any non-2xx other than 404 into the state `error`; `_ci_status_guard` then blocks `pr_pass` on `error` and tells the reviewer it is a retryable platform blip ("wait and call pr_pass again"). So the reviewer can never pass the PR, ends its run without a verdict, gets respawned (one more breaker strike each time), and eventually trips the breaker. The dispatcher polls the same endpoint every minute, which is why `get_pr_ci_status check-runs non-2xx` (403) appears in its log continuously. The guard is in the gateway shared by **all** PR reviewers, so it should also affect the Sonnet-pinned reviewers on this project; that part is inferred from the code, not observed.

**CORRECTION (2026-10-01): the first fix advice above was wrong.** The assistant first advised setting a "Checks: Read-only" permission on the fine-grained token. The user could not find it, and GitHub's own documentation confirms why: **fine-grained personal access tokens have no Checks permission and do not support the Checks API at all**; for a private repository the check-runs endpoint needs a **classic PAT with the `repo` scope** (or a GitHub App). Evidence gathered with RoboCo's own code and the stored token (read-only GETs, status codes only):

| Request (repo is **private**) | Result |
|---|---|
| repo metadata | 200 |
| check-runs, with token (what RoboCo uses) | **403** "Resource not accessible by personal access token" |
| check-runs, without a token | 404 (private repo) |
| commit status | 403 (token lacks "Commit statuses: Read") |
| Actions workflow runs for the commit | 403 (token lacks "Actions: Read") |
| Actions workflows | 403 |

RoboCo has no setting that disables or relaxes this gate (searched config and policy), and its logic reads check-runs first and gives up with `error` on the 403 before it ever looks at workflows, so **adding Actions or Commit-statuses permissions to the token would not help without a code change**.

**Options (decision belongs to the user; none applied):**
1. **Classic PAT with `repo` scope** for this project in RoboCo. Fastest and needs no code change, but it grants far broader access (read/write on every repository the user can reach) than the current fine-grained token.
2. **Patch RoboCo to fall back to GitHub Actions runs** when check-runs returns 401/403 (derive success, failure or pending from the head commit's workflow runs), and add "Actions: Read-only" (and optionally "Commit statuses: Read-only") to the fine-grained token. Least privilege and keeps the gate meaningful, but it is real development in `roboco/services/git.py` (the source already has a note that a statuses fallback may be needed), plus tests, an image rebuild and a redeploy (with the registry-env gotcha from §10.9), and CodeQL scope-gating is keyed on check-run names and would need care.
3. **Stopgap: make the gate fail open with an evidence note** when GitHub answers 401/403 (same treatment as `no_ci_configured`). Quickest code change, but PRs could then pass without verified CI; the CEO still performs the only merge to `master`.

**Afterwards (once the gate can read CI).** The reviewer will see real CI state: `success` lets `pr_pass` through, `failure` routes to `pr_fail`, `pending` is retried. If `ux-pr-reviewer` has already re-tripped the breaker on `0a4af768` by then, reset only that row with `~/AI/roboco-tools/reset-breaker.sh 0a4af768:ux-pr-reviewer`, when no agent is mid-task (stopping the dispatcher also stops running agents).

### 10.15 Token fix confirmed; what the local agents have actually done (2026-10-01, evening)

**CI gate fixed.** The user replaced the project's git token with a classic PAT (`repo` scope). Read-only check afterwards: GitHub returns 200 on check-runs for PR #9; the repo has no CI configured (zero check-runs), so RoboCo classifies the PR as `no_ci_configured`, which passes `pr_pass` through with a note. (The user's other token, with `workflow` and `write:packages`, is for GHCR and was deliberately not reused.) Decision recorded: Atomic Frontier **stays a single project**; a project per department is noted by the user for a *new* product (the YouTube creator's walkthrough is a month old and RoboCo may have changed).

**Cards moved by local models so far:**
- `907d2958`: `be-qa` (GLM) passed QA, `awaiting_qa` to `awaiting_documentation` (about 16 min of agent time, after the thinking cap).
- `305f4cda`: `be-doc` (**gpt-oss-20b**) wrote the documentation in about 2 minutes (container exit 0), then the Sonnet `be-pm` completed the task. First completed card with a local documenter.
- `0a4af768` (PR review) did **not** advance, see below.

**New blocker: the local GLM PR reviewer cannot get a verdict past the gateway.** The audit log (`gateway.rejected`) shows `ux-pr-reviewer` reaching a `pr_fail` decision but being rejected on every attempt: twice for `malformed finding: actual - String should have at most 300 characters`, once for `finding criterion not found on this task` (it quoted the acceptance criteria in its own words instead of matching them), and once `not_authorized: not assigned to you` after it lost its claim (the task is repeatedly reassigned, "Task reassigned" about every 32 s, and I had also restarted the dispatcher, which releases claims). Each run exits 0 without a verdict and costs a breaker strike. This is the structured-envelope fidelity limit RoboCo's own docs describe for weaker models (§5), now seen on a real review: the model reasons about the PR but cannot produce a finding that satisfies the schema.

**Not traced:** `be-pm` (Sonnet) took `46781ab0` from `pending` to `claimed` to `in_progress`, then went idle and the task became `paused` (03:26:34 UTC). Whether that is normal PM waiting behavior was not determined. The earlier dispatcher restart (my breaker reset, 03:22:42) also stopped `be-pm` mid-claim; it was respawned 3 minutes later.

**Proposed, then APPLIED (see §11):** pin `ux-pr-reviewer` and `cell-pr-reviewer-2` to Sonnet (the other three PR reviewers already are), so every PR gate uses a model that can satisfy the schema; keep GLM for QA, where it has passed. Alternative: leave them and let the breaker park the task.

---

## 11. Agent roster and models (live routing as of 2026-10-01 03:33 UTC)

Generated from RoboCo's routing service (`resolve_for_agent`, including the router reachability probe), not typed by hand. **26 agents** (RoboCo's docs count 25, plus the floating second cell PR reviewer `cell-pr-reviewer-2`); the CEO and the internal `system` actor are excluded. "Assigned via" shows which rule decided it: an agent pin beats a role row, which beats the global default.

| Agent | Role | Team | Provider | Model | Runs on | Assigned via |
|---|---|---|---|---|---|---|
| product-owner | product_owner | board | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| head-marketing | head_marketing | board | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| auditor | auditor | board | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| intake-1 | prompter | board | local | `glm47-flash-reap` | local GPU (llama router) | global default |
| secretary-1 | secretary | board | local | `glm47-flash-reap` | local GPU (llama router) | global default |
| cell-pr-reviewer-2 | pr_reviewer | board | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| pr-reviewer-1 | pr_reviewer | board | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| main-pm | main_pm | main_pm | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| be-pm | cell_pm | backend | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| be-dev-1 | developer | backend | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| be-dev-2 | developer | backend | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| be-qa | qa | backend | local | `glm47-flash-reap` | local GPU (llama router) | role `qa` |
| be-doc | documenter | backend | local | `gpt-oss-20b` | local GPU (llama router) | role `documenter` |
| be-pr-reviewer | pr_reviewer | backend | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| fe-pm | cell_pm | frontend | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| fe-dev-1 | developer | frontend | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| fe-dev-2 | developer | frontend | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| fe-qa | qa | frontend | local | `glm47-flash-reap` | local GPU (llama router) | role `qa` |
| fe-doc | documenter | frontend | local | `gpt-oss-20b` | local GPU (llama router) | role `documenter` |
| fe-pr-reviewer | pr_reviewer | frontend | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| ux-pm | cell_pm | ux_ui | anthropic | `sonnet-5.5` | Claude subscription | agent pin |
| ux-dev-1 | developer | ux_ui | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| ux-dev-2 | developer | ux_ui | local | `qwen36-reap` | local GPU (llama router) | role `developer` |
| ux-qa | qa | ux_ui | local | `glm47-flash-reap` | local GPU (llama router) | role `qa` |
| ux-doc | documenter | ux_ui | local | `gpt-oss-20b` | local GPU (llama router) | role `documenter` |
| ux-pr-reviewer | pr_reviewer | ux_ui | anthropic | `sonnet-5.5` | Claude subscription | agent pin |

**Totals:** 14 agents on the **local GPU** (6 developers on `qwen36-reap`, 3 QA and the Intake and Secretary chats on `glm47-flash-reap`, 3 documenters on `gpt-oss-20b`) and **12 on Claude Sonnet 5.5** (`sonnet-5.5` = `claude-sonnet-5-5`: Main PM, 3 cell PMs, all 5 PR reviewers, Product Owner, Head of Marketing, Auditor). **No agent is on plain `sonnet` (Sonnet 5), Opus or Gemma.** (Updated 2026-10-01 03:33 UTC: the nine agents previously on `sonnet` were moved to `sonnet-5.5` at the user's request.)

**Exceptions and notes**
- **High-complexity developer tasks** (`developer:high` role row): Ollama Cloud `kimi-k2.7-code:cloud`, not the local Qwen. If it fails too often, candidates in order: `glm-5.3:cloud`, `minimax-m3:cloud`, `nemotron-3-ultra:cloud`, or Claude Sonnet (§10.6).
- **Role rows that exist but never apply** because an agent pin shadows them: `auditor`, `cell_pm`, `head_marketing`, `main_pm`, `product_owner` (all point at OpenRouter free models). They would only matter if a pin were removed.
- **Local model settings** (`~/.config/llama-server/models.ini`, 64k context, ubatch 1024): `qwen36-reap` `n-cpu-moe 22`, 6 threads, f16 KV, `reasoning-budget 8192`; `glm47-flash-reap` `n-cpu-moe 25`, 4 threads, f16 KV, `reasoning-budget 8192`; `gpt-oss-20b` `n-cpu-moe 6`, 8 threads, `q8_0` KV, `reasoning-effort low`; `gemma4-12b` all-GPU, f16 KV (a preset only; no agent uses it). Served by the `llama-router` user service on `172.20.0.1:1234`, one model loaded at a time, one local agent at a time.
- **Saved RoboCo routing presets** (apply from the AI-routing card in Settings): `pre-local-router-2026-09-30` (original state, 20 rows, with the three Opus pins); `local-llama-router-2026-09-30` (21 rows); `local-llama-router+sonnet55-2026-10-01` (21 rows); `local-llama-router+reviewers-sonnet-2026-10-01` (23 rows); **`local-llama-router+all-sonnet55-2026-10-01`** (23 rows, the current state, all Claude agents on Sonnet 5.5).
- **What changed last:** the two remaining local PR reviewers (`ux-pr-reviewer`, `cell-pr-reviewer-2`) were pinned to `sonnet`, because the local GLM reviewer reached verdicts the gateway rejected (finding text over 300 characters, criteria not matched exactly; §10.15). The change takes effect on the next spawn of those agents; no restart was needed.
- **Open items:** whether the local roles hold up on real work is still being observed (so far: QA passed once, a documenter completed once, no local developer result yet); `be-dev-1` (Qwen) had just started its first run when this was written.
- **Code-level defaults still say plain `sonnet` or `opus` (not changed; needs a code edit and a rebuild):** `ROLE_MODEL_MAP` in `roboco/models/runtime.py` (developer, QA, documenter, cell PM and main PM default to `sonnet`; PR reviewer, auditor, product owner, head of marketing, CEO, intake and secretary default to `opus`), the capability floor (`_CAPABILITY_FLOOR_MODEL = "sonnet"` in `roboco/services/llm.py`), and the fallback route. These only apply when no DB assignment covers an agent, or when a self-hosted server is unreachable and RoboCo silently falls back to Anthropic. **Practical consequence:** if the `llama-router` service is down, the 14 local agents do not stay local; they fall back to those defaults (developers, QA, documenters on `sonnet` = Sonnet 5; Intake and Secretary on **`opus`**), spending Claude quota. Making every Claude path Sonnet 5.5 would mean repointing the `sonnet` alias (or the defaults) to `claude-sonnet-5-5`, updating the unit test that pins `sonnet` to Sonnet 5, and rebuilding and redeploying the orchestrator and dispatcher.

### 10.16 `be-dev-1` stranded: Qwen's chat template rejected Claude Code's requests (2026-10-01, late)

**Symptom.** The user received a "stranded" notification for `be-dev-1` (task `c0ab13b2`) and its breaker reached 4. The audit log showed 4 spawns in 10 minutes, two by `_crash_retry_or_escalate`, and **no verb ever called by the agent**.

**Cause.** Every call from the agent to the router returned **HTTP 500**. Claude Code retries 10 times, gives up and exits with code 1; RoboCo counts that as a crash, auto-restarts twice, then escalates (the notification). The router log shows why: Qwen3.6's built-in chat template raises `System message must be at the beginning.` for any system message that is not first in the conversation, and Claude Code's real request histories contain one. (GLM's and gpt-oss's templates accept this, which is why only the Qwen developers failed. My earlier eval requests used a plain-string system prompt and never produced the case, so it did not show there.) Which exact part of Claude Code's payload creates the later system message was **not identified**; the server's Anthropic conversion itself merges the `system` field into a single first message, and a request with several `system` blocks passed.

**Fix (applied).** A tolerant copy of the template at `~/.config/llama-server/templates/qwen36-tolerant.jinja` (one change: render a later system message inline as a system turn instead of raising) is referenced by `chat-template-file` in the `qwen36-reap` preset (backup: `models.ini.bak-2026-10-01b`); the router was restarted. Before/after on the same request (a conversation with a later system message): old template HTTP 500 with the exact error, new template 200; ordinary Anthropic requests unaffected. **Confirmed on the real agent:** after the fix `be-dev-1` started, called `give_me_work` and attempted `open_pr` (rejected by the gateway as `invalid_state`, an ordinary sequencing mistake), with zero template errors in the router log. Side effect to keep in mind: a mid-conversation system message is now passed to Qwen as a system turn, which is not a layout Qwen was trained on; quality impact not measured.

**Breaker reset (user request).** `reset-breaker.sh --all-tripped` removed all 9 tripped rows (`be-pm`/`46781ab0`, `head-marketing`/`510135fe`, `main-pm`/`539560ca`, `auditor`/`5ec7be4c`, `main-pm` and `ux-pm`/`74f865c9`, `ux-qa`/`a83c7959`, `product-owner`/`b0071f75`, `be-dev-1`/`c0ab13b2`); it was run in the gap between two agents so nothing was killed. The overview's "stalled" count comes from a separate durable marker on each task, which RoboCo clears only on real progress, so the 8 tasks still carrying `breaker_tripped` were cleared through `TaskService.clear_stalled_marker` (the code path RoboCo itself uses); 0 remain. Expect the dispatcher to retry all of them: tasks that are genuinely blocked (for example `a83c7959`, waiting on `be6971da`) will re-trip within a few minutes, and the board-cycle tasks now spawn Sonnet 5.5 agents again (Claude quota). Three of the reset tasks are `paused` (`46781ab0`, `539560ca`, `74f865c9`) and will not spawn until someone resumes them.

### 10.17 Two questions from the overview page (2026-10-01, early morning UTC)

**`a83c7959` (UX/UI "Documentation Reconciliation ...", `awaiting_qa`) looks like it is waiting for a heartbeat; claimed 14 hours ago.** Row facts: claimed by `ux-qa` at 12:55 UTC on 09-30, last heartbeat 13:57 UTC. It is not a hung agent. The task depends on `be6971da` ("Asset Integrity Audit", `awaiting_documentation`, itself claimed 15:34 and last heartbeat 15:43 UTC on 09-30), and the dispatcher refuses to spawn `ux-qa` ("waiting on non-terminal dependency `be6971da`"), so no heartbeat can arrive. Each refused spawn is a breaker strike: after the 03:47 reset the stalled marker came back at 03:49 (re-tripped within two minutes, as predicted in §10.11). The chain is therefore `a83c7959` (QA) waits on `be6971da`, which waits on a documentation run by `ux-doc`. `ux-doc` has **not run in the last 10 hours** and every dispatcher tick logs "Spawn skipped: local concurrency cap reached", because the single local slot is held by `be-dev-1` working `c0ab13b2` (a long run; see below). Nothing is broken; it is slot starvation behind a slow developer run. Not done: forcing the dependency or status, or raising `ROBOCO_LOCAL_MAX_CONCURRENT` (two local agents would force model swaps on one 12 GB GPU).

**The "Librarian playbook-mining cycle" (`5ec7be4c`) shows completed with nothing visible: it did produce output, but not on the task card.** By design the Librarian program completes its task in the same call that drafts the playbooks (`propose_playbook_drafts`), so the task carries no notes. Evidence: the `auditor` agent spawned on `sonnet-5.5` at 03:47:27 UTC, called `propose_playbook_drafts` at 03:47:49, and a playbook row was created at 03:47:50 and approved at 03:48:40 (the creator and approver ids are both the auditor's; a `board_program.decision` "approved" event is recorded). The playbook is **"Parse machine-readable config deliverables with their real parser before submit/pass"** (scope `org`, `source_program = librarian`, indexed in the knowledge base). It was mined from recurring journal learnings on tasks `305f4cda` and `907d2958` (a Markdown `**bold**` list item in `.roboco/conventions.yml` raised a YAML `ScannerError`; the lesson recurred 5 times) and tells agents to run machine-read files (`.yml/.json/.toml`) through their real parser before submitting. Where to look: the panel's Knowledge Base page (playbooks category) and the dashboard's playbook review queue, not the task page. Oddities not traced: the completed task has an empty `completed_at`, and the same agent id both drafted and approved the playbook within 50 seconds. Yesterday 09-30 18:01 to 18:04 UTC the same task had five failed spawns on the free OpenRouter model (`opencode/nemotron-3.5-lightning-free`), which is why it sat tripped until today's Sonnet 5.5 run.

**`be-dev-1` progress (first full local developer task, `c0ab13b2`).** After the template fix (§10.16) it claimed and started the task at 03:49 UTC; at 04:01 UTC (about 12 minutes later) it tried `open_pr` and the gateway rejected it with "required tracing missing: commits>=1, commit at least one change before opening a PR". It is working through the gateway's guidance, but slowly (about 12 minutes between audited verbs, with a local-model run holding the slot throughout). Router template errors: 0.

### 10.18 Overnight status check (2026-10-01, about 05:35 UTC)

**Healthy:** orchestrator, dispatcher, nginx, panel, postgres up; `llama-router` active; dispatch unpaused; 0 router errors in the last 90 minutes (the 47 in 6 hours were 45 old Qwen template errors from before the fix plus 2 earlier 500s). Board: 10 completed (2 at the previous check), 2 in progress, 1 awaiting QA, 2 awaiting documentation, 5 pending, **11 paused** (PM-paused tasks; not investigated).

**Progress:** `c0ab13b2` (be-dev-1, Qwen) reached `awaiting_documentation` at 04:45 UTC, so a local developer's work passed QA. `be6971da` ("Asset Integrity Audit", the dependency that blocked `a83c7959`) and `c5395dbf` completed after the local documenters (`gpt-oss-20b`) ran; PMs on Sonnet 5.5 completed several more. Dispatcher spawns in the last 2 hours: 12 auditor, 10 ux-dev-1, 7 ux-pm, 4 fe-pm, 3 fe-dev-1, 2 fe-doc, and single runs of be-dev-1, be-dev-2, be-qa, ux-qa, ux-doc, ux-pr-reviewer, cell-pr-reviewer-2, head-marketing, product-owner. **Clean exits for every GLM QA and gpt-oss documenter run (5 of 5) and every Sonnet 5.5 run.**

**Needs attention (1): `a83c7959`.** Its dependency (`be6971da`) is now completed, but the `ux-qa` breaker row on it is still tripped (count 4) with the stalled marker, so it will not spawn until reset. The cause is gone, so a reset should now stick (`reset-breaker.sh a83c7959:ux-qa`, in a gap between agents).

**Needs attention (2): Qwen developers crash most runs.** 11 of 15 Qwen developer spawns in the last 2 hours ended with exit code 1 ("Agent container stopped unexpectedly"): ux-dev-1 7 of 10, fe-dev-1 3 of 3, be-dev-1 1 of 1 (be-dev-2 and one ux-dev-1 and one be-dev-1 run exited cleanly). Runs lasted 2.5 to 12.5 minutes, so it is not a fixed timeout. RoboCo restarts a crashed agent twice and then escalates, which produces the "stranded" notifications. Evidence from the one retained container log (`fe-dev-1`): the model emitted a tool call whose JSON arguments were **cut off mid-sentence** (206 bytes ending "...Cross-reference"); Claude Code then reported `terminal_reason: api_error` and the session ended with exit 1. Cause of the cut-off not established: it could be the model stopping early, or the stream being dropped (the router logged only 2 "Connection handling canceled" in 3 hours). Observation, not a diagnosis: the router serves overlapping Qwen requests in parallel slots (a 29k-token prompt processed while another request decodes), which collapses decode speed (37 to 0.2 tokens/s in one sample). A Sonnet agent that investigated the `ux-dev-1` stranded alert reported it could not confirm the agent was stuck.

**Options for the Qwen developers (not applied):** (a) keep Qwen and capture the stop reason of the failing responses (router verbose logging or a logging proxy) before changing anything; (b) move the six developers to Claude Sonnet 5.5 (reliable, uses Pro quota); (c) move them to the already-configured Ollama Cloud `kimi-k2.7-code:cloud` (as `developer:high` already is; limits and cost of that plan not checked); (d) restrict the router to one parallel slot for Qwen (`parallel = 1`) to rule out concurrency effects.

### 10.19 Root cause of the Qwen developer crashes: the 64k context overflows (2026-10-01, about 05:50 UTC)

At the user's direction the cause was captured **before** changing anything (the one-request-at-a-time idea was dropped from `models.ini`: an edit adding `parallel = 1` was made and then reverted before any restart; backups `models.ini.bak-2026-10-01b` and `...c` exist).

**Cause (evidence from the router log, all three crashes sampled).** Immediately before each crash (01:16:33, 01:27:04, 01:36:05 EDT) the router logged a generation that ended with `stop processing: n_tokens = 65535, truncated = 1` (the whole 64k context filled, so the response was cut off, which is the tool call "cut off mid-sentence" seen in the `fe-dev-1` transcript), followed at once by the next request failing with `request (65749 / 65742 / 69864 tokens) exceeds the available context size (65536 tokens)`. Claude Code received that error, recorded `terminal_reason: api_error` and exited with code 1, which RoboCo counts as a crash (auto-restart twice, then escalate and the "stranded" notification). Not concurrency: the router showed no error or cancel in the 8 seconds before any crash.
**Scope over the last 8 hours (router log):** requests proxied: qwen36-reap 450, glm47-flash-reap 274, gpt-oss-20b 169; overflow errors: Qwen 12, gpt-oss 3, GLM 1; generations cut at the context limit: Qwen 9, gpt-oss 4, GLM 1. So every local model is exposed; Qwen developers are hit most because their sessions are the longest.
**Why Claude Code does not trim its history in time:** the agent log says `[claude-code:unrecognized_model] qwen36-reap`, so Claude Code does not know the model's context window and, as far as can be seen, assumes a large default; nothing makes it compact before 64k. The installed Claude Code (2.1.276 in the agent image) contains the settings `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, `DISABLE_AUTO_COMPACT`; **RoboCo sets none of them** (searched `roboco/runtime`, `roboco/llm`, `docker/scripts`). RoboCo regenerates each agent's `settings.json` at every spawn (no env hook) and injects only `ANTHROPIC_BASE_URL` for local routes (`roboco/runtime/engines/spawn_launch.py` around line 1017). Not tested: that `CLAUDE_CODE_AUTO_COMPACT_WINDOW` actually changes compaction for an unrecognized model (the string is in the binary; its behavior was not verified).

**Options (none applied):**
1. **Patch RoboCo to tell Claude Code the real window for local agents** (a new setting, for example `ROBOCO_LOCAL_CONTEXT_WINDOW`, passed as `-e CLAUDE_CODE_AUTO_COMPACT_WINDOW=<about 56000>` next to `ANTHROPIC_BASE_URL`, plus a unit test, an image rebuild and a redeploy with the registry-env gotcha of §10.9). Keeps the 64k context (fast prefill) and fixes the cause for all three local models. Success check: overflow errors and `truncated = 1` in the router log drop to zero.
2. **Raise the server context** (Qwen3.6's KV cache is small because it is a hybrid model, so 128k or more may fit with a few more `--n-cpu-moe` layers). Needs a fit test with a real prompt and costs speed; without option 1 it only moves the point of overflow, and long prompts take about 65 s to process per 29k tokens at roughly 450 tokens/s.
3. Both, with a larger window and compaction set below it.
Recommended: option 1.

**Card reset.** `reset-breaker.sh a83c7959:ux-qa` run in a gap between agents (nothing killed; 1 row), and the stale marker cleared. The dependency (`be6971da`) is complete, so the dispatcher no longer refuses the spawn; at the last check `ux-qa` was only queued behind the single local slot ("local concurrency cap reached", which does not add a breaker strike).

### 10.20 Fix for the context overflow: bigger Qwen window plus a compaction window for Claude Code (2026-10-01, morning)

Both halves of the fix from §10.19 were applied (the user said "do both"). Dispatch and engines were paused by the assistant for the GPU tests (stamped `claude-code (user-authorized)`, 3-hour expiry, resumed afterwards); no agent was killed.

**Overnight evidence that the problem was ongoing:** in the 6 hours before the fix the router logged 29 context-overflow errors for Qwen (511 requests), 1 for GLM (28) and 1 for gpt-oss (134).

**Half 2: larger server context for Qwen** (standalone `llama-server` runs with the real flags, then a real long prompt; Qwen3.6 is a hybrid model, so its KV cache is small):

| Context | KV | `--n-cpu-moe` | Result |
|---|---|---|---|
| 131,072 | f16 | 22 | does not fit (out of memory at load) |
| 131,072 | f16 | 24 | **fits**; a real 97,487-token prompt took 245 s (prefill 399 t/s) and decoded at **43.1 t/s** at that depth; VRAM about 11.25 GB peak |
| 196,608 | q8_0 | 24 | fits; a 147,495-token prompt took 387 s (prefill 383 t/s) but decoded at only **16.4 t/s** at that depth |

Chosen: **`ctx-size = 131072`, f16 KV, `--n-cpu-moe 26`** (the 24 that fit plus 2 layers of headroom, per the standing rule that benchmarked minimums are not safe serving values) in the `qwen36-reap` preset (backup `models.ini.bak-2026-10-01d`). GLM and gpt-oss stay at 65,536 (their KV caches are not small enough to double cheaply; each logged at most 1 overflow). Verified in the router's `/v1/models`: qwen36-reap ctx 131072, n-cpu-moe 26, f16.

**Half 1: tell Claude Code the real window** (code change, local and uncommitted like the earlier edits):
- `roboco/config.py`: new setting `local_context_windows` (env `ROBOCO_LOCAL_CONTEXT_WINDOWS`, format `model=tokens,model=tokens,*=tokens`) and `Settings.local_compact_window_for(model)`; malformed entries are ignored; empty means unchanged behavior.
- `roboco/runtime/engines/spawn_launch.py` (`_append_provider_env`): for `provider_type == "local"` only, adds `-e CLAUDE_CODE_AUTO_COMPACT_WINDOW=<tokens>` next to `ANTHROPIC_BASE_URL`. Other providers are untouched.
- `docker-compose.registry.yml` (shared env anchor): `ROBOCO_LOCAL_CONTEXT_WINDOWS: ${ROBOCO_LOCAL_CONTEXT_WINDOWS:-qwen36-reap=120000,*=56000}`, so the default lives in the compose file and `.env` was not touched. **The compose values and the `ctx-size` values in `models.ini` must stay consistent** (each window below its server context with room for the 8,192 thinking budget and the reply): Qwen 120000 under 131072; GLM and gpt-oss 56000 under 65536.
- Tests: `tests/unit/runtime/test_local_context_window.py` (10 tests: parsing, exact beats wildcard, malformed ignored, local agents get the env, other providers do not). In a throwaway container with the new source mounted: 249 tests across the new file plus `tests/unit/models`, `tests/unit/billing` and the earlier guard test, all passing.
- Deploy: orchestrator image rebuilt; orchestrator and dispatcher recreated with `ROBOCO_REGISTRY=ghcr.io/rennf93` (the registry-env gotcha of §10.9); `settings.local_compact_window_for` resolves to 120000 (Qwen) and 56000 (GLM, gpt-oss) in the running container.
- **Verified live:** the first local agent after the deploy (`be-dev-1`, Qwen) started with `CLAUDE_CODE_AUTO_COMPACT_WINDOW=120000` in its container environment.

**Not yet verified:** that Claude Code actually compacts early enough for these windows (its own text says the window can come from this variable, a settings key, or a default for an unrecognized model, but the exact compaction threshold was not read). The check is the router log: overflow errors and `truncated = 1` generations should drop to zero. **Not covered:** the Intake and Secretary live chat sessions use a separate launcher (`interactive_sessions.py`) and were not patched. **Costs to expect (corrected 2026-10-01 after the user pointed out that the tutorial says only changed context is reprocessed):** decode slows with depth (43 t/s at 97k vs about 58 t/s at 16k). The earlier claim that a deep session pays minutes of re-prefill on every turn was **wrong**. Measured from the router log over 9 hours, prompt tokens that were reused from the server's prompt cache instead of being reprocessed: **Qwen 92%** (729 requests, 33.3 M prompt tokens sent, 2.8 M processed; also 92% for prompts over 20k), **GLM 94%** (85 requests), **gpt-oss 96%** (155 requests). A normal agent turn therefore processes only the new tokens. Full-price prefill (about 400 to 485 tokens/s, so a 29k-token prompt takes about 65 s) happens only when the cache cannot help: a new agent session's first request (36 near-full prefills of 27k to 29k tokens for Qwen in 9 hours, which is the agent's initial prompt), a model swap or router restart (the cache is lost), and any change early in the history (for example after Claude Code compacts, everything after the first changed token is reprocessed, but the compacted prompt is far shorter than the pre-compaction one). Caveat found in the same log: the server printed `cache_reuse is not supported by this context, it will be disabled` (5 events for qwen36-reap, 3 for gpt-oss-20b), so the tutorial's chunk-level `--cache-reuse 256` (reusing chunks after a mid-prompt change) is **not active** for these two architectures here; the 92 to 96% comes from ordinary prefix reuse (the server picks the slot with the longest common prefix). It is a llama.cpp prompt-cache feature, not something specific to REAP models.

### 10.21 Alert: task `f44b74db` flip-flopping block/unblock, and a zombie developer container (2026-10-01, about 13:10 UTC)

**Alert:** "Task 'Fix contradictory Conclusion in sim audi' (#f44b74db) has been blocked and unblocked 3 times" (a structural wedge).

**Sequence (audit log, UTC):** 12:37 `be-dev-1` (Qwen) claimed the task and called `sync_branch`, which the gateway rejected with `DIRTY_WORKSPACE: Cannot rebase with uncommitted changes` and the explicit remediation `sync_branch(stash=True)`. The developer did not follow it and called `i_am_blocked` (12:38:30), escalating to `be-pm`. `be-pm` (Sonnet 5.5, spawned by `_dispatch_blocker_work`) unblocked it at 12:38:58; **8 seconds later `be-dev-1` blocked it again** (12:39:06) and escalated again: the flip-flop, three times. The task was then reassigned to `be-pm`.

**Still wrong at 13:09 UTC:**
- The task is `blocked`, assigned to `be-pm`, with the stalled marker (`breaker_tripped`).
- **`be-dev-1`'s container has been running 36+ minutes as a zombie**: from 12:39 to 13:08 it kept calling verbs on a task it no longer owns (`i_will_work_on`, `sync_branch`, `unclaim`, `i_am_blocked`, all rejected by the gateway: "blocked", "no longer assigned to you", "not assigned to you"). Last rejected call 13:08:49. It never terminated on its own and **holds the single local-agent slot**, so every other local agent waits behind it.
- Its dev notes claim the report's Conclusion was already correct "verified by clean working tree", but its clone (`/data/workspaces/atomic-frontier/backend/be-dev-1`) is on `main` with a **dirty tree** (modified `.gitignore`, untracked `Makefile`, `docs/architecture/`, `scripts/blender/`, `scripts/bootstrap-env.sh`, ...), probably leftovers from earlier crashed Qwen runs. So that note is inaccurate.

**Reading:** the first block came from a model that did not apply the gateway's own remediation (stash and sync), the flip-flop from a still-running developer session re-blocking right after the PM's unblock, and the long tail from a session that does not exit when it loses ownership. This is the third distinct Qwen developer failure mode seen today (context overflow, §10.19; remediation not followed; zombie loop), so it adds weight to moving the developers to a more reliable model (§10.18 options).

**Not done (awaiting the user's decision):** stopping the zombie container (RoboCo may auto-restart a stopped agent, so this needs care), clearing the stalled marker, cleaning `be-dev-1`'s workspace (RoboCo already resets a clone on a fresh claim), and any model change.

### 10.22 First evidence on the compaction window, from task `7a769f6a` (2026-10-01, about 13:57 UTC)

`fe-dev-1` (Qwen) was claimed on `7a769f6a` ("Audit project/ Godot integration layer and write findings log") at 13:12 UTC and, 44 minutes later, was still running with its last heartbeat at 13:12; the user offered to pause RoboCo. It was **not stuck**: its container log showed 172 assistant turns and 137 tool results (mostly `Read` of project source files, which are not audited gateway verbs and therefore send no heartbeat), no API retries, and the router showed it decoding at about 44 tokens/s with the GPU at 54%.

**Evidence the window setting works.** The container logged 7 `compact_boundary` events, all `trigger: auto`, at **88,653 to 103,287 tokens** (below the 131,072 server limit and consistent with `CLAUDE_CODE_AUTO_COMPACT_WINDOW=120000` minus a buffer), shrinking the context to between 7,235 and 56,159 tokens. Router counters since the fix at 08:35 EDT: **0 `exceeds the available context size` errors and 0 `truncated = 1` generations in 209 requests** (29 overflow errors in the 6 hours before the fix).

**What it costs.** Each compaction took 89.8 to 245.4 seconds (about 17 minutes of a 44-minute run, roughly 40%): the compaction request makes the model write a long summary at about 45 tokens/s, and a file-reading audit refills the context every few minutes. The window values are therefore a speed trade-off: a larger window means fewer compactions but slower decode at depth, a smaller one means more frequent compactions. No change made. The breaker row `fe-dev-1`/`7a769f6a` stood at count 3 (it trips at 4).

