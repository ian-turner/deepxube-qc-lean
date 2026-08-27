# dxlean design

## What this is

A generic Lean 4 theorem-proving environment for deepxube. The research framing: hold
**pre-trained** models fixed — a tactic generator (policy) and a value estimator — and
study what the *search harness* contributes: deepxube's batch weighted A* with
transposition merging vs. the best-first/MCTS harnesses those models shipped with,
under equal budgets. No model training in v1; every (state, tactic, outcome) is logged
by construction, so training (expert iteration, learned value) can be added later
without redesign. A quantum-circuit domain over QLean is a planned future plugin;
nothing here is quantum-specific.

## The mapping

| deepxube concept | dxlean realization |
|---|---|
| `State` | `LeanState`: full proof state (all goals) + tactic prefix from the root; hash/eq on (theorem, canonicalized goal text) → transpositions merge in the CLOSED dict |
| `Action` | `TacticAction`: normalized tactic text |
| `Goal` | `LeanGoal`: the theorem; `is_solved` ⇔ no goals remain |
| `Domain` | `LeanDomain(ActsEnum)` — per-state, variable-size action sets |
| heuristic | any `ValueProvider` via `as_heurv` → deepxube's `HeurVFn` **Protocol** (any callable; no torch, no registration) |
| search | `GraphSearchHeurNodeActsEnum(domain, PFNsHeurV(heurv=...), batch_size, weight, eps)`; cost = `W·g + h` (weight on path cost; lower W = greedier) |

## The inversion (core design decision)

Tactic applicability is only decidable by running the tactic, and deepxube has no
"action failed" channel. So `get_state_actions` does the real work:

```
PROPOSE  (ActionProviders: backbone menu ∪ LLM samples; batch across frontier)
VALIDATE (REPL applies each candidate; failures → per-state negative cache,
          fed back into later prompts)
CACHE    (successor states stored)
```

and `next_state` is a pure cache lookup. States with zero valid tactics return empty
action lists; the instance's frontier drains and it finishes unsolved (verified to
flow through `ActsEnum.expand` — see tests).

## Components (src/dxlean/)

- `repl.py` — leanprover-community REPL adapter. JSON-over-stdio protocol (verified
  v4.30.0): `{"cmd"}` → env/sorries, `{"tactic","proofState"}` → new state or
  `{"message": "Lean error..."}`. Must run under `lake env` from the project dir
  (bare invocation yields an Init-less env where even `+` fails to parse).
  `REPLManager` adds header env, state→proofState-id map, timeout kill/restart with
  replay (states carry their tactic prefix, so any process can reconstruct them), and
  `check_full_proof` — fresh elaboration of the assembled script; `sorry`/`admit`/
  `native_decide` banned. Failure taxonomy: ok / solved / error / **no_progress**
  (tactic succeeded, state unchanged — filtered to prevent self-loops) / timeout.
- `providers.py` — `ActionProvider` ABC (batch propose), `BackboneProvider` (fixed
  menu; guaranteed recall on routine closers), `LLMSampler` (prompt → k tactics, one
  per line; parses/normalizes/filters), `UnionProvider` (order-preserving dedupe, cap).
- `values.py` — `ValueProvider` ABC, `GoalCountValue` (deterministic, model-free),
  `LLMJudgeValue` ("steps remaining" integer, cached by state key), `as_heurv` adapter.
- `llm.py` — minimal OpenAI-compatible chat client (works with ollama/LM Studio/mlx
  locally and vLLM on the CUDA server) + `FakeChatClient` for deterministic tests.
- `domain.py` — the `ActsEnum` domain: propose→validate→cache, stats counters.
- `solve.py` — all theorems run as concurrent search instances (provider/value calls
  batch across the whole frontier); finished instances → path extraction
  (`get_path`) → certification → `SolveResult`.
- `cli.py` — `dxlean solve` with knobs for backbone menu, endpoint/model, value
  choice, W/B/eps/k/cap/itr-max, results + verified-proof output.

## Verified deepxube integration points

- `HeurVFn`/`PolicyFn` are `runtime_checkable` Protocols (`base/pathfind_fns.py`) —
  plain callables plug in; the `PolicyFn` shape means `ActsPolicy` search variants
  (model-proposed edges + random exploration) are available later for free.
- Solve loop: `make_instances(states, goals, inst_infos)` → `add_instances` →
  `step()` until `remove_finished_instances(itr_max)` drains; per-instance
  `inst_info` carries the theorem name.
- `set_is_solved` runs on *popped* nodes; a solved child must be popped to register —
  fine, it has h = 0 so it pops immediately.

## Current status / known limits (v1)

- Single REPL process, sequential validation. Fine Mathlib-free (ms/tactic); a worker
  pool with state affinity is the first scaling step for Mathlib-based benchmarks.
- One LLM request per state proposal (no cross-state HTTP batching yet); vLLM
  continuous batching will want concurrent requests.
- Transition/negative caches are in-memory per run; persisting them (and logging them
  as the training-data harvest) is designed but not built.
- Dev corpus is core-Lean-only. `exact?` in the default backbone one-shots most of it —
  use `--backbone` without it to exercise real search; miniF2F/Mathlib benchmarks are
  the next corpus step (per-benchmark toolchain/REPL matching needed).

## Roadmap

1. **Server config**: vLLM-served open prover pair (e.g. InternLM2.5-StepProver +
   its critic as a `ValueProvider`; prompt formats matched to the models' training).
2. **Benchmarks**: Mathlib-extracted dev corpus (volume, difficulty spread), then
   miniF2F-test for citable numbers.
3. **Harness grid**: {best-first reproduction, WA* W-sweep, batch-size sweep, eps
   exploration} × {logprob, critic, judge} with honest wall-clock/GPU accounting.
4. **Scaling**: REPL worker pool with state affinity; concurrent LLM requests;
   persistent transition cache doubling as the harvest log.
5. **Later**: expert iteration on the sampler; learned value via deepxube training
   (requires a difficulty-parameterized problem generator — domain-specific by
   nature); QLean quantum-circuit plugin with its scrambler as both curriculum and
   in-domain SFT corpus.
