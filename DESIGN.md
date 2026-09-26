# dxlean design

## What this is

A Lean 4 theorem-proving environment for deepxube. The research framing: hold a
**pre-trained** policy+value model (nanoproof) fixed and study what the *search harness*
contributes: deepxube's batch weighted A* with transposition merging vs. the MCTS harness
the model shipped with, under equal budgets. No model training here; every (state,
tactic, outcome) is logged by construction, so training (expert iteration, learned value)
can be added later without redesign. A quantum-circuit domain over QLean is a planned
future plugin; nothing here is quantum-specific.

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
PROPOSE  (ActionProviders: nanoproof samples ∪ backbone menu; batch across frontier)
VALIDATE (REPL applies each candidate; failures → per-state negative cache,
          fed back to providers on re-proposal)
CACHE    (successor states stored)
```

and `next_state` is a pure cache lookup. A state whose every candidate fails is
re-proposed up to `--resamples` times with the failed set surfaced to the providers
(nanoproof re-samples with a fresh server seed); if still empty it returns an empty
action list, the instance's frontier drains and it finishes unsolved (verified to flow
through `ActsEnum.expand` — see tests).

## Components (dxlean/)

- `repl.py` — leanprover-community REPL adapter. JSON-over-stdio protocol (verified
  v4.30.0): `{"cmd"}` → env/sorries, `{"tactic","proofState"}` → new state or
  `{"message": "Lean error..."}`. Must run under `lake env` from the project dir
  (bare invocation yields an Init-less env where even `+` fails to parse).
  `REPLManager` adds header env, state→proofState-id map, kill/restart with replay
  (states carry their tactic prefix, so any process can reconstruct them), and
  `check_full_proof` — fresh elaboration of the assembled script; `sorry`/`admit`/
  `native_decide` banned. Failure taxonomy: ok / solved / error / **no_progress**
  (tactic succeeded, state unchanged — filtered to prevent self-loops) / timeout.
  Any timeout restarts the process: a timed-out REPL is still computing and would
  answer the *next* request with the stale reply, desyncing everything after it
  (the reader thread is bound to its own queue for the same reason).
  **Self-reference guard**: the theorem is declared `:= by sorry`, so its own
  sorry-backed constant exists in the search-time environment; tactics mentioning
  the theorem's name are rejected at the gate (found live — a model proposed
  `apply and_swap` inside `and_swap` and search "solved" it; certification in the
  pristine header env rejected it, and now search never accepts it either).
- `nanoproof.py` — the pre-trained nanoproof checkpoint as plugins. Speaks nanoproof's
  own Flask inference protocol (`POST /generate {"states"} -> tactics+logprobs+value`,
  503 = busy, retried). One `NanoproofOracle` cache (keyed by prompt string) backs both
  `NanoproofProvider` (samples as candidates, logprob as score; re-proposal or an empty
  sample re-queries with a fresh server seed and merges) and `NanoproofValue` (h =
  predicted remaining proof depth, 1..64 bins, unscaled). deepxube scores children at
  generation and expands them later, so the value query already stocks the tactic
  cache: one GPU call per state. Goal text is re-rendered the way leantree
  (nanoproof's training data) prints it — grouped hypotheses `a b : ℕ` split one per
  line.
- `providers.py` — `ActionProvider` ABC (batch propose), `BackboneProvider` (fixed
  menu; guaranteed recall on routine closers and the model-free path for tests),
  `UnionProvider` (order-preserving dedupe, cap).
- `values.py` — `ValueProvider` ABC, `GoalCountValue` (deterministic, model-free),
  `as_heurv` adapter.
- `domain.py` — the `ActsEnum` domain: propose→validate→cache, resampling, stats.
- `solve.py` — all theorems run as concurrent search instances (provider/value calls
  batch across the whole frontier); finished instances → path extraction
  (`get_path`) → certification → `SolveResult`. Budgets `itr_max` (iterations) and
  `calls_max` (model calls) are enforced through deepxube's `remove_instances`; costs
  come from one ledger (`states.Ledger`, theorem → counters) that the domain charges
  for REPL validations and the nanoproof oracle for model calls — each theorem pays
  once per prompt it asks for, so its cost is what it would pay alone. The search
  stops at the first proof (nanoproof's rule), not at deepxube's W-optimality bound.
- `cli.py` — `dxlean solve` (nanoproof URL, backbone menu, value choice,
  W/B/eps/cap/itr-max/calls-max, results + verified-proof output) and `dxlean viz`.
- `viz.py` — visualization of the environment and search process, all text-first
  on the REPL's pretty-printed goals: `render_state_goal` (matplotlib figure —
  backs the deepxube `StateGoalVizable` mixin on `LeanDomain`, which also
  implements `StringToAct`), `interactive` (proof shell with provider proposals
  and undo), `traced_search` (per-iteration narration + search-tree print/figure;
  deepxube keeps the tree in `Node.edge_dict`, so the tree view is a free walk).

## Verified deepxube integration points

- `HeurVFn`/`PolicyFn` are `runtime_checkable` Protocols (`base/pathfind_fns.py`) —
  plain callables plug in; the `PolicyFn` shape means `ActsPolicy` search variants
  (model-proposed edges + random exploration) are available later for free.
- Solve loop: `make_instances(states, goals, inst_infos)` → `add_instances` →
  `step()` until `remove_instances(done)` drains (`done`: solved, frontier drained,
  or a budget hit); per-instance `inst_info` carries the theorem name.
- `set_is_solved` runs on *popped* nodes, and a solved child (h = 0, f = W·g) is not
  necessarily popped next when W > 1 or costs tie. `solve.py` therefore registers
  solved nodes waiting in `_nodes_curr` / `open_set` itself via `record_goal`, so a
  proof found within budget counts and is never dropped by a budget removal.

## Current status / known limits

- Single REPL process, sequential validation. Fine Mathlib-free (ms/tactic); a worker
  pool with state affinity is the first scaling step for Mathlib-based benchmarks.
- Transition/negative caches are in-memory per run; persisting them (and logging them
  as the training-data harvest) is designed but not built.
- Dev corpus is core-Lean-only. `exact?` in the default backbone one-shots most of it —
  use `--backbone` without it to exercise real search; miniF2F/Mathlib benchmarks are
  the next corpus step (per-benchmark toolchain/REPL matching needed).
- nanoproof pins Lean v4.27.0 + Mathlib; testproj is v4.30.0 without Mathlib. Its
  samples are Mathlib-shaped, so evaluate it against the cluster's Mathlib project
  with a matching REPL (`scripts/setup_repl.sh nanoproof` reads the pin from the
  training script, keeping versions in sync). Open question to measure:
  `--np-value sum|first|all` — the value head was trained on single factorized
  branches, dxlean states carry the whole goal list.

## Roadmap

1. **Benchmarks**: miniF2F valid/test exported by `scripts/export_minif2f.py` with
   nanoproof's parsing, so `scripts/compare.py` joins its `theorems.jsonl` with
   dxlean's `results.jsonl` by theorem (done); a Mathlib-extracted dev corpus for
   volume and difficulty spread.
2. **Harness grid**: nanoproof MCTS (`scripts/bench_nanoproof.sh`) vs {WA* W-sweep,
   batch-size sweep, eps exploration} × {`--np-value` modes} (`scripts/bench_dxlean.sh`), all at the same
   `--calls-max`; wall clock only once the REPL pool exists.
3. **Scaling**: REPL worker pool with state affinity; persistent transition cache
   doubling as the harvest log.
4. **Later**: expert iteration on the sampler; learned value via deepxube training
   (requires a difficulty-parameterized problem generator — domain-specific by
   nature); QLean quantum-circuit plugin with its scrambler as both curriculum and
   in-domain SFT corpus.
