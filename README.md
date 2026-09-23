# dxlean

Lean 4 theorem proving as a [deepxube](https://github.com/forestagostinelli/deepxube)
pathfinding domain: **states** are Lean proof states, **actions** are tactics proposed by
the pre-trained [nanoproof](https://github.com/Kripner/nanoproof) policy (plus an optional
fixed backbone menu), **guidance** is nanoproof's value head (predicted remaining proof
depth), and **search** is deepxube's batch weighted A*. Found proofs are re-certified by
fresh elaboration.

See [DESIGN.md](DESIGN.md) for architecture and roadmap.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"   # installs deepxube (pinned git sha), torch, etc.
./scripts/setup_repl.sh                          # clone + build leanprover-community/repl v4.30.0
cd lean/testproj && lake build && cd ../..       # build the dev Lean project (no Mathlib, seconds)
.venv/bin/python -m pytest tests/ -q             # needs the REPL built; no GPU or server needed
```

Requires `elan` (the toolchain in `lean/testproj/lean-toolchain` is fetched automatically).

## Usage

nanoproof's policy+value checkpoint is served by nanoproof's own HTTP inference server
(`scripts/train_nanoproof.sh` trains it and has a `serve` stage), so the model never
leaves the GPU cluster:

```bash
# on the cluster — serve prints the companion dxlean command with the right flags
scripts/train_nanoproof.sh serve            # port NP_INFER_PORT, default 5001
```

The model was trained on Lean v4.27.0 + Mathlib states, so run dxlean **on the cluster**
against the Mathlib project the training script builds (`leanproj` stage), with a REPL
built for the matching Lean version:

```bash
# build the REPL version that matches nanoproof's Lean pin (reads it from train_nanoproof.sh)
./scripts/setup_repl.sh nanoproof           # -> vendor/repl-v4.27.0/.lake/build/bin/repl

# run on the same machine as the Lean project (--project must be a local path)
dxlean solve --problems my_bench.jsonl --nanoproof http://localhost:5001 --backbone "" \
    --project /work/$USER/nptraining --header "import Mathlib" \
    --repl-bin vendor/repl-v4.27.0/.lake/build/bin/repl --out results/run1
```

One request per state yields tactic samples (with logprobs, kept as candidate scores) and
the value-head prediction, *remaining proof depth in tactic steps*, used unscaled as `h`.
`--np-goals` picks what the policy sees (first goal, as in nanoproof's factorized search,
or all goals), `--np-value` how `h` is assembled (sum of per-goal depths, first goal, or
all goals joined). `--backbone ""` disables the built-in tactic menu; omit it to union the
menu with the model's samples, or pass your own comma-separated menu.

Without `--nanoproof` the search runs model-free (backbone menu + goal-count heuristic),
which is the GPU-free smoke path used by the tests:

```bash
dxlean solve --problems problems/dev.jsonl
```

Problem files are JSONL: `{"name": ..., "statement": "theorem foo ... : ..."}` — the
statement without a proof; the harness appends `:= by sorry` and searches from there.
`--header` sets the imports for the session (default `import TestProj`); `--project`
points at the Lean project whose environment the REPL runs in. `--cmd-timeout` (default
600s) bounds the header import and full-proof checks; a cold `import Mathlib` from a
network filesystem can take several minutes, and the header is re-imported on every REPL
restart.

Key knobs: `--weight` (on path cost: `W*g + h`, lower = greedier), `--batch` (nodes
expanded per search iteration per theorem), `--cap` (max candidates REPL-validated per
state), `--resamples` (re-proposal rounds with failure feedback before a state is a dead
end), `--itr-max` (search iterations per theorem), `--calls-max` (model calls per theorem).

Results land in `--out`: `results.jsonl` (per theorem: solved/verified, tactics,
iterations, `model_calls`, `validations`), `summary.json` (run totals, wall time,
arguments) and `proofs/<name>.lean` for every proof that passed certification.

## Benchmarking against nanoproof's MCTS

Same checkpoint, same theorems, same Lean environment, same budget currency. nanoproof's
MCTS spends one model call per simulation; dxlean charges each theorem one call per
prompt it asks for (one per goal under `--np-value sum`, cached by goal text, and a
goal another theorem already cached still costs), so `--calls-max` is the budget that
compares across harnesses. Iterations do not: deepxube scores every generated child and
expands only the ones it pops. The search stops at the first proof, like nanoproof.

```bash
# 1. theorems: miniF2F parsed exactly like nanoproof's loader, so names match its ids
scripts/export_minif2f.py        # problems/minif2f_{valid,test}.jsonl + minif2f_header.lean

# 2. nanoproof's own MCTS, on the cluster (prover_eval.py; BUDGET simulations, default 512)
NP_CKPT=<model_NNNNNN.pt> scripts/bench_nanoproof.sh
#    -> <checkpoint dir>/eval_<step>_minif2f_512/{theorems.jsonl,summary.toml}

# 3. deepxube A*: the same checkpoint served, one `dxlean solve` per GRID variant at --calls-max BUDGET
NP_CKPT=<model_NNNNNN.pt> GRID="w1.0 w0.5 b4 vfirst" scripts/bench_dxlean.sh
#    -> results/minif2f_valid_512/<variant>/{results.jsonl,summary.json,proofs/}

# 4. join by theorem: solve rates, solved-within-budget curve, per-theorem disagreements
scripts/compare.py <checkpoint dir>/eval_<step>_minif2f_512 results/minif2f_valid_512/*/
```

The exact `dxlean solve` flags are in [bench_dxlean.sh](scripts/bench_dxlean.sh) (`--backbone ""`,
the Mathlib project, the header file, `--calls-max`). Pilot first with `PILOT=20 BUDGET=64` on
both scripts, which take the same first theorems: a cold `import Mathlib` costs minutes per REPL
start. Wall clock is not comparable yet, since nanoproof drives two dozen Lean workers in
parallel while dxlean validates through one REPL process; compare model calls and validations.
Both scripts read the same sampler settings (`NP_NUM_SAMPLES`, `NP_FIRST_TOKEN_CAP`,
`NP_DISABLE_SOLVERS`); under `NP_DISABLE_SOLVERS=1` nanoproof's eval also appends `grind` to
every expansion, which `--backbone grind` mirrors.

## Visualization

```bash
# watch the search think: per-iteration narration (popped node, f=W*g+h, which
# candidates validated, provenance) + final search tree with the solution path starred;
# --fig also renders the tree as a PNG
dxlean viz --problems problems/dev.jsonl --name and_swap \
    --backbone "intro h,constructor,assumption,rfl,omega" --fig and_swap.png

# interactive proof shell: type tactics, :p asks providers for validated
# proposals, :u undoes, :g reprints goals, :q quits
dxlean viz --problems problems/dev.jsonl --name or_swap --interactive
```

Both take the same `--nanoproof` / `--backbone` flags as `solve`. Tree legend: `★`
solution path, `✓` solved state, `✗` expanded dead end (no valid tactics), `·` generated
but never expanded.
