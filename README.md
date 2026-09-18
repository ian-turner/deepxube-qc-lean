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
end), `--itr-max` (search budget per theorem).

Results land in `--out`: `results.jsonl` plus `proofs/<name>.lean` for every proof
that passed certification.

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
