# dxlean

Generic Lean 4 theorem proving as a [deepxube](https://github.com/forestagostinelli/deepxube)
pathfinding domain: **states** are Lean proof states, **actions** are tactics proposed by
pluggable providers (a fixed backbone menu and/or an LLM behind any OpenAI-compatible
endpoint), **guidance** comes from pluggable value providers, and **search** is deepxube's
batch weighted A*. Found proofs are re-certified by fresh elaboration.

Runs locally on a MacBook with small models (ollama / LM Studio / mlx) and on a CUDA
server with vLLM-served provers — the code only ever sees an endpoint URL.

See [DESIGN.md](DESIGN.md) for architecture and roadmap.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"   # installs deepxube (pinned git sha), torch, etc.
./scripts/setup_repl.sh                          # clone + build leanprover-community/repl v4.30.0
cd lean/testproj && lake build && cd ../..       # build the dev Lean project (no Mathlib, seconds)
.venv/bin/python -m pytest tests/ -q             # 8 tests, needs the REPL built
```

Requires `elan` (the toolchain in `lean/testproj/lean-toolchain` is fetched automatically).

## Usage

```bash
# No LLM anywhere: backbone tactic menu + deterministic goal-count heuristic
dxlean solve --problems problems/dev.jsonl --no-llm

# Restricted backbone (forces real multi-step search; some problems honestly fail)
dxlean solve --problems problems/dev.jsonl --no-llm \
    --backbone "intro h,constructor,assumption,rfl,omega"

# Local model via ollama (`ollama serve` + `ollama pull qwen2.5-coder:7b` first)
dxlean solve --problems problems/dev.jsonl \
    --endpoint http://localhost:11434/v1 --model qwen2.5-coder:7b --value judge

# CUDA server: point at vLLM serving a prover model
dxlean solve --problems my_bench.jsonl \
    --endpoint http://gpu-server:8000/v1 --model <served-model> --value judge \
    --k 12 --batch 4 --weight 1.0 --itr-max 300 --out results/run1
```

Problem files are JSONL: `{"name": ..., "statement": "theorem foo ... : ..."}` — the
statement without a proof; the harness appends `:= by sorry` and searches from there.
`--header` sets the imports for the session (default `import TestProj`); `--project`
points at the Lean project whose environment the REPL runs in.

Key knobs: `--weight` (on path cost: `W*g + h`, lower = greedier), `--batch` (nodes
expanded per search iteration per theorem), `--k` (LLM samples per state), `--cap`
(max candidates REPL-validated per state), `--itr-max` (search budget per theorem).

Results land in `--out`: `results.jsonl` plus `proofs/<name>.lean` for every proof
that passed certification.

## Visualization

```bash
# watch the search think: per-iteration narration (popped node, f=W*g+h, which
# candidates validated, provenance) + final search tree with the solution path starred
dxlean viz --problems problems/dev.jsonl --name and_swap --no-llm \
    --backbone "intro h,constructor,assumption,rfl,omega"

# interactive proof shell: type tactics, :p asks providers for validated
# proposals, :u undoes, :g reprints goals, :q quits
dxlean viz --problems problems/dev.jsonl --name or_swap --interactive

# render the root state/goal as a matplotlib figure (deepxube StateGoalVizable)
dxlean viz --problems problems/dev.jsonl --name and_swap --no-llm --fig root.png

# LLM-only search (no backbone menu): every tactic in the trace and tree is
# tagged [llm]. When every sample fails, the state is re-proposed with the
# failed tactics fed back to the sampler (--resamples, default 2) before it
# becomes a dead end; low temperature still helps small models.
dxlean viz --problems problems/dev.jsonl --name imp_chain \
    --endpoint http://localhost:11434/v1 --model qwen2.5-coder:7b \
    --backbone "" --temperature 0.2 --itr-max 25 --fig imp_chain_llm.png
```

Tree legend: `★` solution path, `✓` solved state, `✗` expanded dead end (no valid
tactics), `·` generated but never expanded.
