#!/usr/bin/env bash
# End-to-end nanoproof training on a single H200.
#
# Pipeline: setup -> data -> tokenizer -> pretrain -> midtrain -> sft
#           (default stops here: this is the fixed policy+value checkpoint dxlean needs)
# Opt-in:   leanproj / leanserver / rl / eval  (the AlphaProof RL loop + MiniF2F)
#
# Usage (run inside your conda env, e.g. `conda create -n nanoproof python=3.12`):
#   conda activate nanoproof
#   scripts/train_nanoproof.sh                 # setup..sft
#   scripts/train_nanoproof.sh smoke           # tiny pipeline sanity check first!
#   scripts/train_nanoproof.sh rl              # after sft; starts leanserver itself
#   scripts/train_nanoproof.sh eval
#   scripts/train_nanoproof.sh setup data      # run individual stages
#
# Knobs (env vars):
#   NP_PYTHON         python to use (default: active env's `python`, i.e. your conda env)
#   NP_TORCH_INDEX    PyTorch wheel index (default cu128 for H100/H200; use
#                     https://download.pytorch.org/whl/cpu on CPU-only hosts)
#   NP_WORK_DIR       where repos/checkpoints live   (default /work/$USER)
#   NP_DEPTH          transformer depth              (default 26, ~1B params; 20 halves cost)
#   NP_FP8            --fp8 for pretrain: auto = only on compute capability >= 9.0
#                     (Hopper: H100/H200); 1/0 force it on/off (default auto)
#   NP_LEAN_PROCS     leanserver --max-processes     (default 24)
#   NP_RSS_LIMIT_GIB  per-REPL-worker hard memory cap (default 16)
#   NP_PSS_RECYCLE_GIB worker recycle threshold       (default 6)
#   NP_WARMUP_WAIT    seconds to let leanserver import Mathlib before RL/eval (default 900)
#   NP_PORT           leanserver port                (default 8000)
#   NP_FORCE          1 = re-run a training stage even if it already has a checkpoint
#   NP_SAVE_EVERY     pretrain checkpoint interval in steps (default 2000, ~every 4h
#                     at depth 26; -1 = only at end). Each save is ~5GB (model +
#                     optimizer) and old saves are not pruned automatically; safe to
#                     delete older model_/optim_ pairs, keep the newest for resume.
#
# Caveats: written against nanoproof/leantree main as of 2026-08; not smoke-tested
# end-to-end here. Nemotron-CC-Math is gated: accept terms on HuggingFace and
# `hf auth login` before the data stage. Run the `smoke` stage before burning GPU-days.
set -euo pipefail

# Training progress is plain print(), which block-buffers under SLURM/pipes
# and can look hung for many minutes; stream it live instead.
export PYTHONUNBUFFERED=1

USER=`whoami`
WORK_DIR="${NP_WORK_DIR:-/work/$USER}"
NP_REPO="$WORK_DIR/nanoproof"
LT_REPO="$WORK_DIR/leantree"
LEAN_PROJECT="$WORK_DIR/nptraining"
export NANOPROOF_HOME="${NANOPROOF_HOME:-$WORK_DIR/nanoproof-home}"

DEPTH="${NP_DEPTH:-26}"
FP8="${NP_FP8:-auto}"
LEAN_PROCS="${NP_LEAN_PROCS:-24}"
RSS_LIMIT="${NP_RSS_LIMIT_GIB:-16}"
PSS_RECYCLE="${NP_PSS_RECYCLE_GIB:-6}"
WARMUP_WAIT="${NP_WARMUP_WAIT:-900}"
PORT="${NP_PORT:-8000}"
FORCE="${NP_FORCE:-0}"
SAVE_EVERY="${NP_SAVE_EVERY:-2000}"

LEAN_VERSION="v4.27.0"   # pinned by leantree/nanoproof (dataset whitelists, REPL fork)
REPL_EXE="$LT_REPO/lean-repl/.lake/build/bin/repl"

# Python from the active (conda) environment; NP_PYTHON overrides.
PY="${NP_PYTHON:-$(command -v python || true)}"
[ -n "$PY" ] || { echo "no python on PATH — activate your conda env first" >&2; exit 1; }
LEANSERVER="$(dirname "$PY")/leanserver"   # console script installed next to python
TORCH_SPEC="torch==2.9.1"                  # pinned by nanoproof's pyproject.toml
TORCH_INDEX="${NP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SERVER_LOG="$WORK_DIR/leanserver.log"
SERVER_PID_FILE="$WORK_DIR/leanserver.pid"

log() { printf '\n[train_nanoproof] %s\n' "$*"; }
die() { printf '[train_nanoproof] ERROR: %s\n' "$*" >&2; exit 1; }

# Newest checkpoint for a stage ($NANOPROOF_HOME/models/<stage>/<run>/model_NNNNNN.pt).
# Sort by mtime: run dir names are HH-MM-SS_DD-MM-YY, which does not sort
# chronologically across days.
latest_ckpt() {
    local files
    files=$(find "$NANOPROOF_HOME/models/$1" -name 'model_*.pt' 2>/dev/null)
    [ -n "$files" ] || return 0
    # shellcheck disable=SC2086  # run paths contain no whitespace
    ls -t $files | head -1
}

stage_done() {  # skip a training stage that already produced a checkpoint
    [ "$FORCE" != 1 ] && [ -n "$(latest_ckpt "$1")" ]
}

# ---------------------------------------------------------------- setup ------

do_setup() {
    log "setup: repos + pip install into $PY"
    command -v elan  >/dev/null || die "elan not found (https://github.com/leanprover/elan)"
    command -v nvidia-smi >/dev/null || log "WARNING: nvidia-smi not found; GPU stages will fail"
    # leantree requires >=3.12; torch.compile requires <3.14
    "$PY" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info < (3, 14) else 1)' \
        || die "need python 3.12 or 3.13, got $("$PY" -V 2>&1) — conda create -n nanoproof python=3.12"
    "$PY" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix or "conda" in sys.version.lower() or "CONDA_PREFIX" in __import__("os").environ else 1)' \
        || log "WARNING: $PY does not look like a conda/virtual env; installing into it anyway"
    mkdir -p "$WORK_DIR"

    [ -d "$NP_REPO" ] || git clone https://github.com/Kripner/nanoproof "$NP_REPO"
    [ -d "$LT_REPO" ] || git clone --recurse-submodules https://github.com/Kripner/leantree "$LT_REPO"

    # Torch first from the CUDA wheel index so `pip install -e .` sees the pin
    # already satisfied and doesn't pull the default PyPI build.
    "$PY" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
    # leantree from the local clone, NOT PyPI: the only wheel there is
    # cp312+glibc>=2.39 and the sdist cannot build (its build script runs
    # `lake build` in the lean-repl submodule, which the sdist omits). The
    # clone has the submodule, so the same build script works here — and
    # builds the REPL binary as a side effect.
    "$PY" -m pip install "$LT_REPO"
    "$PY" -m pip install -e "$NP_REPO"
    "$PY" -m pip install pytest ruff   # nanoproof's dev group

    # Build the LeanTree REPL fork (elan fetches the pinned toolchain automatically)
    if [ ! -x "$REPL_EXE" ]; then
        log "building lean-repl fork ($LEAN_VERSION)"
        ( cd "$LT_REPO/lean-repl" && lake build repl )
    fi
}

do_data() {
    log "data: downloading datasets (Nemotron is gated on HuggingFace)"
    "$PY" - <<'EOF' || { echo "Not logged in to HuggingFace. Accept the Nemotron-CC-Math-v1 terms, then: hf auth login"; exit 1; }
from huggingface_hub import HfApi
HfApi().whoami()
EOF
    # everything except midtrain: the download-all path includes leangithubraw,
    # whose HF repo (Kripi/Lean-Github-Raw) was never published and 404s
    ( cd "$NP_REPO" && "$PY" -m nanoproof.data.download pretrain sft rl bench )
    # midtrain corpus is built locally instead (clones source repos; needs git)
    ( cd "$NP_REPO" && "$PY" -m nanoproof.data.midtrain.leangithubraw build )
}

do_tokenizer() {
    log "tokenizer: GPT-2 BPE + Lean/math special tokens"
    ( cd "$NP_REPO" && "$PY" -m scripts.tok_build )
}

do_smoke() {
    log "smoke: tiny pretrain to validate the pipeline (minutes, negligible compute)"
    ( cd "$NP_REPO" && "$PY" -m nanoproof.pretrain \
        --depth=4 --max-seq-len=512 --device-batch-size=1 \
        --eval-tokens=512 --total-batch-size=512 --num-iterations=20 )
    # The smoke run writes a toy checkpoint into the shared models/pretrain
    # namespace, which would satisfy stage_done and let midtrain chain from a
    # depth-4 model. Remove it, but only after confirming it is the smoke run.
    local run_dir
    run_dir=$(ls -td "$NANOPROOF_HOME"/models/pretrain/*/ 2>/dev/null | head -1)
    if [ -n "$run_dir" ] && grep -q '"depth": 4,' \
            "$NANOPROOF_HOME/logs/pretrain/$(basename "$run_dir")/args.json" 2>/dev/null; then
        rm -rf "$run_dir"
        log "smoke: removed toy checkpoint $run_dir"
    elif [ -n "$run_dir" ]; then
        log "WARNING: could not confirm $run_dir is the smoke run; not deleting it"
    fi
}

# ------------------------------------------------------------- training ------

# Pretrain saves intermediate checkpoints (NP_SAVE_EVERY), so "a checkpoint
# exists" no longer means "finished". Completion is tracked by a marker file
# written only when the training process exits successfully; anything short
# of that resumes from the newest checkpoint on the next run.
do_pretrain() {
    local marker="$NANOPROOF_HOME/models/pretrain/.complete"
    if [ "$FORCE" = 1 ]; then
        rm -f "$marker"
    elif [ -f "$marker" ]; then
        log "pretrain: complete, skipping (NP_FORCE=1 to redo)"; return
    fi
    local fp8_flag="" resume_flag="" ckpt meta cap
    if [ "$FP8" = "auto" ]; then
        # Triton's fp8e4nv kernels need Hopper+ (cc 9.0); on an A100 the run
        # dies with "type fp8e4nv not supported in this architecture"
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
        if [ -n "$cap" ] && awk -v c="$cap" 'BEGIN { exit !(c >= 9.0) }'; then
            FP8=1; log "pretrain: fp8 enabled (compute capability $cap)"
        else
            FP8=0; log "pretrain: fp8 disabled (compute capability ${cap:-unknown} < 9.0)"
        fi
    fi
    [ "$FP8" = 1 ] && fp8_flag="--fp8"
    ckpt="$(latest_ckpt pretrain)"
    if [ -n "$ckpt" ] && [ "$FORCE" != 1 ]; then
        # Only resume a checkpoint of the same depth — a leftover smoke-test
        # checkpoint would otherwise die with size-mismatch errors on load.
        meta="$(dirname "$ckpt")/$(basename "$ckpt" | sed 's/^model_/meta_/; s/\.pt$/.json/')"
        if grep -q "\"depth\": $DEPTH," "$meta" 2>/dev/null; then
            log "pretrain: resuming from $ckpt"
            resume_flag="--resume-from=$ckpt"
        else
            log "WARNING: ignoring $ckpt (depth != $DEPTH or missing $(basename "$meta")); starting fresh — delete stale runs under $NANOPROOF_HOME/models/pretrain to silence this"
        fi
    fi
    log "pretrain: depth=$DEPTH on Nemotron-CC-Math (~20B tokens; expect ~3-4 days at depth 26)"
    ( cd "$NP_REPO" && "$PY" -m nanoproof.pretrain --depth="$DEPTH" \
        --save-every="$SAVE_EVERY" $fp8_flag $resume_flag )
    touch "$marker"
}

do_midtrain() {
    if stage_done midtrain; then log "midtrain: checkpoint exists, skipping"; return; fi
    local ckpt; ckpt="$(latest_ckpt pretrain)"
    [ -n "$ckpt" ] || die "no pretrain checkpoint found under $NANOPROOF_HOME/models/pretrain"
    log "midtrain: Lean GitHub corpus, from $ckpt"
    ( cd "$NP_REPO" && "$PY" -m nanoproof.midtrain --model-path "$ckpt" )
}

do_sft() {
    if stage_done sft; then log "sft: checkpoint exists, skipping"; return; fi
    local ckpt; ckpt="$(latest_ckpt midtrain)"
    [ -n "$ckpt" ] || die "no midtrain checkpoint found under $NANOPROOF_HOME/models/midtrain"
    log "sft: LeanTree Mathlib transitions, from $ckpt"
    ( cd "$NP_REPO" && "$PY" -m nanoproof.sft --model-path "$ckpt" )
    log "SFT done. Policy+value checkpoint: $(latest_ckpt sft)"
}

# ------------------------------------------------- lean infrastructure ------

do_leanproj() {
    log "leanproj: Lean $LEAN_VERSION project with Mathlib + formal_conjectures"
    # pip-installed leantree has no bundled REPL, so repl_path must be explicit
    [ -x "$REPL_EXE" ] || die "REPL fork not built; run the setup stage"
    if [ ! -d "$LEAN_PROJECT" ]; then
        ( cd "$WORK_DIR" && "$PY" -c "
from leantree import LeanProject
LeanProject.create('$LEAN_PROJECT', lean_version='$LEAN_VERSION', libraries=['mathlib'],
                   repl_path='$REPL_EXE')
" )
    fi
    if ! grep -q formal_conjectures "$LEAN_PROJECT/lakefile.toml"; then
        cat >> "$LEAN_PROJECT/lakefile.toml" <<'EOF'

[[require]]
name = "formal_conjectures"
scope = "google-deepmind"
git = "https://github.com/google-deepmind/formal-conjectures"
rev = "89c6801f9f05cf63105d66843ed70b1e4ceb0c69"
EOF
        # root module = the single .lean file at the project root
        local root_mod
        root_mod="$(find "$LEAN_PROJECT" -maxdepth 1 -name '*.lean' | head -1)"
        [ -n "$root_mod" ] || die "could not find root module in $LEAN_PROJECT"
        printf '\nimport FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot\nimport FormalConjectures.Util.Answer\n' >> "$root_mod"
    fi
    ( cd "$LEAN_PROJECT" && lake update && lake build )
}

do_leanserver() {
    [ -x "$LEANSERVER" ] || LEANSERVER="$(command -v leanserver || true)"
    [ -n "$LEANSERVER" ] || die "leanserver not found; run the setup stage (installed with leantree)"
    [ -x "$REPL_EXE" ] || die "REPL fork not built; run the setup stage"
    [ -d "$LEAN_PROJECT/.lake" ] || die "Lean project not built; run the leanproj stage"
    if [ -f "$SERVER_PID_FILE" ] && kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
        log "leanserver: already running (pid $(cat "$SERVER_PID_FILE"))"; return
    fi
    log "leanserver: $LEAN_PROCS workers on port $PORT (log: $SERVER_LOG)"
    "$LEANSERVER" \
        --project-path "$LEAN_PROJECT" \
        --repl-exe "$REPL_EXE" \
        --imports Mathlib \
            FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot \
            FormalConjectures.Util.Answer \
        --max-processes "$LEAN_PROCS" \
        --address 127.0.0.1 --port "$PORT" \
        --warmup \
        --rss-hard-limit-gib "$RSS_LIMIT" \
        --pss-recycle-limit-gib "$PSS_RECYCLE" \
        > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
    # /status reports ready before imports actually settle (README warning),
    # so give Mathlib imports a fixed head start.
    log "waiting ${WARMUP_WAIT}s for Mathlib import warmup..."
    sleep "$WARMUP_WAIT"
}

# ------------------------------------------------------------- rl / eval -----

do_rl() {
    local ckpt; ckpt="$(latest_ckpt rl || true)"
    [ -n "$ckpt" ] || ckpt="$(latest_ckpt sft)"
    [ -n "$ckpt" ] || die "no sft checkpoint; run training stages first"
    do_leanproj; do_leanserver
    log "rl: single-GPU loop from $ckpt (monitor: http://localhost:5050)"
    ( cd "$NP_REPO" && "$PY" -m nanoproof.rl \
        --model-path "$ckpt" \
        --lean-servers "127.0.0.1:$PORT" \
        --lean-project "$LEAN_PROJECT" \
        ${NP_RL_EXTRA_ARGS:-} )
}

do_eval() {
    local ckpt; ckpt="$(latest_ckpt rl || true)"
    [ -n "$ckpt" ] || ckpt="$(latest_ckpt sft)"
    [ -n "$ckpt" ] || die "no checkpoint to evaluate"
    do_leanproj; do_leanserver
    log "eval: MiniF2F-Valid, 512 simulations, $ckpt"
    ( cd "$NP_REPO" && "$PY" scripts/prover_eval.py \
        --model-path "$ckpt" \
        --lean-servers "127.0.0.1:$PORT" \
        --datasets minif2f --split valid \
        --num-simulations 512 )
}

do_stop() {
    if [ -f "$SERVER_PID_FILE" ]; then
        kill "$(cat "$SERVER_PID_FILE")" 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
        log "leanserver stopped"
    fi
}

# ---------------------------------------------------------------- main -------

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(setup data tokenizer pretrain midtrain sft)

for stage in "${STAGES[@]}"; do
    case "$stage" in
        setup)      do_setup ;;
        data)       do_data ;;
        tokenizer)  do_tokenizer ;;
        smoke)      do_setup; do_data; do_tokenizer; do_smoke ;;
        pretrain)   do_pretrain ;;
        midtrain)   do_midtrain ;;
        sft)        do_sft ;;
        leanproj)   do_leanproj ;;
        leanserver) do_leanserver ;;
        rl)         do_rl ;;
        eval)       do_eval ;;
        stop)       do_stop ;;
        all)        do_setup; do_data; do_tokenizer; do_pretrain; do_midtrain; do_sft; do_rl ;;
        *)          die "unknown stage: $stage" ;;
    esac
done

log "done: ${STAGES[*]}"
