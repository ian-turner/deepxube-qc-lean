#!/usr/bin/env bash
# deepxube A* with the nanoproof checkpoint on miniF2F: serves the model, runs one
# `dxlean solve` per grid variant at the same model-call budget, stops the server.
#   NP_CKPT=<model_NNNNNN.pt> scripts/bench_dxlean.sh
# Knobs: BUDGET  model calls per theorem (default 512)
#        SPLIT   valid | test (default valid)
#        PILOT   first N theorems only, in file order (default: all)
#        GRID    variants, one run each: wW (--weight), bB (--batch), vMODE (--np-value);
#                default "w1.0 w0.5 b4"
#        OUT     results dir (default results/minif2f_<split>_<budget>), one subdir per variant
#        DXLEAN  the dxlean executable (default .venv/bin/dxlean)
#        NP_WORK_DIR, NP_INFER_PORT, NP_NUM_SAMPLES, NP_FIRST_TOKEN_CAP, NP_DISABLE_SOLVERS
#        as in train_nanoproof.sh (keep them equal to the bench_nanoproof.sh run)
# Runs on the GPU node next to the Mathlib project the leanproj stage built.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NP_CKPT:?set NP_CKPT to the model_NNNNNN.pt checkpoint}"
BUDGET="${BUDGET:-512}"; SPLIT="${SPLIT:-valid}"; GRID="${GRID:-w1.0 w0.5 b4}"
OUT="${OUT:-results/minif2f_${SPLIT}_${BUDGET}}"; mkdir -p "$OUT"
DXLEAN="${DXLEAN:-.venv/bin/dxlean}"
PORT="${NP_INFER_PORT:-5001}"
PROJECT="${NP_WORK_DIR:-/work/$USER}/nptraining"
REPL=vendor/repl-v4.27.0/.lake/build/bin/repl
PROBLEMS="problems/minif2f_${SPLIT}.jsonl"

if [ -n "${PILOT:-}" ]; then   # same theorems prover_eval --max-theorems takes
    { head -1 "$PROBLEMS"; grep -v '^#' "$PROBLEMS" | head -n "$PILOT"; } > "$OUT/pilot.jsonl"
    PROBLEMS="$OUT/pilot.jsonl"
fi

[ -x "$REPL" ] || scripts/setup_repl.sh nanoproof
scripts/train_nanoproof.sh serve > "$OUT/serve.log" 2>&1 &
trap 'pkill -P $! 2>/dev/null; kill $! 2>/dev/null' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do sleep 5; done

for v in $GRID; do
    case $v in
        w*) flags="--weight ${v#w}" ;;
        b*) flags="--batch ${v#b}" ;;
        v*) flags="--np-value ${v#v}" ;;
        *)  echo "unknown variant $v (wW | bB | vMODE)" >&2; exit 1 ;;
    esac
    "$DXLEAN" solve --problems "$PROBLEMS" --nanoproof "http://localhost:$PORT" --backbone "" \
        --project "$PROJECT" --header "$(cat problems/minif2f_header.lean)" --repl-bin "$REPL" \
        --calls-max "$BUDGET" $flags --out "$OUT/$v" --quiet \
        || true   # exit 1 only means not every theorem was solved
done
echo "results: $OUT/{${GRID// /,}}  ->  scripts/compare.py <nanoproof eval dir> $OUT/*/"
