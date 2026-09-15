#!/usr/bin/env bash
# Clone and build the leanprover-community REPL.
#   ./scripts/setup_repl.sh            tag from lean/testproj/lean-toolchain -> vendor/repl
#   ./scripts/setup_repl.sh nanoproof  Lean version pinned in train_nanoproof.sh -> vendor/repl-<ver>
#   ./scripts/setup_repl.sh v4.27.0    explicit tag -> vendor/repl-<tag>
# The REPL binary must be built with the same Lean version as the project it runs in;
# pass it to dxlean with --repl-bin (the project supplies Mathlib via `lake env`).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN_FILE="$ROOT/lean/testproj/lean-toolchain"
DEFAULT_TAG="$(sed 's/.*://' "$TOOLCHAIN_FILE")"   # e.g. leanprover/lean4:v4.30.0 -> v4.30.0

ARG="${1:-}"
if [ "$ARG" = "nanoproof" ]; then
    # read the version pinned in the training script so it stays in sync automatically
    TAG="$(grep -m1 '^LEAN_VERSION=' "$ROOT/scripts/train_nanoproof.sh" | cut -d'"' -f2)"
    [ -n "$TAG" ] || { echo "error: LEAN_VERSION not found in train_nanoproof.sh" >&2; exit 1; }
else
    TAG="${ARG:-$DEFAULT_TAG}"
fi

if [ "$TAG" = "$DEFAULT_TAG" ]; then
    DEST="$ROOT/vendor/repl"
else
    DEST="$ROOT/vendor/repl-$TAG"
fi

if [ ! -d "$DEST" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/leanprover-community/repl "$DEST"
fi
cd "$DEST"
lake build repl
echo "REPL built: $DEST/.lake/build/bin/repl"
