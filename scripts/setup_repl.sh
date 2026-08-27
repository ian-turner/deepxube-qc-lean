#!/usr/bin/env bash
# Clone and build the leanprover-community REPL at the tag matching lean/testproj's toolchain.
# Result: vendor/repl/.lake/build/bin/repl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN_FILE="$ROOT/lean/testproj/lean-toolchain"
TAG="$(sed 's/.*://' "$TOOLCHAIN_FILE")"   # e.g. leanprover/lean4:v4.30.0 -> v4.30.0
DEST="$ROOT/vendor/repl"

if [ ! -d "$DEST" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/leanprover-community/repl "$DEST"
fi
cd "$DEST"
lake build repl
echo "REPL built: $DEST/.lake/build/bin/repl"
