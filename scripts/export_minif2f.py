#!/usr/bin/env python3
"""Export miniF2F (google-deepmind/miniF2F, Lean 4) as dxlean problem files.

Parses Valid.lean / Test.lean exactly as nanoproof's loader does
(nanoproof/data/bench/minif2f.py: a theorem starts at a line beginning with
`theorem` and ends at its `sorry` line; the id is the theorem name; two test
theorems that ship with proofs are patched back to `sorry`), so `name` here equals
`id` in nanoproof's theorems.jsonl and scripts/compare.py can join the two.

    scripts/export_minif2f.py     # -> problems/minif2f_{valid,test}.jsonl + minif2f_header.lean

The header file holds the imports and `open scoped` preamble the statements need
(`answer(...)` comes from formal_conjectures, which the training script's leanproj
stage adds to the Mathlib project; these are the imports nanoproof's leanserver is
started with). Pass it as --header "$(cat problems/minif2f_header.lean)".
"""
import json
import os
import re
import urllib.request
from typing import List, Tuple

BASE_URL = "https://raw.githubusercontent.com/google-deepmind/miniF2F/refs/heads/main/MiniF2F/"
FILES = {"valid": "Valid.lean", "test": "Test.lean"}
COUNTS = {"valid": 256, "test": 244}
# upstream test theorems shipped with proofs instead of stubs (nanoproof patches the same two)
PATCHES = {"test": [
    ("\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 :=\n  rfl\n",
     "\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 := by\n  sorry\n"),
    ("\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  norm_num [div_pow]\n",
     "\ntheorem mathd_algebra_302 : (Complex.I / 2) ^ 2 = -(1 / 4) := by\n  sorry\n"),
]}
HEADER = """import Mathlib
import FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot
import FormalConjectures.Util.Answer
open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial
"""
_NAME = re.compile(r"\btheorem\s+(\S+)")
_TAIL = re.compile(r"\s*:=\s*by\s*$")


def parse(text: str, split: str = "valid") -> List[Tuple[str, str]]:
    """(name, statement) per theorem, statement without the `:= by sorry` tail."""
    for old, new in PATCHES.get(split, ()):
        text = text.replace(old, new)
    out, cur = [], None
    for line in text.split("\n"):
        if line.lstrip().startswith("theorem"):
            assert cur is None, "overlapping theorems"
            cur = [line]
        elif line.lstrip().startswith("sorry"):
            assert cur is not None, "sorry without theorem"
            stmt, n = _TAIL.subn("", "\n".join(cur))
            assert n == 1 and "sorry" not in stmt, stmt
            out.append((_NAME.search(stmt).group(1), stmt))
            cur = None
        elif cur is not None:
            cur.append(line)
    assert len({name for name, _ in out}) == len(out), "duplicate theorem ids"
    return out


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "problems")
    header_path = os.path.join(out_dir, "minif2f_header.lean")
    with open(header_path, "w") as f:
        f.write(HEADER)
    for split, filename in FILES.items():
        url = BASE_URL + filename
        with urllib.request.urlopen(url) as resp:
            rows = parse(resp.read().decode(), split)
        assert len(rows) == COUNTS[split], f"{split}: expected {COUNTS[split]} theorems, got {len(rows)}"
        out = os.path.join(out_dir, f"minif2f_{split}.jsonl")
        with open(out, "w") as f:
            f.write(f"# miniF2F {split} ({len(rows)} theorems) from {url}; "
                    f"run with --header \"$(cat problems/minif2f_header.lean)\"\n")
            for name, stmt in rows:
                f.write(json.dumps({"name": name, "statement": stmt}, ensure_ascii=False) + "\n")
        print(f"wrote {out} ({len(rows)} theorems)")
    print(f"wrote {header_path}")


if __name__ == "__main__":
    main()
