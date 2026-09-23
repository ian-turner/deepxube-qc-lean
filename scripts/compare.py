#!/usr/bin/env python3
"""Compare nanoproof's MCTS eval with dxlean runs on the same theorems.

    scripts/compare.py <nanoproof eval dir> <dxlean out dir>...   (any mix, any number)

The budget currency is model calls: nanoproof's `num_iterations` (one MCTS
simulation = one model call) and dxlean's `model_calls` (ledger). A theorem counts
as solved within budget t iff it was proved, and certified, in <= t calls (nanoproof's
success_rate_by_simulations rule; nanoproof rows with an `error` are unsolved). Each
run's own budget comes from its summary (num_simulations / --calls-max), else from
the largest count seen; the headline uses it and larger thresholds print '-'.
"""
import json
import os
import sys
import tomllib
from typing import Dict, List, Optional, Tuple

THRESHOLDS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
Run = Tuple[str, int, Dict[str, Tuple[bool, int]]]  # label, budget, theorem -> (proved, calls)


def load(path: str) -> Run:
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    rows: Dict[str, Tuple[bool, int]] = {}
    rows_path = path if os.path.isfile(path) else next(
        p for p in (os.path.join(d, n) for n in ("theorems.jsonl", "results.jsonl")) if os.path.exists(p))
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            if "id" in row:  # nanoproof
                rows[row["id"]] = (row.get("proof") is not None and row.get("error") is None, row["num_iterations"])
            else:            # dxlean
                rows[row["name"]] = (row["verified"], row["model_calls"])
    budget: Optional[int] = None
    if os.path.exists(os.path.join(d, "summary.toml")):
        with open(os.path.join(d, "summary.toml"), "rb") as f:
            budget = tomllib.load(f)["num_simulations"]
    elif os.path.exists(os.path.join(d, "summary.json")):
        with open(os.path.join(d, "summary.json")) as f:
            budget = json.load(f)["args"]["calls_max"]
    return os.path.basename(d), budget or max(c for _, c in rows.values()), rows


def curve(run: Run, names: List[str]) -> Dict[int, Optional[float]]:
    _, budget, rows = run
    return {t: sum(ok and c <= t for ok, c in (rows[n] for n in names)) / len(names) if t <= budget else None
            for t in THRESHOLDS}


def report(runs: List[Run]) -> str:
    names = sorted(set.intersection(*(set(r[2]) for r in runs)))
    out = []
    if any(len(r[2]) != len(names) for r in runs):
        out.append(f"note: runs cover different theorem sets; comparing the {len(names)} in common")
    w = max(len(r[0]) for r in runs)
    out.append(f"{'run':<{w}}  budget  solved     " + "".join(f"@{t:<6}" for t in THRESHOLDS))
    for run in runs:
        label, budget, rows = run
        n = sum(ok and c <= budget for ok, c in (rows[k] for k in names))
        out.append(f"{label:<{w}}  {budget:>6}  {n:>3}/{len(names)} {100 * n / len(names):5.1f}%  "
                   + "".join("   -   " if v is None else f"{100 * v:5.1f}% " for v in curve(run, names).values()))
    diff = [k for k in names if len({r[2][k][0] for r in runs}) > 1]
    if diff:
        out.append(f"\n{len(diff)} theorems solved by some runs only (model calls at solve, '-' unsolved):")
        for k in diff:
            out.append(f"  {k}: " + "  ".join(f"{l}={r[k][1] if r[k][0] else '-'}" for l, _, r in runs))
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(report([load(p) for p in sys.argv[1:]]))
