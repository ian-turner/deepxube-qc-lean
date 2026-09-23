"""Command-line entry point.

Examples:
  # pre-trained nanoproof policy+value served from the GPU cluster
  # (scripts/train_nanoproof.sh serve, then ssh -L 5001:localhost:5001 <node>):
  dxlean solve --problems problems/dev.jsonl --nanoproof http://localhost:5001 --backbone ""

  # no model anywhere: backbone tactic menu + goal-count heuristic (smoke test):
  dxlean solve --problems problems/dev.jsonl

  # watch the search think on one theorem (per-iteration narration + tree):
  dxlean viz --problems problems/dev.jsonl --name and_swap \
      --backbone "intro h,constructor,assumption,rfl,omega"

  # interactive proof shell (:p asks the providers, :u undoes, :q quits):
  dxlean viz --problems problems/dev.jsonl --name or_swap --interactive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import List, Optional, Tuple

from .domain import LeanDomain
from .nanoproof import (GOAL_MODES, VALUE_MODES, NanoproofClient, NanoproofOracle,
                        NanoproofProvider, NanoproofValue)
from .providers import ActionProvider, BackboneProvider, UnionProvider
from .repl import REPLManager
from .solve import solve
from .states import TheoremSpec
from .values import GoalCountValue, ValueProvider

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_problems(path: str) -> List[TheoremSpec]:
    theorems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                theorems.append(TheoremSpec(row["name"], row["statement"]))
    return theorems


def _add_common_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--problems", required=True, help="JSONL file: {name, statement} per line")
    s.add_argument("--project", default=os.path.join(ROOT, "lean", "testproj"), help="Lean project dir")
    s.add_argument("--repl-bin", default=os.path.join(ROOT, "vendor", "repl", ".lake", "build", "bin", "repl"))
    s.add_argument("--header", default="import TestProj", help="Lean header (imports) for the session env")
    s.add_argument("--nanoproof", default=None, metavar="URL",
                   help="nanoproof inference server, e.g. http://localhost:5001 "
                        "(scripts/train_nanoproof.sh serve; ssh -L to reach it from a laptop)")
    s.add_argument("--np-goals", choices=GOAL_MODES, default="first",
                   help="goals shown to the nanoproof policy: the first goal (as in nanoproof's "
                        "own factorized search) or all goals joined")
    s.add_argument("--np-value", choices=VALUE_MODES, default="sum",
                   help="nanoproof h: sum of per-goal depth predictions, first goal only, or all goals joined")
    s.add_argument("--np-chunk", type=int, default=32, help="states per nanoproof HTTP request")
    s.add_argument("--backbone", default=None,
                   help="comma-separated backbone tactic menu ('' disables backbone; default: built-in menu)")
    s.add_argument("--value", choices=["goalcount", "nanoproof"], default=None,
                   help="heuristic (default: nanoproof when --nanoproof is given, else goalcount)")
    s.add_argument("--cap", type=int, default=24, help="max candidates validated per state")
    s.add_argument("--resamples", type=int, default=2,
                   help="re-proposal rounds (failed tactics fed back) before a state is a dead end")
    s.add_argument("--weight", type=float, default=1.0, help="weight on path cost (W*g + h); lower = greedier")
    s.add_argument("--batch", type=int, default=1, help="nodes expanded per search iteration per instance")
    s.add_argument("--eps", type=float, default=0.0, help="chance of random pop (exploration)")
    s.add_argument("--itr-max", type=int, default=None,
                   help="search iterations per theorem (default 100, unlimited with --calls-max)")
    s.add_argument("--calls-max", type=int, default=None,
                   help="model calls per theorem, nanoproof's simulation budget (default: unlimited)")
    s.add_argument("--tactic-timeout", type=float, default=20.0)
    s.add_argument("--cmd-timeout", type=float, default=600.0,
                   help="seconds allowed for header import, root elaboration and full-proof checks "
                        "(a cold `import Mathlib` can take several minutes)")


def _build(p: argparse.ArgumentParser, args: argparse.Namespace
           ) -> Tuple[REPLManager, LeanDomain, ValueProvider, Optional[NanoproofOracle]]:
    if args.itr_max is None:
        args.itr_max = 10**9 if args.calls_max else 100  # one budget knob at a time
    ledger = defaultdict(Counter)  # per-theorem model calls + validations, shared by oracle and domain
    providers: List[ActionProvider] = []
    if args.backbone is None:
        providers.append(BackboneProvider())
    elif args.backbone.strip():
        providers.append(BackboneProvider([t.strip() for t in args.backbone.split(",") if t.strip()]))

    oracle = None
    if args.nanoproof:
        oracle = NanoproofOracle(NanoproofClient(args.nanoproof, chunk=args.np_chunk),
                                 goal_mode=args.np_goals, value_mode=args.np_value, ledger=ledger)
        try:
            oracle.client.health()
        except Exception as e:
            p.error(f"nanoproof server {args.nanoproof} not reachable: {e}")
        providers.append(NanoproofProvider(oracle))
    if not providers:
        p.error("no action providers: give --backbone or --nanoproof")

    value_choice = args.value or ("nanoproof" if oracle is not None else "goalcount")
    if value_choice == "nanoproof":
        if oracle is None:
            p.error("--value nanoproof requires --nanoproof URL")
        value: ValueProvider = NanoproofValue(oracle)
    else:
        value = GoalCountValue()

    repl = REPLManager(args.project, args.repl_bin, header=args.header,
                       tactic_timeout=args.tactic_timeout, cmd_timeout=args.cmd_timeout)
    domain = LeanDomain(repl, UnionProvider(providers, cap=args.cap), max_resamples=args.resamples,
                        ledger=ledger)
    return repl, domain, value, oracle


def _cmd_solve(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    theorems = load_problems(args.problems)
    repl, domain, value, oracle = _build(p, args)
    t0 = time.time()
    repl.start()
    try:
        results = solve(theorems, repl, domain, value, weight=args.weight,
                        batch_size=args.batch, eps=args.eps, itr_max=args.itr_max,
                        calls_max=args.calls_max, verbose=not args.quiet)
    finally:
        repl.stop()
    elapsed = time.time() - t0
    if oracle is not None and not args.quiet:
        print(f"[dxlean] nanoproof: {oracle.stats} | http requests: {oracle.client.n_requests}, "
              f"busy retries: {oracle.client.n_busy}")

    n_solved = sum(r.solved for r in results)
    n_verified = sum(r.verified for r in results)
    if args.out:
        os.makedirs(os.path.join(args.out, "proofs"), exist_ok=True)
        with open(os.path.join(args.out, "results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps({k: v for k, v in asdict(r).items() if k != "proof"}) + "\n")
        for r in results:
            if r.verified:
                with open(os.path.join(args.out, "proofs", f"{r.name}.lean"), "w") as f:
                    f.write(r.proof)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"total": len(results), "solved": n_solved, "verified": n_verified,
                       "elapsed_seconds": elapsed,
                       "model_calls": sum(r.model_calls for r in results),
                       "validations": sum(r.validations for r in results),
                       "repl_requests": repl.n_requests, "repl_restarts": repl.n_restarts,
                       "args": vars(args)}, f, indent=1)
        print(f"[dxlean] wrote {args.out}/results.jsonl and summary.json")

    print(f"[dxlean] solved {n_solved}/{len(results)}, verified {n_verified}/{len(results)}")
    sys.exit(0 if n_solved == len(results) else 1)


def _cmd_viz(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    from . import viz

    theorems = load_problems(args.problems)
    by_name = {t.name: t for t in theorems}
    if args.name is None:
        thm = theorems[0]
    elif args.name in by_name:
        thm = by_name[args.name]
    else:
        p.error(f"unknown theorem {args.name!r}; available: {', '.join(by_name)}")

    repl, domain, value, _ = _build(p, args)
    repl.start()
    try:
        if args.interactive:
            viz.interactive(repl, domain.provider, value, thm)
        else:
            viz.traced_search(thm, repl, domain, value, weight=args.weight,
                              batch_size=args.batch, eps=args.eps, itr_max=args.itr_max,
                              fig_path=args.fig)
    finally:
        repl.stop()


def main() -> None:
    p = argparse.ArgumentParser(prog="dxlean", description="Lean theorem proving with deepxube search")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve a JSONL problem set")
    _add_common_args(s)
    s.add_argument("--out", default=None, help="output dir for results.jsonl and proofs/")
    s.add_argument("--quiet", action="store_true")

    v = sub.add_parser("viz", help="visualize the environment or the search process for one theorem")
    _add_common_args(v)
    v.add_argument("--name", default=None, help="theorem name from the problem file (default: first)")
    v.add_argument("--interactive", action="store_true", help="proof shell instead of traced search")
    v.add_argument("--fig", default=None, help="also render the search tree as a matplotlib figure to this PNG")

    args = p.parse_args()
    if args.command == "solve":
        _cmd_solve(p, args)
    else:
        _cmd_viz(p, args)


if __name__ == "__main__":
    main()
