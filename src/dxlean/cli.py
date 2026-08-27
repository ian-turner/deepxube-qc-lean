"""Command-line entry point.

Examples:
  # backbone tactics only, deterministic heuristic (no LLM anywhere):
  dxlean solve --problems problems/dev.jsonl --no-llm

  # local model via any OpenAI-compatible server (ollama, LM Studio, mlx, vllm):
  dxlean solve --problems problems/dev.jsonl \
      --endpoint http://localhost:11434/v1 --model qwen2.5-coder:7b --value judge

  # watch the search think on one theorem (per-iteration narration + tree):
  dxlean viz --problems problems/dev.jsonl --name and_swap --no-llm \
      --backbone "intro h,constructor,assumption,rfl,omega"

  # interactive proof shell (:p asks the providers, :u undoes, :q quits):
  dxlean viz --problems problems/dev.jsonl --name or_swap --interactive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .domain import LeanDomain
from .llm import ChatClient
from .providers import BackboneProvider, LLMSampler, UnionProvider
from .repl import REPLManager
from .solve import solve
from .states import TheoremSpec
from .values import GoalCountValue, LLMJudgeValue

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_problems(path: str) -> List[TheoremSpec]:
    theorems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            theorems.append(TheoremSpec(row["name"], row["statement"]))
    return theorems


def _add_common_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--problems", required=True, help="JSONL file: {name, statement} per line")
    s.add_argument("--project", default=os.path.join(ROOT, "lean", "testproj"), help="Lean project dir")
    s.add_argument("--repl-bin", default=os.path.join(ROOT, "vendor", "repl", ".lake", "build", "bin", "repl"))
    s.add_argument("--header", default="import TestProj", help="Lean header (imports) for the session env")
    s.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    s.add_argument("--model", default=None)
    s.add_argument("--no-llm", action="store_true", help="backbone tactics only")
    s.add_argument("--backbone", default=None,
                   help="comma-separated backbone tactic menu ('' disables backbone; default: built-in menu)")
    s.add_argument("--value", choices=["goalcount", "judge"], default="goalcount")
    s.add_argument("--k", type=int, default=8, help="LLM tactic samples per state")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--cap", type=int, default=24, help="max candidates validated per state")
    s.add_argument("--weight", type=float, default=1.0, help="weight on path cost (W*g + h); lower = greedier")
    s.add_argument("--batch", type=int, default=1, help="nodes expanded per search iteration per instance")
    s.add_argument("--eps", type=float, default=0.0, help="chance of random pop (exploration)")
    s.add_argument("--itr-max", type=int, default=100, help="search iterations per theorem")
    s.add_argument("--tactic-timeout", type=float, default=20.0)


def _build(p: argparse.ArgumentParser, args: argparse.Namespace):
    """Construct (repl, provider, value) from parsed common args."""
    use_llm = not args.no_llm and args.endpoint is not None
    if not args.no_llm and args.endpoint is None:
        print("[dxlean] no --endpoint given: running backbone-only (pass --no-llm to silence)")
    client = None
    if use_llm:
        if args.model is None:
            p.error("--model is required with --endpoint")
        client = ChatClient(args.endpoint, args.model)

    providers = []
    if args.backbone is None:
        providers.append(BackboneProvider())
    elif args.backbone.strip():
        providers.append(BackboneProvider([t.strip() for t in args.backbone.split(",") if t.strip()]))
    if client is not None:
        providers.append(LLMSampler(client, k=args.k, temperature=args.temperature))
    if not providers:
        p.error("no action providers: give --backbone or an --endpoint")
    provider = UnionProvider(providers, cap=args.cap)

    if args.value == "judge":
        if client is None:
            p.error("--value judge requires --endpoint/--model")
        value = LLMJudgeValue(client)
    else:
        value = GoalCountValue()

    repl = REPLManager(args.project, args.repl_bin, header=args.header,
                       tactic_timeout=args.tactic_timeout)
    return repl, provider, value


def _cmd_solve(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    theorems = load_problems(args.problems)
    repl, provider, value = _build(p, args)
    repl.start()
    try:
        domain = LeanDomain(repl, provider, {t.name: t for t in theorems})
        results = solve(theorems, repl, domain, value, weight=args.weight,
                        batch_size=args.batch, eps=args.eps, itr_max=args.itr_max,
                        verbose=not args.quiet)
    finally:
        repl.stop()

    if args.out:
        os.makedirs(os.path.join(args.out, "proofs"), exist_ok=True)
        with open(os.path.join(args.out, "results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps({
                    "name": r.name, "solved": r.solved, "verified": r.verified,
                    "tactics": r.tactics, "path_cost": r.path_cost,
                    "iterations": r.iterations, "message": r.message,
                }) + "\n")
        by_name = {t.name: t for t in theorems}
        for r in results:
            if r.verified:
                with open(os.path.join(args.out, "proofs", f"{r.name}.lean"), "w") as f:
                    f.write(r.proof_text(by_name[r.name].statement))
        print(f"[dxlean] wrote {args.out}/results.jsonl")

    n_solved = sum(r.solved for r in results)
    n_verified = sum(r.verified for r in results)
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

    repl, provider, value = _build(p, args)
    repl.start()
    try:
        if args.interactive:
            viz.interactive(repl, provider, thm, value_provider=value)
        else:
            domain = LeanDomain(repl, provider, {thm.name: thm})
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
