"""Solve harness: run deepxube batch weighted A* over a list of theorems.

All theorems run as concurrent search instances, so provider and value calls
batch across the whole frontier. Found proofs are re-certified from scratch
(fresh elaboration of the assembled tactic script — the trust anchor).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from deepxube.base.pathfind_fns import PFNsHeurV
from deepxube.base.pathfinding import get_path
from deepxube.pathfinding.graph_search import GraphSearchHeurNodeActsEnum

from .domain import LeanDomain
from .repl import REPLError, REPLManager
from .states import LeanGoal, TheoremSpec
from .values import ValueProvider, as_heurv


@dataclass
class SolveResult:
    name: str
    solved: bool
    verified: bool = False
    tactics: List[str] = field(default_factory=list)
    path_cost: float = 0.0
    iterations: int = 0
    message: str = ""

    def proof_text(self, statement: str) -> str:
        body = "\n".join(f"  {t}" for t in self.tactics)
        return f"{statement} := by\n{body}\n"


def solve(theorems: List[TheoremSpec], repl: REPLManager, domain: LeanDomain,
          value_provider: ValueProvider, weight: float = 1.0, batch_size: int = 1,
          eps: float = 0.0, itr_max: int = 100, verbose: bool = False) -> List[SolveResult]:
    search = GraphSearchHeurNodeActsEnum(
        domain, PFNsHeurV(heurv=as_heurv(value_provider)),
        batch_size=batch_size, weight=weight, eps=eps)

    results: dict[str, SolveResult] = {}

    # roots: theorems that fail to elaborate become error results immediately
    states, goals, names = [], [], []
    for thm in theorems:
        try:
            states.append(repl.init_theorem(thm))
            goals.append(LeanGoal(thm))
            names.append(thm.name)
        except REPLError as e:
            results[thm.name] = SolveResult(thm.name, False, message=f"init failed: {e}")

    if states:
        instances = search.make_instances(states, goals, inst_infos=names)
        search.add_instances(instances)

    t0 = time.time()
    while search.instances:
        search.step(verbose=False)
        for inst in search.remove_finished_instances(itr_max):
            name = inst.inst_info
            thm = domain.theorems[name]
            res = SolveResult(name, inst.has_soln(), iterations=inst.itr)
            if inst.has_soln():
                _, actions, _, path_cost = get_path(inst.goal_node)
                res.tactics = [a.tactic for a in actions]
                res.path_cost = path_cost
                ok, msg = repl.check_full_proof(thm, tuple(res.tactics))
                res.verified = ok
                if not ok:
                    res.message = f"verification failed: {msg}"
            else:
                res.message = "search exhausted" if inst.frontier_size() == 0 else "iteration limit"
            results[name] = res
            if verbose:
                status = "SOLVED+VERIFIED" if res.verified else ("SOLVED (verify FAILED)" if res.solved else "unsolved")
                print(f"[dxlean] {name}: {status} in {inst.itr} itrs"
                      + (f" | {' ; '.join(res.tactics)}" if res.tactics else f" | {res.message}"))

    if verbose:
        wall = time.time() - t0
        n_ok = sum(r.solved for r in results.values())
        print(f"[dxlean] {n_ok}/{len(theorems)} solved in {wall:.1f}s | domain stats: {domain.stats} "
              f"| repl requests: {repl.n_requests}, restarts: {repl.n_restarts}")

    return [results[t.name] for t in theorems if t.name in results]
