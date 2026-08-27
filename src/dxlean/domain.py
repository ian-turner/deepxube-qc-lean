"""LeanDomain: generic Lean theorem proving as a deepxube ActsEnum domain.

The usual deepxube pattern is inverted: tactic applicability is only decidable
by running the tactic, so `get_state_actions` does the real work
(propose -> REPL-validate -> cache successors) and `next_state` is a pure
cache lookup. Failed tactics feed a per-state negative cache that is surfaced
back to providers on re-expansion.

States that produce zero valid tactics simply return an empty action list;
the instance's frontier then drains and the search reports it unsolved.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from deepxube.base.domain import ActsEnum

from .providers import ActionProvider, ProposalRequest
from .repl import REPLManager
from .states import LeanGoal, LeanState, TacticAction, TheoremSpec

StateKey = Tuple[str, str]


class LeanDomain(ActsEnum[LeanState, TacticAction, LeanGoal]):
    def __init__(self, repl: REPLManager, provider: ActionProvider,
                 theorems: Dict[str, TheoremSpec], verbose: bool = False):
        super().__init__()
        self.repl = repl
        self.provider = provider
        self.theorems = theorems
        self.verbose = verbose

        self._actions: Dict[StateKey, List[TacticAction]] = {}
        self._successor: Dict[Tuple[StateKey, str], LeanState] = {}
        self._failed: Dict[StateKey, Set[str]] = {}
        self.stats: Dict[str, int] = {
            "expansions": 0, "validations": 0, "valid": 0,
            "errors": 0, "no_progress": 0, "timeouts": 0,
        }

    # -- deepxube Domain interface -------------------------------------------

    def get_state_actions(self, states: List[LeanState]) -> List[List[TacticAction]]:
        # build proposal batch for states that need expansion
        todo: List[int] = []
        reqs: List[ProposalRequest] = []
        for i, s in enumerate(states):
            if s.solved or s.key in self._actions:
                continue
            todo.append(i)
            failed = tuple(sorted(self._failed.get(s.key, ())))
            reqs.append(ProposalRequest(s, self.theorems[s.thm_name], failed))

        if reqs:
            cands_l = self.provider.propose(reqs)
            for i, cands in zip(todo, cands_l):
                self._expand_one(states[i], [c.tactic for c in cands])

        return [[] if s.solved else list(self._actions.get(s.key, [])) for s in states]

    def _expand_one(self, state: LeanState, tactics: List[str]) -> None:
        self.stats["expansions"] += 1
        failed = self._failed.setdefault(state.key, set())
        valid: List[TacticAction] = []
        for tac in tactics:
            if tac in failed:
                continue
            self.stats["validations"] += 1
            res = self.repl.apply_tactic(state, tac)
            if res.status in ("ok", "solved"):
                assert res.state is not None
                if res.state == state:  # paranoid: canonicalization missed a no-op
                    failed.add(tac)
                    continue
                self.stats["valid"] += 1
                valid.append(TacticAction(tac))
                self._successor[(state.key, tac)] = res.state
            else:
                if res.status == "no_progress":
                    self.stats["no_progress"] += 1
                elif res.status == "timeout":
                    self.stats["timeouts"] += 1
                else:
                    self.stats["errors"] += 1
                failed.add(tac)
                if self.verbose:
                    print(f"[dxlean]   {state.thm_name}: {tac!r} -> {res.status}: {res.message[:100]}")
        self._actions[state.key] = valid
        if self.verbose:
            print(f"[dxlean] expanded {state}: {len(valid)}/{len(tactics)} tactics valid")

    def next_state(self, states: List[LeanState], actions: List[TacticAction]) -> Tuple[List[LeanState], List[float]]:
        out: List[LeanState] = []
        for s, a in zip(states, actions):
            nxt = self._successor.get((s.key, a.tactic))
            assert nxt is not None, f"next_state before validation: {s} {a}"
            out.append(nxt)
        return out, [1.0] * len(out)

    def is_solved(self, states: List[LeanState], goals: List[LeanGoal]) -> List[bool]:
        return [s.solved for s in states]

    def sample_state_action(self, states: List[LeanState]) -> List[TacticAction]:
        import random
        actions_l = self.get_state_actions(states)
        return [random.choice(acts) if acts else TacticAction("skip") for acts in actions_l]

    def sample_problem_instances(self, num_steps_l, times=None):
        raise NotImplementedError("dxlean v1 feeds problem instances directly; no generator")
