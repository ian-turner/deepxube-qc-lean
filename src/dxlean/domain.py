"""LeanDomain: generic Lean theorem proving as a deepxube ActsEnum domain.

The usual deepxube pattern is inverted: tactic applicability is only decidable
by running the tactic, so `get_state_actions` does the real work
(propose -> REPL-validate -> cache successors) and `next_state` is a pure
cache lookup. Failed tactics feed a per-state negative cache that is surfaced
back to providers on re-proposal.

A state whose every candidate fails is re-proposed up to `max_resamples` times
in the same call, with the failed tactics surfaced back to the providers (the
nanoproof provider re-samples with a fresh server seed), so one bad sample
round from a stochastic provider does not permanently kill the state. A state
still empty after that returns an empty action list; the instance's frontier
then drains and the search reports it unsolved.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from deepxube.base.domain import ActsEnum, StateGoalVizable, StringToAct
from matplotlib.figure import Figure

from .providers import ActionProvider, Candidate, ProposalRequest
from .repl import ApplyResult, REPLManager
from .states import LeanGoal, LeanState, TacticAction

StateKey = Tuple[str, str]


class LeanDomain(ActsEnum[LeanState, TacticAction, LeanGoal],
                 StateGoalVizable[LeanState, TacticAction, LeanGoal],
                 StringToAct[LeanState, TacticAction, LeanGoal]):
    def __init__(self, repl: REPLManager, provider: ActionProvider, max_resamples: int = 2):
        super().__init__()
        self.repl = repl
        self.provider = provider
        self.max_resamples = max_resamples

        self._actions: Dict[StateKey, List[TacticAction]] = {}
        self._successor: Dict[Tuple[StateKey, str], LeanState] = {}
        self._failed: Dict[StateKey, Set[str]] = {}
        self.stats: Dict[str, int] = {
            "expansions": 0, "validations": 0, "valid": 0,
            "error": 0, "no_progress": 0, "timeout": 0, "resamples": 0,
        }

    # -- deepxube Domain interface -------------------------------------------

    def get_state_actions(self, states: List[LeanState]) -> List[List[TacticAction]]:
        need = [s for s in states if not s.solved and s.key not in self._actions]
        self._propose_round(need)
        for _ in range(self.max_resamples):
            dead = [s for s in need if not self._actions[s.key]]
            if not dead:
                break
            self.stats["resamples"] += len(dead)
            self._propose_round(dead)
        return [[] if s.solved else list(self._actions[s.key]) for s in states]

    def _propose_round(self, states: List[LeanState]) -> None:
        if not states:
            return
        reqs = [ProposalRequest(s, tuple(sorted(self._failed.get(s.key, ())))) for s in states]
        for s, cands in zip(states, self.provider.propose(reqs)):
            self._expand_one(s, cands)

    def _expand_one(self, state: LeanState, cands: List[Candidate]) -> None:
        self.stats["expansions"] += 1
        failed = self._failed.setdefault(state.key, set())
        valid: List[TacticAction] = []
        for cand in cands:
            tac = cand.tactic
            if tac in failed:
                continue
            self.stats["validations"] += 1
            res = self.repl.apply_tactic(state, tac)
            if res.state is not None and res.state == state:  # canonicalization missed a no-op
                res = ApplyResult("no_progress", None, tac)
            if res.state is None:
                self.stats[res.status] += 1
                failed.add(tac)
                continue
            self.stats["valid"] += 1
            valid.append(TacticAction(tac, cand.provenance, cand.score))
            self._successor[(state.key, tac)] = res.state
        self._actions[state.key] = valid

    def next_state(self, states: List[LeanState], actions: List[TacticAction]) -> Tuple[List[LeanState], List[float]]:
        out = [self._successor[(s.key, a.tactic)] for s, a in zip(states, actions)]
        return out, [1.0] * len(out)

    def is_solved(self, states: List[LeanState], goals: List[LeanGoal]) -> List[bool]:
        return [s.solved for s in states]

    def sample_problem_instances(self, num_steps_l, times=None):
        raise NotImplementedError("dxlean feeds problem instances directly; no generator")

    # -- introspection (used by viz) -----------------------------------------

    def expanded(self, state: LeanState) -> bool:
        return state.key in self._actions

    def valid_actions(self, state: LeanState) -> Optional[List[TacticAction]]:
        return self._actions.get(state.key)

    def failed_tactics(self, state: LeanState) -> Set[str]:
        return set(self._failed.get(state.key, ()))

    # -- deepxube visualization mixins ---------------------------------------

    def visualize_state_goal(self, state: LeanState, goal: LeanGoal, fig: Figure) -> None:
        from .viz import render_state_goal
        render_state_goal(state, goal, fig)

    def string_to_action(self, act_str: str) -> Optional[TacticAction]:
        act_str = act_str.strip()
        return TacticAction(act_str, "user") if act_str else None

    def string_to_action_help(self) -> str:
        return "Any Lean 4 tactic, e.g. 'simp', 'omega', 'intro h', 'exact h.2' (validated by the REPL on apply)"
