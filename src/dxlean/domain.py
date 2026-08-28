"""LeanDomain: generic Lean theorem proving as a deepxube ActsEnum domain.

The usual deepxube pattern is inverted: tactic applicability is only decidable
by running the tactic, so `get_state_actions` does the real work
(propose -> REPL-validate -> cache successors) and `next_state` is a pure
cache lookup. Failed tactics feed a per-state negative cache that is surfaced
back to providers on re-proposal.

A state whose every candidate fails is re-proposed up to `max_resamples` times
in the same call, with the failed tactics surfaced back to the providers (LLM
samplers put them in the prompt), so one bad sample round from a stochastic
provider does not permanently kill the state. A state still empty after that
returns an empty action list; the instance's frontier then drains and the
search reports it unsolved.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from deepxube.base.domain import ActsEnum, StateGoalVizable, StringToAct
from matplotlib.figure import Figure

from .providers import ActionProvider, Candidate, ProposalRequest
from .repl import REPLManager
from .states import LeanGoal, LeanState, TacticAction, TheoremSpec

StateKey = Tuple[str, str]


class LeanDomain(ActsEnum[LeanState, TacticAction, LeanGoal],
                 StateGoalVizable[LeanState, TacticAction, LeanGoal],
                 StringToAct[LeanState, TacticAction, LeanGoal]):
    def __init__(self, repl: REPLManager, provider: ActionProvider,
                 theorems: Dict[str, TheoremSpec], verbose: bool = False,
                 max_resamples: int = 2):
        super().__init__()
        self.repl = repl
        self.provider = provider
        self.theorems = theorems
        self.verbose = verbose
        self.max_resamples = max_resamples

        self._actions: Dict[StateKey, List[TacticAction]] = {}
        self._successor: Dict[Tuple[StateKey, str], LeanState] = {}
        self._failed: Dict[StateKey, Set[str]] = {}
        self.stats: Dict[str, int] = {
            "expansions": 0, "validations": 0, "valid": 0,
            "errors": 0, "no_progress": 0, "timeouts": 0, "resamples": 0,
        }

    # -- deepxube Domain interface -------------------------------------------

    def get_state_actions(self, states: List[LeanState]) -> List[List[TacticAction]]:
        need = [s for s in states if not s.solved and s.key not in self._actions]
        self._propose_round(need)
        # a freshly expanded state whose every candidate failed gets re-proposed
        # with the failed tactics surfaced back to the providers, so one bad
        # sample round does not permanently dead-end it
        for _ in range(self.max_resamples):
            dead = [s for s in need if not self._actions.get(s.key)]
            if not dead:
                break
            self.stats["resamples"] += len(dead)
            if self.verbose:
                for s in dead:
                    print(f"[dxlean] resampling dead end {s}")
            self._propose_round(dead)

        return [[] if s.solved else list(self._actions.get(s.key, [])) for s in states]

    def _propose_round(self, states: List[LeanState]) -> None:
        if not states:
            return
        reqs = [ProposalRequest(s, self.theorems[s.thm_name],
                                tuple(sorted(self._failed.get(s.key, ()))))
                for s in states]
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
            if res.status in ("ok", "solved"):
                assert res.state is not None
                if res.state == state:  # paranoid: canonicalization missed a no-op
                    failed.add(tac)
                    continue
                self.stats["valid"] += 1
                valid.append(TacticAction(tac, cand.provenance, cand.score))
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
            print(f"[dxlean] expanded {state}: {len(valid)}/{len(cands)} tactics valid")

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
