"""Core types: theorem specs, proof states, and tactic actions.

A LeanState is a full Lean proof state (all remaining goals) reached from a
theorem's root `sorry` by a sequence of tactics. Identity (hash/eq) is the
canonicalized goal text plus the theorem name, so transposition merging in
deepxube's CLOSED dict works across different tactic orderings, while replay
(needed after a REPL restart) stays anchored to a concrete root.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from deepxube.base.domain import Action, Goal, State

_WS = re.compile(r"\s+")


def canonical_goals(goals: Tuple[str, ...]) -> str:
    """Whitespace-collapsed, order-preserving rendering of a goal list."""
    return "\n<goal>\n".join(_WS.sub(" ", g).strip() for g in goals)


@dataclass(frozen=True)
class TheoremSpec:
    """A theorem to prove. `statement` is the declaration without a proof,
    e.g. 'theorem foo (n : Nat) : n + 0 = n'. The harness appends ':= by sorry'.
    """
    name: str
    statement: str


class LeanGoal(Goal):
    """deepxube Goal: prove the given theorem (solved = no goals remaining)."""

    def __init__(self, theorem: TheoremSpec):
        self.theorem = theorem

    def __repr__(self) -> str:
        return f"LeanGoal({self.theorem.name})"


class LeanState(State):
    def __init__(self, thm_name: str, goals: Tuple[str, ...], tactics: Tuple[str, ...]):
        self.thm_name = thm_name
        self.goals = goals
        self.tactics = tactics
        self._key = (thm_name, canonical_goals(goals))

    @property
    def key(self) -> Tuple[str, str]:
        return self._key

    @property
    def solved(self) -> bool:
        return len(self.goals) == 0

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LeanState) and self._key == other._key

    def __repr__(self) -> str:
        if self.solved:
            return f"LeanState({self.thm_name}, SOLVED, {len(self.tactics)} tactics)"
        head = self.goals[0].replace("\n", " ")[:60]
        return f"LeanState({self.thm_name}, {len(self.goals)} goals, '{head}...')"


class TacticAction(Action):
    """A single tactic string. Identity is the tactic text only, so the same
    tactic proposed by different providers is one action."""

    def __init__(self, tactic: str, provenance: str = "?", score: Optional[float] = None):
        self.tactic = tactic
        self.provenance = provenance
        self.score = score

    def __hash__(self) -> int:
        return hash(self.tactic)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TacticAction) and self.tactic == other.tactic

    def __repr__(self) -> str:
        return self.tactic
