"""Value providers: cost-to-go estimates for proof states.

`as_heurv` adapts any ValueProvider to deepxube's HeurVFn protocol
(states, goals, contexts) -> List[float].
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .states import LeanGoal, LeanState


class ValueProvider(ABC):
    @abstractmethod
    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        ...


class GoalCountValue(ValueProvider):
    """Deterministic, model-free heuristic: one unit per open goal plus mild
    pressure on goal size. Exact zero on solved states."""

    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        return [0.0 if s.solved else sum(1.0 + min(len(g), 400) / 400.0 for g in s.goals)
                for s in states]


def as_heurv(vp: ValueProvider):
    """Wrap a ValueProvider as a deepxube HeurVFn."""
    def heurv(states, goals, contexts) -> List[float]:
        return vp.estimate(states, goals)
    return heurv
