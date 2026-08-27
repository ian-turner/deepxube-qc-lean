"""Value providers: cost-to-go estimates for proof states.

`as_heurv` adapts any ValueProvider to deepxube's HeurVFn protocol
(states, goals, contexts) -> List[float]. Estimates are cached by state key —
graph search re-evaluates transpositions freely, LLM judges must not pay twice.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .states import LeanGoal, LeanState

_INT = re.compile(r"-?\d+")


class ValueProvider(ABC):
    @abstractmethod
    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        ...


class GoalCountValue(ValueProvider):
    """Deterministic, model-free heuristic: one unit per open goal plus mild
    pressure on goal size. Exact zero on solved states."""

    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        out: List[float] = []
        for s in states:
            if s.solved:
                out.append(0.0)
            else:
                out.append(sum(1.0 + min(len(g), 400) / 400.0 for g in s.goals))
        return out


JUDGE_SYSTEM = """You are an expert Lean 4 proof assistant. Estimate how many tactic steps a \
competent prover needs to finish the CURRENT proof state. Reply with a single integer only: \
0 if already finished, 1 if one standard tactic closes everything, larger for harder states. \
Use 25 or more if the state looks unprovable or hopeless."""

JUDGE_USER = """Theorem:
{statement}

Current goals:
{goals}

Estimated tactic steps remaining (integer only):"""


class LLMJudgeValue(ValueProvider):
    def __init__(self, client, default: float = 8.0, cap: float = 50.0):
        self.client = client
        self.default = default
        self.cap = cap
        self._cache: Dict[Tuple[str, str], float] = {}

    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        out: List[float] = []
        for s, g in zip(states, goals):
            if s.solved:
                out.append(0.0)
                continue
            cached = self._cache.get(s.key)
            if cached is not None:
                out.append(cached)
                continue
            user = JUDGE_USER.format(statement=g.theorem.statement, goals="\n\n".join(s.goals))
            try:
                text = self.client.chat(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=16)
                m = _INT.search(text)
                val = float(min(max(int(m.group()), 0), self.cap)) if m else self.default
            except Exception as e:
                print(f"[dxlean] LLM judge error ({type(e).__name__}): {e}")
                val = self.default
            self._cache[s.key] = val
            out.append(val)
        return out


def as_heurv(vp: ValueProvider):
    """Wrap a ValueProvider as a deepxube HeurVFn."""
    def heurv(states, goals, contexts) -> List[float]:
        return vp.estimate(states, goals)
    return heurv
