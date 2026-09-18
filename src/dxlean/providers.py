"""Action providers: sources of candidate tactics for a proof state.

Providers only PROPOSE; validation against the REPL happens in the domain
(propose -> validate -> cache). Batch-in/batch-out so a model backend can
serve a whole search frontier per call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .states import LeanState


@dataclass(frozen=True)
class Candidate:
    tactic: str
    provenance: str = "?"
    score: Optional[float] = None


@dataclass(frozen=True)
class ProposalRequest:
    state: LeanState
    failed: Tuple[str, ...] = ()  # tactics already known to fail on this state


class ActionProvider(ABC):
    @abstractmethod
    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        ...


DEFAULT_BACKBONE = [
    "rfl",
    "decide",
    "assumption",
    "simp",
    "simp_all",
    "omega",
    "constructor",
    "intro h",
    "exact?",
]


class BackboneProvider(ActionProvider):
    """Fixed menu of always-worth-trying tactics. Guaranteed recall on routine
    closers; the REPL filters whatever does not apply. Also the model-free
    path for tests and smoke runs without a nanoproof server."""

    def __init__(self, menu: Optional[Sequence[str]] = None):
        self.menu = list(menu) if menu is not None else list(DEFAULT_BACKBONE)

    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        return [[Candidate(t, "backbone") for t in self.menu] for _ in reqs]


class UnionProvider(ActionProvider):
    """Merge providers in order, dedupe by tactic text, cap per state."""

    def __init__(self, providers: Sequence[ActionProvider], cap: int = 24):
        self.providers = list(providers)
        self.cap = cap

    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        merged: List[List[Candidate]] = [[] for _ in reqs]
        seen: List[set] = [set() for _ in reqs]
        for provider in self.providers:
            for i, cands in enumerate(provider.propose(reqs)):
                for c in cands:
                    if c.tactic not in seen[i] and len(merged[i]) < self.cap:
                        seen[i].add(c.tactic)
                        merged[i].append(c)
        return merged
