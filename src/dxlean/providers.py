"""Action providers: sources of candidate tactics for a proof state.

Providers only PROPOSE; validation against the REPL happens in the domain
(propose -> validate -> cache). Batch-in/batch-out so an LLM backend can
serve a whole search frontier per call.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .repl import BANNED_SUBSTRINGS
from .states import LeanState, TheoremSpec


@dataclass(frozen=True)
class Candidate:
    tactic: str
    provenance: str = "?"
    score: Optional[float] = None


@dataclass(frozen=True)
class ProposalRequest:
    state: LeanState
    theorem: TheoremSpec
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
    closers; the REPL filters whatever does not apply."""

    def __init__(self, menu: Optional[Sequence[str]] = None):
        self.menu = list(menu) if menu is not None else list(DEFAULT_BACKBONE)

    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        return [[Candidate(t, "backbone") for t in self.menu] for _ in reqs]


_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")


def parse_tactic_lines(text: str, k: int) -> List[str]:
    """Extract up to k tactic candidates from an LLM response: one per line,
    strip bullets/numbering/backticks, drop fences, comments, and banned text."""
    out: List[str] = []
    seen = set()
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("--") or stripped.startswith("```"):
            continue
        line = _LINE_PREFIX.sub("", stripped).strip().strip("`").strip()
        if not line or line.startswith("--"):
            continue
        low = line.lower()
        if any(b in low for b in BANNED_SUBSTRINGS):
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
        if len(out) >= k:
            break
    return out


SAMPLER_SYSTEM = """You are an expert Lean 4 proof assistant. Given a theorem and the current \
proof goals, propose candidate NEXT tactics (a single tactic step each, applied to the first goal).
Rules:
- Output up to {k} tactics, ONE PER LINE, most promising first.
- No explanations, no numbering, no code fences, no `by`.
- Never use sorry, admit, or native_decide.
- Prefer simple standard tactics; use `exact`/`apply` with explicit terms when you see the proof."""

SAMPLER_USER = """Theorem:
{statement}

Current goals:
{goals}
{failed_block}
Tactics ({k} max, one per line):"""


class LLMSampler(ActionProvider):
    def __init__(self, client, k: int = 8, temperature: float = 0.7, max_tokens: int = 400):
        self.client = client
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        out: List[List[Candidate]] = []
        for req in reqs:
            goals_txt = "\n\n".join(req.state.goals) if req.state.goals else "(no goals)"
            failed_block = ""
            if req.failed:
                failed_lines = "\n".join(f"- {t}" for t in req.failed[-12:])
                failed_block = f"\nThese tactics already FAILED on this state, do not repeat them:\n{failed_lines}\n"
            user = SAMPLER_USER.format(statement=req.theorem.statement, goals=goals_txt,
                                       failed_block=failed_block, k=self.k)
            try:
                text = self.client.chat(SAMPLER_SYSTEM.format(k=self.k), user,
                                        temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception as e:  # endpoint down mid-run: degrade, don't die
                print(f"[dxlean] LLM sampler error ({type(e).__name__}): {e}")
                out.append([])
                continue
            out.append([Candidate(t, "llm") for t in parse_tactic_lines(text, self.k)])
        return out


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
