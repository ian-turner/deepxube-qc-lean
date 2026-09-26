"""nanoproof integration: the pre-trained policy+value checkpoint as dxlean plugins.

nanoproof (github.com/Kripner/nanoproof) serves its model over a small Flask
app (`nanoproof.inference.start_inference_server`):

    POST {base_url}/generate   {"states": [state_str, ...]}
    200  {"results": [{"tactics": [...], "logprobs": [...], "value": float}
                      | {"error": str}, ...]}
    503  {"busy": true}        (batching queue saturated; retry later)

One request yields BOTH tactic samples (with summed logprobs) and the value
for every state, so a single `NanoproofOracle` backs two dxlean plugins:

- `NanoproofProvider` (ActionProvider): the sampled tactics as candidates,
  logprob carried as `Candidate.score`. Re-proposal after failures re-queries
  the server (it draws a fresh seed per call) and merges the new samples.
- `NanoproofValue` (ValueProvider): the value head predicts *remaining proof
  depth* as an expected bin index in 1..64 — a cost-to-go in tactic steps,
  i.e. deepxube's h with unit edge costs, used unscaled.

deepxube evaluates h on children as soon as they are generated and only
later expands the ones it pops; the oracle caches by prompt string, so the
value query already stocks the tactic cache and expansion costs no second
GPU call.

Prompt format: nanoproof was trained on leantree's rendering of a (usually
single-goal) factorized proof state — one `name : type` hypothesis per line,
then `⊢ target`, goals separated by a blank line, optional `case tag` line.
The community REPL's pretty-printed goals are the same shape except that Lean
groups same-typed hypotheses (`a b : ℕ`); `leantree_goal` splits those.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import httpx

from .providers import ActionProvider, Candidate, ProposalRequest
from .repl import BANNED_SUBSTRINGS
from .states import LeanGoal, LeanState, Ledger
from .values import ValueProvider

GOAL_MODES = ("first", "all")          # what the policy sees
VALUE_MODES = ("sum", "first", "all")  # how h is assembled from per-goal values


@dataclass
class NanoResult:
    tactics: List[str] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    value: Optional[float] = None  # None: server reported an error for this state
    error: str = ""


# -- goal text normalization --------------------------------------------------

# a bare hypothesis name as Lean prints it: no whitespace, no brackets/binders
_NAME = re.compile(r"^[^\s()\[\]{}⟨⟩,:]+$")


def leantree_goal(goal: str) -> str:
    """Render one REPL goal the way leantree (nanoproof's training data) does:
    split grouped hypotheses `a b : T` into one line per name. `case` lines,
    the `⊢` line, and indented continuation lines pass through unchanged."""
    out: List[str] = []
    for line in goal.strip().splitlines():
        if not line or line[0].isspace() or line.startswith("case ") or line.startswith("⊢"):
            out.append(line)
            continue
        head, sep, rest = line.partition(" : ")
        names = head.split()
        if sep and len(names) > 1 and all(_NAME.match(n) for n in names):
            out.extend(f"{n} : {rest}" for n in names)
        else:
            out.append(line)
    return "\n".join(out).strip()


def state_prompt(state: LeanState, goal_mode: str = "first") -> str:
    """The state string sent to nanoproof (server appends the task token)."""
    if goal_mode == "first":
        return leantree_goal(state.goals[0]) if state.goals else ""
    return "\n\n".join(leantree_goal(g) for g in state.goals)


# -- HTTP client ------------------------------------------------------------------

class NanoproofClient:
    def __init__(self, base_url: str, timeout: float = 600.0, chunk: int = 32,
                 busy_wait: float = 0.5, busy_max_wait: float = 900.0,
                 transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.chunk = max(1, chunk)
        self.busy_wait = busy_wait
        self.busy_max_wait = busy_max_wait
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self.n_requests = 0
        self.n_busy = 0

    def health(self) -> dict:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def generate(self, states: Sequence[str]) -> List[NanoResult]:
        out: List[NanoResult] = []
        for i in range(0, len(states), self.chunk):
            out.extend(self._generate_chunk(list(states[i:i + self.chunk])))
        return out

    def _generate_chunk(self, states: List[str]) -> List[NanoResult]:
        if not states:
            return []
        waited = 0.0
        while True:
            self.n_requests += 1
            resp = self._client.post(f"{self.base_url}/generate", json={"states": states})
            if resp.status_code == 503:
                self.n_busy += 1
                if waited >= self.busy_max_wait:
                    raise RuntimeError(f"nanoproof server busy for {waited:.0f}s")
                time.sleep(self.busy_wait)
                waited += self.busy_wait
                continue
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if len(results) != len(states):
                raise RuntimeError(f"nanoproof returned {len(results)} results for {len(states)} states")
            return [_parse_result(r) for r in results]


def _parse_result(r: dict) -> NanoResult:
    if "error" in r:
        return NanoResult(error=str(r["error"]))
    tactics, logprobs = [], []
    seen = set()
    for tac, lp in zip(r.get("tactics", []), r.get("logprobs", [])):
        tac = str(tac).strip()
        if not tac or tac in seen or any(b in tac.lower() for b in BANNED_SUBSTRINGS):
            continue
        seen.add(tac)
        tactics.append(tac)
        logprobs.append(float(lp))
    value = r.get("value")
    return NanoResult(tactics, logprobs, None if value is None else float(value))


# -- shared oracle ------------------------------------------------------------

class NanoproofOracle:
    """Prompt-string -> NanoResult cache in front of the client. `query` fetches
    the misses in one batched request; `refresh` re-samples tactics for prompts
    already cached (new server seed) and merges them, keeping the value."""

    def __init__(self, client: NanoproofClient, goal_mode: str = "first",
                 value_mode: str = "sum", ledger: Optional[Ledger] = None):
        assert goal_mode in GOAL_MODES and value_mode in VALUE_MODES
        self.client = client
        self.goal_mode = goal_mode
        self.value_mode = value_mode
        self.ledger: Ledger = ledger if ledger is not None else defaultdict(Counter)
        self._cache: Dict[str, NanoResult] = {}
        self._charged: Set[Tuple[str, str]] = set()  # (theorem, prompt) pairs already paid for
        self.stats: Dict[str, int] = {"states_sent": 0, "cache_hits": 0, "refreshes": 0, "errors": 0}

    def query(self, prompts: Sequence[str], thms: Sequence[str]) -> List[NanoResult]:
        """Cache-first fetch. Each theorem pays one model call the first time it
        asks for a prompt (`thms[i]` owns `prompts[i]`), so its cost is what it
        would have paid alone: a transposition inside the theorem is free, a
        goal some other theorem already cached is not."""
        new = [tp for tp in dict.fromkeys(zip(thms, prompts)) if tp not in self._charged]
        self._charged.update(new)
        for thm, _ in new:
            self.ledger[thm]["model_calls"] += 1
        missing = list(dict.fromkeys(p for p in prompts if p not in self._cache))
        self.stats["cache_hits"] += sum(p in self._cache for p in prompts)
        if missing:
            self.stats["states_sent"] += len(missing)
            for p, res in zip(missing, self._call(missing)):
                self._cache[p] = res
        return [self._cache[p] for p in prompts]

    def refresh(self, prompts: Sequence[str], thms: Sequence[str]) -> List[NanoResult]:
        owner = dict(zip(prompts, thms))
        prompts = list(dict.fromkeys(prompts))
        if not prompts:
            return []
        self.stats["refreshes"] += len(prompts)
        self.stats["states_sent"] += len(prompts)
        for p, fresh in zip(prompts, self._call(prompts)):
            self.ledger[owner[p]]["model_calls"] += 1
            old = self._cache.get(p)
            if old is None or old.error:
                self._cache[p] = fresh
                continue
            for tac, lp in zip(fresh.tactics, fresh.logprobs):
                if tac not in old.tactics:
                    old.tactics.append(tac)
                    old.logprobs.append(lp)
        return [self._cache[p] for p in prompts]

    def _call(self, prompts: List[str]) -> List[NanoResult]:
        try:
            results = self.client.generate(prompts)
        except Exception as e:  # server down mid-run: degrade, don't die
            print(f"[dxlean] nanoproof error ({type(e).__name__}): {e}")
            results = [NanoResult(error=str(e)) for _ in prompts]
        self.stats["errors"] += sum(1 for r in results if r.error)
        return results

    # prompts for the two plugins
    def tactic_prompt(self, state: LeanState) -> str:
        return state_prompt(state, self.goal_mode)

    def value_prompts(self, state: LeanState) -> List[str]:
        if self.value_mode == "sum":
            return [leantree_goal(g) for g in state.goals]
        if self.value_mode == "first":
            return [state_prompt(state, "first")]
        return [state_prompt(state, "all")]


# -- plugins -----------------------------------------------------------------------

class NanoproofProvider(ActionProvider):
    """Sampled tactics, best logprob first. A request whose known-failed set
    already covers every cached sample (or whose sample came back empty) is
    re-sampled with a fresh server seed and merged."""

    def __init__(self, oracle: NanoproofOracle):
        self.oracle = oracle

    def propose(self, reqs: List[ProposalRequest]) -> List[List[Candidate]]:
        prompts = [self.oracle.tactic_prompt(r.state) for r in reqs]
        thms = [r.state.thm_name for r in reqs]
        results = self.oracle.query(prompts, thms)
        stale = [(p, t) for p, t, r, res in zip(prompts, thms, reqs, results)
                 if not res.error and all(x in r.failed for x in res.tactics)]
        if stale:
            self.oracle.refresh([p for p, _ in stale], [t for _, t in stale])
            results = self.oracle.query(prompts, thms)
        out: List[List[Candidate]] = []
        for res in results:
            ranked = sorted(zip(res.tactics, res.logprobs), key=lambda tl: -tl[1])
            out.append([Candidate(t, "nanoproof", lp) for t, lp in ranked])
        return out


class NanoproofValue(ValueProvider):
    """h = predicted remaining proof depth (1..64 per goal). `value_mode`
    'sum' adds per-goal predictions (each goal needs its own proof), 'first'
    scores only the goal the policy sees, 'all' scores the joined goals."""

    def __init__(self, oracle: NanoproofOracle, default: float = 16.0):
        self.oracle = oracle
        self.default = default

    def estimate(self, states: List[LeanState], goals: List[LeanGoal]) -> List[float]:
        per_state = [[] if s.solved else self.oracle.value_prompts(s) for s in states]
        results = iter(self.oracle.query([p for ps in per_state for p in ps],
                                         [s.thm_name for s, ps in zip(states, per_state) for _ in ps]))
        return [sum(self.default if r.value is None else r.value for r in (next(results) for _ in ps))
                for ps in per_state]  # solved states have no prompts -> 0.0
