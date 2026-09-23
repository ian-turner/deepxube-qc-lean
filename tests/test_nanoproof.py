"""nanoproof integration against a fake /generate server (httpx.MockTransport)."""
import json

import httpx

from dxlean.nanoproof import (NanoproofClient, NanoproofOracle, NanoproofProvider,
                              NanoproofValue, leantree_goal, state_prompt)
from dxlean.providers import ProposalRequest
from dxlean.states import LeanGoal, LeanState, TheoremSpec

THM = TheoremSpec("t", "theorem t (p q : Prop) (h : p ∧ q) : q ∧ p")
# already in leantree form (one hypothesis per line): the prompt equals the goal text
GOAL_A = "p : Prop\nq : Prop\nh : p ∧ q\n⊢ q ∧ p"
GOAL_B = "p : Prop\nq : Prop\nh : p ∧ q\n⊢ q"


class FakeServer:
    """Answers by state string; counts calls; optional 503s and per-state errors."""

    def __init__(self, table, busy_first=0):
        self.table = table
        self.busy_left = busy_first
        self.calls = 0
        self.seen = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/generate"
        if self.busy_left > 0:
            self.busy_left -= 1
            return httpx.Response(503, json={"busy": True})
        self.calls += 1
        states = json.loads(request.content)["states"]
        self.seen.append(states)
        results = []
        for s in states:
            entry = self.table.get(s)
            if entry is None:
                results.append({"error": f"unknown state {s!r}"})
            else:
                # each call hands out the next sample set (stochastic server)
                tactics, lps, value = entry.pop(0) if len(entry) > 1 else entry[0]
                results.append({"tactics": tactics, "logprobs": lps, "value": value})
        return httpx.Response(200, json={"results": results})


def make(table, goal_mode="first", value_mode="sum", busy_first=0):
    srv = FakeServer(table, busy_first)
    client = NanoproofClient("http://fake", transport=httpx.MockTransport(srv.handler),
                             busy_wait=0.0)
    return srv, NanoproofOracle(client, goal_mode=goal_mode, value_mode=value_mode)


def test_leantree_goal_splits_grouped_hypotheses():
    goal = "case inl\na b : ℕ\nh : a ≤ b\nhf : ∀ x : ℕ,\n    x = x\n⊢ a + 0 ≤ b"
    assert leantree_goal(goal) == (
        "case inl\na : ℕ\nb : ℕ\nh : a ≤ b\nhf : ∀ x : ℕ,\n    x = x\n⊢ a + 0 ≤ b")
    # binder groups are not hypothesis lists
    assert leantree_goal("(h : p) : q") == "(h : p) : q"


def test_state_prompt_modes():
    raw_a = "p q : Prop\nh : p ∧ q\n⊢ q ∧ p"  # as the community REPL prints it
    s = LeanState("t", (raw_a, GOAL_B), ())
    assert state_prompt(s, "first") == GOAL_A
    assert state_prompt(s, "all") == GOAL_A + "\n\n" + GOAL_B


def test_value_then_propose_is_one_call_and_carries_logprobs():
    srv, oracle = make({GOAL_A: [(["constructor", "exact ⟨h.2, h.1⟩", "sorry"], [-1.5, -0.2, -0.1], 2.4)]})
    s = LeanState("t", (GOAL_A,), ())
    (h,) = NanoproofValue(oracle).estimate([s], [LeanGoal(THM)])
    assert h == 2.4
    (cands,) = NanoproofProvider(oracle).propose([ProposalRequest(s)])
    assert [c.tactic for c in cands] == ["exact ⟨h.2, h.1⟩", "constructor"]  # banned dropped, best first
    assert cands[0].provenance == "nanoproof" and cands[0].score == -0.2
    assert srv.calls == 1  # value query stocked the tactic cache
    assert oracle.ledger["t"]["model_calls"] == 1  # ...and the theorem was charged once


def test_solved_state_is_zero_without_a_call():
    srv, oracle = make({})
    (h,) = NanoproofValue(oracle).estimate([LeanState("t", (), ("simp",))], [LeanGoal(THM)])
    assert h == 0.0 and srv.calls == 0


def test_value_sum_over_goals_and_first_mode():
    table = {GOAL_A: [(["constructor"], [-1.0], 3.0)], GOAL_B: [(["exact h.2"], [-0.5], 1.0)]}
    s = LeanState("t", (GOAL_A, GOAL_B), ())
    srv, oracle = make(dict(table))
    assert NanoproofValue(oracle).estimate([s], [LeanGoal(THM)]) == [4.0]
    assert srv.calls == 1 and sorted(srv.seen[0]) == sorted([GOAL_A, GOAL_B])
    srv, oracle = make(dict(table), value_mode="first")
    assert NanoproofValue(oracle).estimate([s], [LeanGoal(THM)]) == [3.0]


def test_resample_refreshes_and_merges():
    table = {GOAL_A: [(["constructor"], [-1.0], 3.0), (["exact ⟨h.2, h.1⟩", "constructor"], [-0.3, -1.0], 3.0)]}
    srv, oracle = make(table)
    prov = NanoproofProvider(oracle)
    s = LeanState("t", (GOAL_A,), ())
    (c1,) = prov.propose([ProposalRequest(s)])
    assert [c.tactic for c in c1] == ["constructor"] and srv.calls == 1
    # failed set covers everything known -> re-sample, merge, keep value
    (c2,) = prov.propose([ProposalRequest(s, failed=("constructor",))])
    assert [c.tactic for c in c2] == ["exact ⟨h.2, h.1⟩", "constructor"] and srv.calls == 2
    assert NanoproofValue(oracle).estimate([s], [LeanGoal(THM)]) == [3.0] and srv.calls == 2
    assert oracle.stats["refreshes"] == 1
    assert oracle.ledger["t"]["model_calls"] == 2  # the re-sample was a second model call


def test_each_theorem_pays_for_a_shared_goal_once():
    """A goal another theorem already cached is one server call but two charges:
    per-theorem cost must not depend on what else ran (nanoproof pays per node)."""
    srv, oracle = make({GOAL_A: [(["constructor"], [-1.0], 3.0)]})
    s1, s2 = LeanState("t1", (GOAL_A,), ()), LeanState("t2", (GOAL_A,), ())
    NanoproofValue(oracle).estimate([s1, s2, s1], [LeanGoal(THM)] * 3)
    NanoproofProvider(oracle).propose([ProposalRequest(s2)])
    assert srv.calls == 1 and oracle.ledger["t1"]["model_calls"] == 1 and oracle.ledger["t2"]["model_calls"] == 1


def test_empty_sample_is_resampled():
    """A sample set that came back empty (all banned/duplicates) is re-sampled
    right away instead of dead-ending the state."""
    table = {GOAL_A: [(["sorry"], [-0.1], 3.0), (["constructor"], [-1.0], 3.0)]}
    srv, oracle = make(table)
    (cands,) = NanoproofProvider(oracle).propose([ProposalRequest(LeanState("t", (GOAL_A,), ()))])
    assert [c.tactic for c in cands] == ["constructor"] and srv.calls == 2


def test_busy_retry_and_error_fallback():
    srv, oracle = make({GOAL_A: [(["rfl"], [-0.1], 1.0)]}, busy_first=2)
    s_ok, s_bad = LeanState("t", (GOAL_A,), ()), LeanState("t", ("⊢ False",), ())
    vals = NanoproofValue(oracle, default=9.0).estimate([s_ok, s_bad], [LeanGoal(THM)] * 2)
    assert vals == [1.0, 9.0] and srv.calls == 1 and oracle.client.n_busy == 2
    (ok, bad) = NanoproofProvider(oracle).propose([ProposalRequest(s_ok), ProposalRequest(s_bad)])
    assert [c.tactic for c in ok] == ["rfl"] and bad == [] and srv.calls == 1
    assert oracle.stats["errors"] == 1


def test_chunking_splits_requests():
    table = {f"⊢ g{i}": [([f"t{i}"], [-1.0], float(i))] for i in range(5)}
    srv = FakeServer(table)
    client = NanoproofClient("http://fake", chunk=2, transport=httpx.MockTransport(srv.handler))
    res = client.generate(list(table))
    assert [r.value for r in res] == [0.0, 1.0, 2.0, 3.0, 4.0] and srv.calls == 3
