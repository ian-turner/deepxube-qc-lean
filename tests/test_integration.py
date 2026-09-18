"""Integration tests: require vendor/repl built and lean/testproj present
(scripts/setup_repl.sh). Skipped automatically when the REPL binary is absent.
"""
import os
from typing import Dict, List

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPL_BIN = os.path.join(ROOT, "vendor", "repl", ".lake", "build", "bin", "repl")
PROJECT = os.path.join(ROOT, "lean", "testproj")

pytestmark = pytest.mark.skipif(not os.path.exists(REPL_BIN), reason="REPL not built")

from dxlean.domain import LeanDomain                                         # noqa: E402
from dxlean.providers import ActionProvider, BackboneProvider, Candidate     # noqa: E402
from dxlean.repl import REPLManager                                          # noqa: E402
from dxlean.solve import solve                                               # noqa: E402
from dxlean.states import TheoremSpec                                        # noqa: E402
from dxlean.values import GoalCountValue                                     # noqa: E402


class ScriptedProvider(ActionProvider):
    """Stands in for a tactic model: answers by a substring of the first goal.
    `retry` is returned instead once any tactic has failed on the state."""

    def __init__(self, table: Dict[str, List[str]], retry: List[str] = ()):
        self.table, self.retry, self.n_calls = table, list(retry), 0

    def propose(self, reqs):
        out = []
        for r in reqs:
            self.n_calls += 1
            if r.failed and self.retry:
                tactics = self.retry
            else:
                tactics = next((v for k, v in self.table.items() if k in r.state.goals[0]), ["rfl"])
            out.append([Candidate(t, "scripted") for t in tactics])
        return out


@pytest.fixture(scope="module")
def repl():
    mgr = REPLManager(PROJECT, REPL_BIN, header="import TestProj")
    mgr.start()
    yield mgr
    mgr.stop()


def test_apply_tactic_solves_and_errors(repl):
    thm = TheoremSpec("tst_add_zero", "theorem tst_add_zero (n : Nat) : n + 0 = n")
    root = repl.init_theorem(thm)
    assert len(root.goals) == 1 and "n + 0 = n" in root.goals[0]

    res = repl.apply_tactic(root, "simp")
    assert res.status == "solved" and res.state is not None and res.state.solved

    bad = repl.apply_tactic(root, "nonsense_tac")
    assert bad.status == "error"

    banned = repl.apply_tactic(root, "sorry")
    assert banned.status == "error"

    ok, code = repl.check_full_proof(thm, ("simp",))
    assert ok and code == "theorem tst_add_zero (n : Nat) : n + 0 = n := by\n  simp\n"
    ok, _ = repl.check_full_proof(thm, ("nonsense_tac",))
    assert not ok


def test_timeout_restarts_and_replays(repl):
    """A timed-out tactic kills the REPL; later states are rebuilt by replaying
    their tactic prefix, so the search continues on the same states."""
    thm = TheoremSpec("tst_timeout", "theorem tst_timeout (a b : Prop) (h : a ∧ b) : b ∧ a")
    root = repl.init_theorem(thm)
    mid = repl.apply_tactic(root, "constructor").state
    assert mid is not None and len(mid.goals) == 2

    repl.tactic_timeout, saved = 0.05, repl.tactic_timeout
    try:
        res = repl.apply_tactic(mid, "sleep 300")  # core tactic: sleeps 300ms, goals unchanged
    finally:
        repl.tactic_timeout = saved
    assert res.status == "timeout"
    n_restarts = repl.n_restarts
    assert n_restarts >= 1

    # `mid` lost its proofState id with the process; it is replayed transparently
    res = repl.apply_tactic(mid, "exact h.2")
    assert res.status == "ok" and res.state is not None and len(res.state.goals) == 1
    assert repl.n_restarts == n_restarts


def test_expand_handles_empty_action_lists(repl):
    """A solved state (and a state where every tactic fails) must flow through
    ActsEnum.expand without error — structural requirement for proof search."""
    thm = TheoremSpec("tst_triv", "theorem tst_triv : 1 + 1 = 2")
    root = repl.init_theorem(thm)
    solved = repl.apply_tactic(root, "rfl").state
    assert solved is not None and solved.solved

    domain = LeanDomain(repl, BackboneProvider(["rfl"]))
    children, actions, tcs = domain.expand([solved])
    assert children == [[]] and actions == [[]] and tcs == [[]]


def test_solve_backbone_only(repl):
    theorems = [
        TheoremSpec("tst_two", "theorem tst_two : 2 + 2 = 4"),
        TheoremSpec("tst_imp", "theorem tst_imp (p : Prop) : p → p"),
    ]
    domain = LeanDomain(repl, BackboneProvider())
    results = solve(theorems, repl, domain, GoalCountValue(), itr_max=25)
    assert all(r.solved and r.verified for r in results)
    two = next(r for r in results if r.name == "tst_two")
    assert len(two.tactics) == 1 and two.proof.startswith(theorems[0].statement)


def test_solve_multistep_with_scripted_model(repl):
    """Multi-step proof driven by model-proposed tactics that the backbone menu
    cannot make. Exercises the full propose -> validate -> search -> certify loop."""
    thm = TheoremSpec("tst_trans", "theorem tst_trans (p q r s : Prop) "
                      "(h1 : p → q) (h2 : q → r) (h3 : r → s) : p → s")
    provider = ScriptedProvider({
        "⊢ p → s": ["intro hp", "constructor"],
        "⊢ s": ["apply h3", "apply h1"],
        "⊢ r": ["apply h2", "rfl"],
        "⊢ q": ["apply h1", "assumption"],
        "⊢ p": ["exact hp", "omega"],
    })
    domain = LeanDomain(repl, provider)
    (result,) = solve([thm], repl, domain, GoalCountValue(), itr_max=30)

    assert result.solved and result.verified
    assert result.tactics == ["intro hp", "apply h3", "apply h2", "apply h1", "exact hp"]
    assert provider.n_calls >= 5


def test_resample_recovers_from_bad_round(repl):
    """One bad sample round must not permanently dead-end a state: the domain
    re-proposes with the failed tactics fed back to the provider."""
    thm = TheoremSpec("tst_resample", "theorem tst_resample (a b : Prop) (h : a ∧ b) : b ∧ a")
    bad, good = ["exact h.1", "exact h.2"], ["exact ⟨h.2, h.1⟩"]

    domain = LeanDomain(repl, ScriptedProvider({"⊢ b ∧ a": bad}, retry=good))
    root = repl.init_theorem(thm)
    (acts,) = domain.get_state_actions([root])
    assert [a.tactic for a in acts] == good
    assert domain.stats["resamples"] == 1
    assert domain.failed_tactics(root) == set(bad)

    # with resampling disabled, the same bad round is a permanent dead end
    domain0 = LeanDomain(repl, ScriptedProvider({"⊢ b ∧ a": bad}, retry=good), max_resamples=0)
    (acts0,) = domain0.get_state_actions([repl.init_theorem(thm)])
    assert acts0 == [] and domain0.stats["resamples"] == 0


def test_self_reference_rejected(repl):
    """`theorem X := by sorry` puts a sorry-backed `X` in the search env; using
    it is a circular proof that certification rejects — so the tactic gate must
    refuse it up front (found live: a model proposed `apply and_swap`)."""
    thm = TheoremSpec("tst_selfref", "theorem tst_selfref (a b : Prop) (h : a ∧ b) : b ∧ a")
    root = repl.init_theorem(thm)

    res = repl.apply_tactic(root, "apply tst_selfref")
    assert res.status == "error" and "self-reference" in res.message
    assert repl.apply_tactic(root, "exact tst_selfref a b h").status == "error"
    # names that merely contain the theorem name are fine
    assert repl.apply_tactic(root, "exact tst_selfref' h").status == "error"  # unknown ident, but NOT self-ref
    assert "self-reference" not in repl.apply_tactic(root, "exact tst_selfref' h").message
    # and normal progress still works
    assert repl.apply_tactic(root, "constructor").status == "ok"
