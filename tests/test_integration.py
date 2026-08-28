"""Integration tests: require vendor/repl built and lean/testproj present
(scripts/setup_repl.sh). Skipped automatically when the REPL binary is absent.
"""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPL_BIN = os.path.join(ROOT, "vendor", "repl", ".lake", "build", "bin", "repl")
PROJECT = os.path.join(ROOT, "lean", "testproj")

pytestmark = pytest.mark.skipif(not os.path.exists(REPL_BIN), reason="REPL not built")

from dxlean.domain import LeanDomain          # noqa: E402
from dxlean.providers import BackboneProvider  # noqa: E402
from dxlean.repl import REPLManager            # noqa: E402
from dxlean.solve import solve                 # noqa: E402
from dxlean.states import TheoremSpec          # noqa: E402
from dxlean.values import GoalCountValue       # noqa: E402


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

    ok, _ = repl.check_full_proof(thm, ("simp",))
    assert ok
    ok, _ = repl.check_full_proof(thm, ("nonsense_tac",))
    assert not ok


def test_expand_handles_empty_action_lists(repl):
    """A solved state (and a state where every tactic fails) must flow through
    ActsEnum.expand without error — structural requirement for proof search."""
    thm = TheoremSpec("tst_triv", "theorem tst_triv : 1 + 1 = 2")
    root = repl.init_theorem(thm)
    solved = repl.apply_tactic(root, "rfl").state
    assert solved is not None and solved.solved

    domain = LeanDomain(repl, BackboneProvider(["rfl"]), {thm.name: thm})
    children, actions, tcs = domain.expand([solved])
    assert children == [[]] and actions == [[]] and tcs == [[]]


def test_solve_backbone_only(repl):
    theorems = [
        TheoremSpec("tst_two", "theorem tst_two : 2 + 2 = 4"),
        TheoremSpec("tst_imp", "theorem tst_imp (p : Prop) : p → p"),
    ]
    domain = LeanDomain(repl, BackboneProvider(), {t.name: t for t in theorems})
    results = solve(theorems, repl, domain, GoalCountValue(), itr_max=25)
    assert all(r.solved and r.verified for r in results)
    two = next(r for r in results if r.name == "tst_two")
    assert len(two.tactics) == 1


def test_solve_multistep_with_fake_llm(repl):
    """Multi-step proof driven by model-proposed tactics: the fake LLM plays a
    tactic model proposing `apply`/`exact` steps the backbone menu cannot make.
    Exercises the full propose -> validate -> search -> certify loop."""
    from dxlean.llm import FakeChatClient
    from dxlean.providers import LLMSampler, UnionProvider

    thm = TheoremSpec("tst_trans", "theorem tst_trans (p q r s : Prop) "
                      "(h1 : p → q) (h2 : q → r) (h3 : r → s) : p → s")

    def respond(system: str, user: str) -> str:
        if "⊢ p → s" in user:
            return "intro hp\nconstructor"
        if "⊢ s" in user:
            return "apply h3\napply h1"
        if "⊢ r" in user:
            return "apply h2\nrfl"
        if "⊢ q" in user:
            return "apply h1\nassumption"
        if "⊢ p" in user:
            return "exact hp\nomega"
        return "rfl"

    client = FakeChatClient(respond)
    provider = UnionProvider([LLMSampler(client, k=4)], cap=8)
    domain = LeanDomain(repl, provider, {thm.name: thm})
    (result,) = solve([thm], repl, domain, GoalCountValue(), itr_max=30)

    assert result.solved and result.verified
    assert result.tactics == ["intro hp", "apply h3", "apply h2", "apply h1", "exact hp"]
    assert client.n_calls >= 5


def test_resample_recovers_from_bad_round(repl):
    """One bad LLM sample round must not permanently dead-end a state: the
    domain re-proposes with the failed tactics fed back to the sampler (found
    live: qwen2.5-coder proposed only junk for and_swap's root and the search
    died at iteration 0, frontier exhausted)."""
    from dxlean.llm import FakeChatClient
    from dxlean.providers import LLMSampler, UnionProvider

    thm = TheoremSpec("tst_resample", "theorem tst_resample (a b : Prop) (h : a ∧ b) : b ∧ a")

    def respond(system: str, user: str) -> str:
        if "already FAILED" in user:
            return "exact ⟨h.2, h.1⟩"
        return "exact h.1\nexact h.2"

    provider = UnionProvider([LLMSampler(FakeChatClient(respond), k=4)], cap=8)
    domain = LeanDomain(repl, provider, {thm.name: thm})
    root = repl.init_theorem(thm)
    (acts,) = domain.get_state_actions([root])
    assert [a.tactic for a in acts] == ["exact ⟨h.2, h.1⟩"]
    assert domain.stats["resamples"] == 1
    assert domain.failed_tactics(root) == {"exact h.1", "exact h.2"}

    # with resampling disabled, the same bad round is a permanent dead end
    provider0 = UnionProvider([LLMSampler(FakeChatClient(respond), k=4)], cap=8)
    domain0 = LeanDomain(repl, provider0, {thm.name: thm}, max_resamples=0)
    (acts0,) = domain0.get_state_actions([repl.init_theorem(thm)])
    assert acts0 == [] and domain0.stats["resamples"] == 0


def test_self_reference_rejected(repl):
    """`theorem X := by sorry` puts a sorry-backed `X` in the search env; using
    it is a circular proof that certification rejects — so the tactic gate must
    refuse it up front (found live: llama3 proposed `apply and_swap`)."""
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
