import os

import pytest
from matplotlib.figure import Figure

from dxlean.states import LeanGoal, LeanState, TheoremSpec
from dxlean.viz import format_goals, goal_oneliner, render_state_goal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPL_BIN = os.path.join(ROOT, "vendor", "repl", ".lake", "build", "bin", "repl")
PROJECT = os.path.join(ROOT, "lean", "testproj")


def _thm():
    return TheoremSpec("and_swap", "theorem and_swap (a b : Prop) (h : a ∧ b) : b ∧ a")


def test_render_state_goal_draws_text():
    state = LeanState("and_swap", ("a b : Prop\nh : a ∧ b\n⊢ b ∧ a",), ())
    fig = Figure(figsize=(6, 4))
    render_state_goal(state, LeanGoal(_thm()), fig)
    texts = [t.get_text() for t in fig.texts]
    assert any("and_swap" in t for t in texts)
    assert any("⊢ b ∧ a" in t for t in texts)

    solved = LeanState("and_swap", (), ("constructor", "exact h.2", "exact h.1"))
    render_state_goal(solved, LeanGoal(_thm()), fig)
    assert any("proof complete" in t.get_text() for t in fig.texts)


def test_format_helpers():
    two = LeanState("t", ("⊢ b", "⊢ a"), ("constructor",))
    assert "goal 1/2" in format_goals(two)
    assert goal_oneliner(two).startswith("⊢ b") and "+1 more" in goal_oneliner(two)
    assert goal_oneliner(LeanState("t", (), ())) == "✓ no goals"
    long = LeanState("t", ("⊢ " + "x" * 200,), ())
    assert len(goal_oneliner(long, width=40)) == 40


def test_string_to_action_mixin():
    from dxlean.domain import LeanDomain
    from dxlean.providers import BackboneProvider

    domain = LeanDomain.__new__(LeanDomain)  # mixin methods need no REPL
    act = LeanDomain.string_to_action(domain, "  exact h.2 ")
    assert act is not None and act.tactic == "exact h.2" and act.provenance == "user"
    assert LeanDomain.string_to_action(domain, "   ") is None
    assert "tactic" in LeanDomain.string_to_action_help(domain)


@pytest.mark.skipif(not os.path.exists(REPL_BIN), reason="REPL not built")
def test_traced_search_narrates_and_prints_tree(capsys, tmp_path):
    from dxlean.domain import LeanDomain
    from dxlean.providers import BackboneProvider
    from dxlean.repl import REPLManager
    from dxlean.values import GoalCountValue
    from dxlean.viz import traced_search

    repl = REPLManager(PROJECT, REPL_BIN, header="import TestProj")
    repl.start()
    try:
        thm = TheoremSpec("tst_imp_viz", "theorem tst_imp_viz (p : Prop) : p → p")
        domain = LeanDomain(repl, BackboneProvider(["intro h", "assumption", "constructor"]),
                            {thm.name: thm})
        fig_path = str(tmp_path / "tree.png")
        solved, tactics = traced_search(thm, repl, domain, GoalCountValue(), itr_max=20,
                                        fig_path=fig_path)
    finally:
        repl.stop()

    assert solved and tactics == ["intro h", "assumption"]
    out = capsys.readouterr().out
    assert "itr 0: pop" in out
    assert "search tree" in out
    assert "★" in out          # solution path marked
    assert "SOLVED" in out and "certification: PASSED" in out
    assert os.path.getsize(fig_path) > 5000  # tree figure rendered
