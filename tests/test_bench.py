"""Benchmark tooling: the miniF2F exporter and the nanoproof/dxlean comparison."""
import importlib.util
import json
import os
from dataclasses import asdict

from dxlean.solve import SolveResult

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLE = """import MiniF2F.ProblemImports

open scoped Real

/-- doc with the word theorem in it -/
theorem foo (x : ℕ) (h₀ : x = 1) :
    x + 1 = answer(2) := by
  sorry

theorem bar : 1 = 1 :=
  by
  sorry
"""


def test_export_parse_matches_nanoproof_ids_and_strips_tail():
    rows = _script("export_minif2f").parse(SAMPLE)
    assert rows == [("foo", "theorem foo (x : ℕ) (h₀ : x = 1) :\n    x + 1 = answer(2)"),
                    ("bar", "theorem bar : 1 = 1")]


def test_compare_joins_by_name_and_curves_by_model_calls(tmp_path):
    cmp = _script("compare")
    np_dir, dx_dir = tmp_path / "eval_np", tmp_path / "dx_w1"
    np_dir.mkdir(), dx_dir.mkdir()
    (np_dir / "theorems.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "proof": {"t": 1}, "num_iterations": 5, "error": None},
        {"id": "b", "proof": None, "num_iterations": 512, "error": None},
        {"id": "c", "proof": {"t": 1}, "num_iterations": 3, "error": "verification failed"},
    ]) + "\n")
    (np_dir / "summary.toml").write_text("num_simulations = 512\n\n[success_rate_by_simulations]\n8 = 0.5\n")
    # dxlean rows are written from the dataclass (cli.py), so build them the same way
    (dx_dir / "results.jsonl").write_text("\n".join(
        json.dumps({k: v for k, v in asdict(r).items() if k != "proof"}) for r in [
            SolveResult("a", True, verified=True, model_calls=20),
            SolveResult("b", True, verified=True, model_calls=100),
            SolveResult("c", False, model_calls=130),   # ran past the budget, unsolved
            SolveResult("extra", True, verified=True, model_calls=1),
        ]) + "\n")
    np_run, dx_run = cmp.load(str(np_dir)), cmp.load(str(dx_dir / "results.jsonl"))
    assert np_run == ("eval_np", 512, {"a": (True, 5), "b": (False, 512), "c": (False, 3)})
    assert dx_run[:2] == ("dx_w1", 130) and dx_run[2]["c"] == (False, 130)  # no summary: budget = max seen
    names = ["a", "b", "c"]
    assert cmp.curve(np_run, names)[8] == 1 / 3 and cmp.curve(np_run, names)[1024] is None
    assert cmp.curve(dx_run, names)[16] == 0 and cmp.curve(dx_run, names)[128] == 2 / 3
    text = cmp.report([np_run, dx_run])
    assert "3 in common" in text and "b: eval_np=-  dx_w1=100" in text and "   -   " in text
