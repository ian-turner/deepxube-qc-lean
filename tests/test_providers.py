from dxlean.providers import (BackboneProvider, Candidate, ProposalRequest,
                              UnionProvider, parse_tactic_lines)
from dxlean.states import LeanState, TheoremSpec


def test_parse_tactic_lines_strips_and_filters():
    text = """1. simp
- `omega`
```
constructor
```
-- a comment
exact sorry
apply Nat.le_of_lt
simp
"""
    got = parse_tactic_lines(text, 8)
    assert got == ["simp", "omega", "constructor", "apply Nat.le_of_lt"]


def test_parse_tactic_lines_caps():
    text = "\n".join(f"tac{i}" for i in range(20))
    assert len(parse_tactic_lines(text, 5)) == 5


class _Fixed:
    def __init__(self, tactics, provenance):
        self.tactics, self.provenance = tactics, provenance

    def propose(self, reqs):
        return [[Candidate(t, self.provenance) for t in self.tactics] for _ in reqs]


def _req():
    thm = TheoremSpec("t", "theorem t : True")
    return ProposalRequest(LeanState("t", ("⊢ True",), ()), thm)


def test_union_dedupes_and_caps():
    u = UnionProvider([_Fixed(["simp", "omega"], "a"), _Fixed(["omega", "rfl", "decide"], "b")], cap=3)
    (cands,) = u.propose([_req()])
    assert [c.tactic for c in cands] == ["simp", "omega", "rfl"]
    assert cands[1].provenance == "a"  # first provider wins the duplicate


def test_backbone_skips_nothing_by_default():
    (cands,) = BackboneProvider(["rfl", "simp"]).propose([_req()])
    assert [c.tactic for c in cands] == ["rfl", "simp"]
