from dxlean.providers import BackboneProvider, Candidate, ProposalRequest, UnionProvider
from dxlean.states import LeanState


class _Fixed:
    def __init__(self, tactics, provenance):
        self.tactics, self.provenance = tactics, provenance

    def propose(self, reqs):
        return [[Candidate(t, self.provenance) for t in self.tactics] for _ in reqs]


def _req():
    return ProposalRequest(LeanState("t", ("⊢ True",), ()))


def test_union_dedupes_and_caps():
    u = UnionProvider([_Fixed(["simp", "omega"], "a"), _Fixed(["omega", "rfl", "decide"], "b")], cap=3)
    (cands,) = u.propose([_req()])
    assert [c.tactic for c in cands] == ["simp", "omega", "rfl"]
    assert cands[1].provenance == "a"  # first provider wins the duplicate


def test_backbone_menu():
    (cands,) = BackboneProvider(["rfl", "simp"]).propose([_req()])
    assert [c.tactic for c in cands] == ["rfl", "simp"]
    assert len(BackboneProvider().propose([_req()])[0]) > 2  # built-in menu
