/-! Minimal library for the dxlean dev corpus. No dependencies (core Lean only),
so the REPL boots in about a second. Problem statements import this module. -/

namespace TestProj

/-- A trivially provable predicate used by a couple of dev problems. -/
def double (n : Nat) : Nat := n + n

theorem double_eq_two_mul (n : Nat) : double n = 2 * n := by
  simp [double, Nat.two_mul]

end TestProj
