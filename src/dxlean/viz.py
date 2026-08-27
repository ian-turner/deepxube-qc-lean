"""Visualization of the environment and the search process.

Three views, all built on the REPL's pretty-printed goal text:

- `render_state_goal`: draw one (state, goal) pair on a matplotlib Figure —
  the deepxube `StateGoalVizable` hook.
- `interactive`: a proof shell for the environment itself — type tactics, see
  the resulting proof state, ask the action providers for proposals, undo.
- `traced_search`: run the real search on one theorem, narrating every
  iteration (what was popped, its f = W·g + h, which candidates validated),
  then print the search tree (deepxube keeps it in `Node.edge_dict`) with the
  solution path marked.
"""
from __future__ import annotations

import textwrap
from typing import List, Optional, Set, Tuple

from deepxube.base.pathfind_fns import PFNsHeurV
from deepxube.base.pathfinding import Node, get_path
from deepxube.pathfinding.graph_search import GraphSearchHeurNodeActsEnum
from matplotlib.figure import Figure

from .domain import LeanDomain
from .providers import ActionProvider, ProposalRequest
from .repl import REPLManager
from .states import LeanGoal, LeanState, TheoremSpec
from .values import ValueProvider, as_heurv


# -- state/goal figure (StateGoalVizable) ------------------------------------

def render_state_goal(state: LeanState, goal: LeanGoal, fig: Figure) -> None:
    """Theorem on top, current proof state below, monospace throughout."""
    fig.clf()
    fig.set_facecolor("white")
    stmt = "\n".join(textwrap.wrap(goal.theorem.statement, width=88)) or "(no statement)"
    if state.solved:
        goals_txt = "No goals — proof complete."
    else:
        goals_txt = f"\n\n{'─' * 40}\n\n".join(state.goals)
    via = " ; ".join(state.tactics) if state.tactics else "(root)"

    fig.text(0.02, 0.98, f"{goal.theorem.name}   [{len(state.goals)} goal(s)]",
             va="top", ha="left", fontsize=11, fontweight="bold", family="monospace")
    fig.text(0.02, 0.92, stmt, va="top", ha="left", fontsize=9, family="monospace", color="#333333")
    fig.text(0.02, 0.80, f"via: {via}", va="top", ha="left", fontsize=8,
             family="monospace", color="#777777")
    fig.text(0.02, 0.74, goals_txt, va="top", ha="left", fontsize=10, family="monospace",
             color="#006400" if state.solved else "black")


# -- terminal helpers --------------------------------------------------------

def format_goals(state: LeanState, indent: str = "  ") -> str:
    if state.solved:
        return f"{indent}✓ no goals — proof complete"
    parts = []
    for i, g in enumerate(state.goals):
        head = f"{indent}goal {i + 1}/{len(state.goals)}:" if len(state.goals) > 1 else f"{indent}goal:"
        body = "\n".join(f"{indent}  {line}" for line in g.splitlines())
        parts.append(f"{head}\n{body}")
    return "\n".join(parts)


def goal_oneliner(state: LeanState, width: int = 64) -> str:
    if state.solved:
        return "✓ no goals"
    turnstile = state.goals[0].split("⊢")[-1].strip().replace("\n", " ")
    extra = f" (+{len(state.goals) - 1} more)" if len(state.goals) > 1 else ""
    text = f"⊢ {turnstile}{extra}"
    return text if len(text) <= width else text[: width - 1] + "…"


# -- interactive proof shell -------------------------------------------------

INTERACTIVE_HELP = """commands:
  <tactic>   apply a Lean tactic to the current state (e.g. simp, intro h, exact h.2)
  :p         ask the action providers for candidates and validate each against the REPL
  :u         undo the last applied tactic
  :g         reprint the current proof state
  :q         quit"""


def interactive(repl: REPLManager, provider: ActionProvider, theorem: TheoremSpec,
                value_provider: Optional[ValueProvider] = None) -> None:
    stack: List[LeanState] = [repl.init_theorem(theorem)]
    goal = LeanGoal(theorem)
    print(f"\n=== {theorem.name} ===\n{theorem.statement}\n\n{INTERACTIVE_HELP}\n")

    def show(state: LeanState) -> None:
        print(format_goals(state))
        if value_provider is not None and not state.solved:
            (h,) = value_provider.estimate([state], [goal])
            print(f"  [h = {h:.1f}]")

    show(stack[-1])
    while True:
        try:
            line = input(f"\n[{len(stack) - 1} tactics] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        state = stack[-1]
        if not line:
            continue
        if line == ":q":
            return
        if line == ":g":
            show(state)
        elif line == ":u":
            if len(stack) > 1:
                stack.pop()
                print(f"  undid {state.tactics[-1]!r}")
                show(stack[-1])
            else:
                print("  at root, nothing to undo")
        elif line == ":p":
            if state.solved:
                print("  already solved")
                continue
            (cands,) = provider.propose([ProposalRequest(state, theorem, ())])
            if not cands:
                print("  providers proposed nothing")
                continue
            for c in cands:
                res = repl.apply_tactic(state, c.tactic)
                mark = {"ok": "✓", "solved": "★"}.get(res.status, "✗")
                note = "" if res.status in ("ok", "solved") else f"  ({res.status}: {res.message[:60]})"
                print(f"  {mark} [{c.provenance}] {c.tactic}{note}")
        elif line.startswith(":"):
            print(INTERACTIVE_HELP)
        else:
            res = repl.apply_tactic(state, line)
            if res.status in ("ok", "solved"):
                assert res.state is not None
                stack.append(res.state)
                show(res.state)
                if res.state.solved:
                    ok, _ = repl.check_full_proof(theorem, res.state.tactics)
                    print(f"\n  proof: {' ; '.join(res.state.tactics)}")
                    print(f"  certification: {'PASSED' if ok else 'FAILED'}")
            else:
                print(f"  ✗ {res.status}: {res.message[:200]}")


# -- search-tree figure ------------------------------------------------------

def _node_status(state: LeanState, domain: LeanDomain) -> str:
    if state.solved:
        return "solved"
    if domain.expanded(state):
        return "dead" if not domain.valid_actions(state) else "open"
    return "unexpanded"


def render_search_tree(root: Node, domain: LeanDomain, on_path: Set[int], fig: Figure,
                       title: str = "", max_nodes: int = 400) -> None:
    """Draw the search tree on a matplotlib Figure: one row per node in the same
    DFS order as the terminal view, L-shaped connectors, solution path in gold,
    solved states green, dead ends red, unexpanded frontier gray."""
    rows: List[Tuple[Node, int]] = []  # (node, depth), row index = list position

    def walk(node: Node, depth: int) -> None:
        if len(rows) >= max_nodes:
            return
        rows.append((node, depth))
        for _, child in node.edge_dict.values():
            walk(child, depth + 1)

    walk(root, 0)

    fig.clf()
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, family="monospace", loc="left")

    pos = {id(node): (depth * 1.0, -i * 1.0) for i, (node, depth) in enumerate(rows)}
    fills = {"solved": "#c8e6c9", "dead": "#ffcdd2", "unexpanded": "#eeeeee", "open": "white"}
    for node, depth in rows:
        state: LeanState = node.state  # type: ignore[assignment]
        x, y = pos[id(node)]
        if node.parent is not None and id(node.parent) in pos:
            px, py = pos[id(node.parent)]
            ax.plot([px + 0.08, px + 0.08, x - 0.06], [py - 0.28, y, y],
                    color="#aaaaaa", linewidth=0.9, zorder=1)
        status = _node_status(state, domain)
        star = id(node) in on_path
        label = "(root)" if node.parent is None else str(node.parent_action)
        prov = "" if node.parent is None else f" [{getattr(node.parent_action, 'provenance', '?')}]"
        mark = {"solved": " ✓", "dead": " ✗", "unexpanded": " ·", "open": ""}[status]
        text = (f"{'★ ' if star else ''}{label}{prov}{mark}  "
                f"g={node.path_cost:.0f} h={node.heuristic:.1f}  {goal_oneliner(state, width=46)}")
        ax.text(x, y, text, fontsize=8, family="monospace", va="center", ha="left", zorder=2,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=fills[status],
                          edgecolor="#b8860b" if star else "#999999",
                          linewidth=1.6 if star else 0.7))

    depth_max = max(d for _, d in rows)
    ax.set_xlim(-0.4, depth_max + 9.0)
    ax.set_ylim(-len(rows) + 0.2, 1.2)


# -- traced search -----------------------------------------------------------

def traced_search(theorem: TheoremSpec, repl: REPLManager, domain: LeanDomain,
                  value_provider: ValueProvider, weight: float = 1.0, batch_size: int = 1,
                  eps: float = 0.0, itr_max: int = 100,
                  max_tree_lines: int = 200, fig_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Search one theorem, narrating each iteration, then print the tree."""
    search = GraphSearchHeurNodeActsEnum(
        domain, PFNsHeurV(heurv=as_heurv(value_provider)),
        batch_size=batch_size, weight=weight, eps=eps)

    root = repl.init_theorem(theorem)
    goal = LeanGoal(theorem)
    (instance,) = search.make_instances([root], [goal], inst_infos=[theorem.name])
    search.add_instances([instance])

    print(f"\n=== search: {theorem.name}  (W={weight}, B={batch_size}, itr_max={itr_max}) ===")
    print(f"{theorem.statement}\n")
    print(f"root  h={instance.root_node.heuristic:.1f}")
    print(format_goals(root))

    while search.instances:
        itr = instance.itr
        popped, _ = search.step()
        for node in popped:
            state: LeanState = node.state  # type: ignore[assignment]
            f = weight * node.path_cost + node.heuristic
            print(f"\nitr {itr}: pop f={f:.1f} (g={node.path_cost:.0f}, h={node.heuristic:.1f})"
                  f"  via {node.parent_action or '(root)'}")
            print(format_goals(state))
            acts = domain.valid_actions(state)
            if state.solved:
                print("  → solved state popped; goal recorded")
            elif not acts:
                n_failed = len(domain.failed_tactics(state))
                print(f"  → dead end: 0 valid tactics ({n_failed} candidates failed)")
            else:
                shown = ", ".join(f"{a.tactic} [{a.provenance}]" for a in acts[:8])
                more = f" (+{len(acts) - 8} more)" if len(acts) > 8 else ""
                print(f"  → {len(acts)} valid: {shown}{more}")
        search.remove_finished_instances(itr_max)

    solved = instance.has_soln()
    tactics: List[str] = []
    print(f"\n=== search tree ===")
    on_path: Set[int] = set()
    if solved:
        node = instance.goal_node
        while node is not None:
            on_path.add(id(node))
            node = node.parent
    _print_tree(instance.root_node, domain, on_path, "", True, [max_tree_lines])

    if solved:
        _, actions, _, cost = get_path(instance.goal_node)
        tactics = [a.tactic for a in actions]
        ok, _ = repl.check_full_proof(theorem, tuple(tactics))
        print(f"\nSOLVED in {instance.itr} iterations, cost {cost:.0f}"
              f" | certification: {'PASSED' if ok else 'FAILED'}")
        print(f"proof: {' ; '.join(tactics)}")
    else:
        print(f"\nUNSOLVED after {instance.itr} iterations "
              f"({'frontier exhausted' if instance.frontier_size() == 0 else 'iteration limit'})")

    if fig_path is not None:
        outcome = f"SOLVED: {' ; '.join(tactics)}" if solved else "UNSOLVED"
        n_rows = _count_nodes(instance.root_node)
        fig = Figure(figsize=(13, max(3.0, 0.42 * n_rows + 1.5)))
        render_search_tree(instance.root_node, domain, on_path, fig,
                           title=f"{theorem.statement}\n{outcome}   "
                                 f"(W={weight}, {instance.itr} iterations)")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[dxlean] wrote search tree figure: {fig_path}")
    return solved, tactics


def _count_nodes(root: Node) -> int:
    n = 1
    for _, child in root.edge_dict.values():
        n += _count_nodes(child)
    return n


def _print_tree(node: Node, domain: LeanDomain, on_path: Set[int], prefix: str,
                is_last: bool, budget: List[int]) -> None:
    if budget[0] <= 0:
        return
    budget[0] -= 1
    state: LeanState = node.state  # type: ignore[assignment]
    if node.parent is None:
        connector, label = "", "(root)"
    else:
        connector = prefix + ("└─ " if is_last else "├─ ")
        label = str(node.parent_action)
    star = "★ " if id(node) in on_path else ""
    if state.solved:
        status = "✓"
    elif domain.expanded(state) and not domain.valid_actions(state):
        status = "✗"
    elif not domain.expanded(state):
        status = "·"
    else:
        status = " "
    print(f"{connector}{star}{label}  {status} g={node.path_cost:.0f} h={node.heuristic:.1f}  {goal_oneliner(state)}")
    children = [child for _, child in node.edge_dict.values()]
    child_prefix = "" if node.parent is None else prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(children):
        _print_tree(child, domain, on_path, child_prefix, i == len(children) - 1, budget)
    if budget[0] == 0:
        print(f"{child_prefix}… (tree truncated)")
        budget[0] = -1
