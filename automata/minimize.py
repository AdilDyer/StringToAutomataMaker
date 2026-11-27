"""
automata/minimize.py
====================
DFA minimization using the Table-Filling Algorithm (Myhill-Nerode / Hopcroft).

The algorithm operates on COMPLETE DFAs (call dfa.complete() first).

Algorithm outline
-----------------
1. Remove unreachable states (BFS from start).
2. Build a triangular table of all unordered pairs {p, q} with p ≠ q.
3. BASE CASE: Mark (p, q) as distinguishable if exactly one of p, q is a
   final state.  (They respond differently to the empty string ε.)
4. INDUCTIVE STEP: Repeat until no new pairs are marked:
     For every unmarked pair (p, q) and every symbol a:
       if (δ(p,a), δ(q,a)) is already marked → mark (p, q).
   Record the round number in which each pair was first marked.
5. EQUIVALENCE CLASSES: Unmarked pairs are equivalent (Myhill-Nerode
   indistinguishable).  Use union-find to build equivalence classes.
6. MERGED DFA: Each class becomes one state.  Transitions, start, and
   finals are updated accordingly.

Time complexity: O(n² · |Σ|) per round, O(n) rounds → O(n³ · |Σ|) total.
(Hopcroft's algorithm is O(n log n · |Σ|) but table-filling is easier to
visualise and is the standard GATE / textbook method.)
"""

from __future__ import annotations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from automata.dfa import DFA


# ---------------------------------------------------------------------------
# Union-Find (path-compressed, for building equivalence classes)
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def classes(self):
        from collections import defaultdict
        groups: Dict = defaultdict(set)
        for e in self.parent:
            groups[self.find(e)].add(e)
        return list(groups.values())


# ---------------------------------------------------------------------------
# Main minimization entry point
# ---------------------------------------------------------------------------

def minimize(dfa: DFA) -> Tuple[DFA, dict]:
    """
    Minimize *dfa* using the table-filling algorithm.

    Returns
    -------
    min_dfa  : the minimized (complete) DFA
    info     : a dict containing:
                 "table"      : dict {(p,q): round_marked | None}
                                None = unmarked (equivalent)
                 "round_log"  : list of {"round": int, "pairs": [(p,q)],
                                          "reason": str}
                 "classes"    : list of frozensets (each equivalence class)
                 "removed_unreachable": list of state names removed
                 "dead_added" : bool — whether a dead state was added
    """
    # --- Step 0: work on a *copy* so we don't mutate the caller's DFA --------
    dfa_copy = _copy_dfa(dfa)

    # Step 0a: Remove unreachable states
    removed_unreachable = dfa_copy.remove_unreachable()

    # Step 0b: Complete the DFA (add dead state if needed)
    dead_added = dfa_copy.complete()

    states = dfa_copy.states
    n = len(states)

    # --- Step 1: Build the triangular table ----------------------------------
    # Key: frozenset({p, q}), Value: round number (int) or None (unmarked)
    table: Dict[FrozenSet[str], Optional[int]] = {}
    all_pairs: List[Tuple[str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            p, q = states[i], states[j]
            table[frozenset({p, q})] = None
            all_pairs.append((p, q))

    # --- Step 2: Base case (round 0) -----------------------------------------
    # Mark (p, q) if exactly one is in finals.
    round_log: List[Dict] = []
    base_marked = []

    for (p, q) in all_pairs:
        p_final = p in dfa_copy.finals
        q_final = q in dfa_copy.finals
        if p_final != q_final:
            table[frozenset({p, q})] = 0
            base_marked.append((p, q))

    if base_marked:
        round_log.append({
            "round": 0,
            "pairs": base_marked,
            "reason": "Exactly one state is accepting (base case).",
        })

    # --- Step 3: Inductive fixpoint ------------------------------------------
    changed = True
    round_num = 1

    while changed:
        changed = False
        newly_marked = []

        for (p, q) in all_pairs:
            key = frozenset({p, q})
            if table[key] is not None:
                continue  # already marked

            for a in dfa_copy.alphabet:
                dp = dfa_copy.delta.get(p, {}).get(a)
                dq = dfa_copy.delta.get(q, {}).get(a)
                if dp is None or dq is None:
                    continue
                if dp == dq:
                    continue  # same target → not distinguishable from this symbol
                pair_key = frozenset({dp, dq})
                if table.get(pair_key) is not None:
                    # (δ(p,a), δ(q,a)) is distinguishable → so is (p,q)
                    table[key] = round_num
                    newly_marked.append((p, q))
                    changed = True
                    break  # one symbol is enough

        if newly_marked:
            round_log.append({
                "round": round_num,
                "pairs": newly_marked,
                "reason": (
                    f"Distinguished by propagation: ∃ symbol a s.t. "
                    f"(δ(p,a),δ(q,a)) was already marked."
                ),
            })
        round_num += 1

    # --- Step 4: Build equivalence classes ------------------------------------
    uf = _UF(states)
    for (p, q) in all_pairs:
        if table[frozenset({p, q})] is None:
            uf.union(p, q)

    eq_classes = [frozenset(cls) for cls in uf.classes()]

    # --- Step 5: Build the minimized DFA --------------------------------------
    # Name each class by its sorted members, e.g. "{q1,q3}"
    def class_name(cls: FrozenSet[str]) -> str:
        members = sorted(cls)
        if len(members) == 1:
            return members[0]
        return "{" + ",".join(members) + "}"

    # Map each original state to its class representative (the class name)
    state_to_class: Dict[str, str] = {}
    for cls in eq_classes:
        name = class_name(cls)
        for s in cls:
            state_to_class[s] = name

    new_states = sorted(set(state_to_class.values()))
    new_start = state_to_class[dfa_copy.start]
    new_finals = sorted({
        state_to_class[s] for s in dfa_copy.finals
    })

    # Build new delta: take transitions from any representative of each class
    new_delta: Dict[str, Dict[str, str]] = {}
    for cls in eq_classes:
        rep = next(iter(cls))  # pick one representative
        cls_name = state_to_class[rep]
        new_delta[cls_name] = {}
        for a in dfa_copy.alphabet:
            tgt = dfa_copy.delta.get(rep, {}).get(a)
            if tgt is not None:
                new_delta[cls_name][a] = state_to_class[tgt]

    min_dfa = DFA(
        states=new_states,
        alphabet=dfa_copy.alphabet,
        start=new_start,
        finals=new_finals,
        delta=new_delta,
    )

    info = {
        "table": table,
        "round_log": round_log,
        "classes": eq_classes,
        "removed_unreachable": removed_unreachable,
        "dead_added": dead_added,
        "all_pairs": all_pairs,
        "states_after_completion": states,  # for table display
    }
    return min_dfa, info


# ---------------------------------------------------------------------------
# Helper: deep-copy a DFA
# ---------------------------------------------------------------------------

def _copy_dfa(dfa: DFA) -> DFA:
    """Return an independent copy of *dfa*."""
    return DFA(
        states=list(dfa.states),
        alphabet=list(dfa.alphabet),
        start=dfa.start,
        finals=set(dfa.finals),
        delta={s: dict(row) for s, row in dfa.delta.items()},
    )
