"""
automata/lstar.py
=================
Angluin's L* Active Automata Learning Algorithm (from scratch in plain Python).

Given a membership oracle(s) -> bool and an alphabet, L* learns the unique
minimal complete DFA by maintaining an Observation Table (S, E, T) and using
bounded verification as the equivalence oracle.

Textbook steps:
  1. Initialize S = [""], E = [""].
  2. Make (S, E, T) closed and consistent.
  3. Construct hypothesis DFA H from distinct rows in S.
  4. Run verify(H, oracle, max_len) as equivalence query:
       - No counterexample -> return H (minimal complete DFA).
       - Counterexample w   -> add all prefixes of w to S and repeat.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

from automata.dfa import DFA
from automata.verify import verify


def learn_dfa_lstar(
    alphabet: List[str],
    oracle: Callable[[str], bool],
    max_verify_len: int = 10,
    max_rows: int = 80,
) -> Tuple[DFA, int]:
    """
    Learn a minimal complete DFA matching *oracle* up to *max_verify_len*
    using Angluin's L* observation table algorithm.

    Returns (learned_dfa, rounds_used).
    Raises ValueError if the language requires too many states (e.g. non-regular).
    """
    cache: Dict[str, bool] = {}

    def mq(s: str) -> bool:
        """Cached membership query to the oracle."""
        if s not in cache:
            cache[s] = bool(oracle(s))
        return cache[s]

    S: List[str] = [""]
    E: List[str] = [""]

    def row(u: str) -> Tuple[bool, ...]:
        return tuple(mq(u + e) for e in E)

    def make_closed_and_consistent() -> None:
        while True:
            if len(S) > max_rows:
                raise ValueError(
                    f"L* exceeded {max_rows} observation rows; oracle may be non-regular or too complex."
                )

            # Compute rows for S and S·Σ
            s_rows = {s: row(s) for s in S}
            existing = set(s_rows.values())

            # 1. Check closedness: every s+a must match some row in S
            closed_ok = True
            for s in list(S):
                for a in alphabet:
                    sa = s + a
                    if row(sa) not in existing:
                        S.append(sa)
                        closed_ok = False
                        break
                if not closed_ok:
                    break
            if not closed_ok:
                continue

            # 2. Check consistency: equal rows in S must transition to equal rows
            consistent_ok = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    s1, s2 = S[i], S[j]
                    if s_rows[s1] != s_rows[s2]:
                        continue
                    for a in alphabet:
                        r1, r2 = row(s1 + a), row(s2 + a)
                        if r1 != r2:
                            for idx, e in enumerate(E):
                                if r1[idx] != r2[idx]:
                                    new_e = a + e
                                    if new_e not in E:
                                        E.append(new_e)
                                        consistent_ok = False
                                        break
                        if not consistent_ok:
                            break
                    if not consistent_ok:
                        break
                if not consistent_ok:
                    break

            if consistent_ok:
                break

    def build_hypothesis() -> DFA:
        # Assign deterministic names q0, q1, ... preserving order in S
        rep_by_row: Dict[Tuple[bool, ...], str] = {}
        state_by_row: Dict[Tuple[bool, ...], str] = {}
        for s in S:
            r = row(s)
            if r not in state_by_row:
                name = f"q{len(state_by_row)}"
                state_by_row[r] = name
                rep_by_row[r] = s

        states = list(state_by_row.values())
        start = state_by_row[row("")]
        finals = [name for r, name in state_by_row.items() if r[0]]  # E[0] == ""
        delta: Dict[str, Dict[str, str]] = {}
        for r, name in state_by_row.items():
            rep = rep_by_row[r]
            delta[name] = {a: state_by_row[row(rep + a)] for a in alphabet}

        return DFA(states=states, alphabet=alphabet, start=start, finals=finals, delta=delta)

    rounds = 0
    while True:
        rounds += 1
        if rounds > 30:
            raise ValueError("L* exceeded 30 equivalence rounds; oracle may be non-regular.")
        make_closed_and_consistent()
        hyp = build_hypothesis()
        vr = verify(hyp, mq, max_len=max_verify_len)
        if vr["all_mismatch_count"] == 0:
            return hyp, rounds

        cex: Optional[str] = vr["shortest_counterexample"]
        if cex is None:
            raise ValueError("Oracle raised an error during L* equivalence check.")

        # Add counterexample and all its prefixes to S
        for k in range(1, len(cex) + 1):
            pref = cex[:k]
            if pref not in S:
                S.append(pref)
