"""
automata/verify.py
==================
Verifier: enumerate strings over the alphabet up to length N and compare
the DFA's decision against an oracle function.

Why only up to length N?
  Formal language equivalence for DFAs CAN be decided exactly (in polynomial
  time via table-filling on the product automaton), but that requires the oracle
  to be expressed as a DFA.  Here the oracle is arbitrary Python code, so we
  fall back to bounded enumeration.  The bounded test is *sound* (any reported
  mismatch is a real error) but *incomplete* (passing does not guarantee full
  correctness).  See the README for discussion.

String enumeration order: by length (shortest first), then lexicographic within
each length.  This ensures the first reported counterexample is the shortest.
"""

from __future__ import annotations
import itertools
import time
from typing import Callable, Iterator, List, Optional, Tuple

from automata.dfa import DFA

MAX_TOTAL_DEFAULT = 300_000  # safety cap
MAX_VERIFY_TIME_S = 20       # seconds


def _all_strings(alphabet: List[str], max_len: int) -> Iterator[str]:
    """
    Yield all strings over *alphabet* in length order (shortest first),
    with lexicographic ordering within each length.
    Includes the empty string.
    """
    for length in range(max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield "".join(combo)


def count_strings(alphabet: List[str], max_len: int) -> int:
    """Return |Σ|^0 + |Σ|^1 + … + |Σ|^max_len = (|Σ|^(max_len+1) − 1) / (|Σ| − 1)."""
    k = len(alphabet)
    if k == 0:
        return 1  # only the empty string
    if k == 1:
        return max_len + 1
    return (k ** (max_len + 1) - 1) // (k - 1)


def verify(
    dfa: DFA,
    oracle: Callable[[str], bool],
    max_len: int = 10,
    max_total: int = MAX_TOTAL_DEFAULT,
) -> dict:
    """
    Enumerate all strings over *dfa.alphabet* of length 0 … max_len, compare
    dfa.accepts(s) with oracle(s), and collect mismatches.

    Returns a dict:
      total_checked  : int
      max_len_used   : int   (may be reduced due to cap)
      mismatches     : list of {"string": s, "dfa": bool, "oracle": bool}
      capped         : bool  (True if max_len was auto-reduced)
      timed_out      : bool
      shortest_counterexample : str | None  (shortest mismatch found)
    """
    # Auto-reduce max_len if total count exceeds cap
    capped = False
    while max_len > 0 and count_strings(dfa.alphabet, max_len) > max_total:
        max_len -= 1
        capped = True

    total = count_strings(dfa.alphabet, max_len)
    mismatches = []
    timed_out = False
    start_time = time.monotonic()

    for s in _all_strings(dfa.alphabet, max_len):
        if time.monotonic() - start_time > MAX_VERIFY_TIME_S:
            timed_out = True
            break
        dfa_ans = dfa.accepts(s)
        try:
            oracle_ans = bool(oracle(s))
        except Exception as exc:
            oracle_ans = None  # treat as mismatch
            mismatches.append({
                "string": repr(s),
                "dfa": dfa_ans,
                "oracle": f"ERROR: {exc}",
            })
            continue

        if dfa_ans != oracle_ans:
            mismatches.append({"string": s, "dfa": dfa_ans, "oracle": oracle_ans})

    shortest = None
    if mismatches:
        # Already in length order because we enumerate shortest-first
        first = mismatches[0]
        if isinstance(first["string"], str):
            shortest = first["string"]

    return {
        "total_checked": total,
        "max_len_used": max_len,
        "mismatches": mismatches[:10],  # limit display to 10
        "all_mismatch_count": len(mismatches),
        "capped": capped,
        "timed_out": timed_out,
        "shortest_counterexample": shortest,
    }


def cross_verify(dfa_a: DFA, dfa_b: DFA, max_len: int = 10) -> dict:
    """
    Verify that two DFAs agree on all strings up to max_len.
    Used to confirm the minimized DFA is equivalent to the original.
    """
    mismatches = []
    total = count_strings(dfa_a.alphabet, max_len)

    for s in _all_strings(dfa_a.alphabet, max_len):
        a_ans = dfa_a.accepts(s)
        b_ans = dfa_b.accepts(s)
        if a_ans != b_ans:
            mismatches.append({"string": s, "original": a_ans, "minimized": b_ans})

    return {
        "total_checked": total,
        "mismatches": mismatches[:10],
        "all_mismatch_count": len(mismatches),
        "agree": len(mismatches) == 0,
    }
