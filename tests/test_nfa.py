"""
tests/test_nfa.py
=================
Tests for automata/nfa.py:
  - Epsilon closure
  - Subset construction
  - Classic "third symbol from end is 1" NFA → 8-state DFA
  - NFA with epsilon moves
"""

import pytest
from automata.nfa import NFA, EPSILON


# ---------------------------------------------------------------------------
# Epsilon closure
# ---------------------------------------------------------------------------

class TestEpsilonClosure:
    def test_no_epsilon_transitions(self):
        nfa = NFA(
            states=["q0", "q1"],
            alphabet=["a"],
            start="q0",
            finals=["q1"],
            delta={"q0": {"a": ["q1"]}},
        )
        closure = nfa.epsilon_closure({"q0"})
        assert closure == frozenset({"q0"})

    def test_single_epsilon_hop(self):
        nfa = NFA(
            states=["q0", "q1", "q2"],
            alphabet=["a"],
            start="q0",
            finals=["q2"],
            delta={
                "q0": {EPSILON: ["q1"]},
                "q1": {"a": ["q2"]},
            },
        )
        closure = nfa.epsilon_closure({"q0"})
        assert closure == frozenset({"q0", "q1"})

    def test_transitive_epsilon_closure(self):
        """q0 -ε-> q1 -ε-> q2: closure of {q0} = {q0,q1,q2}."""
        nfa = NFA(
            states=["q0", "q1", "q2"],
            alphabet=["a"],
            start="q0",
            finals=[],
            delta={
                "q0": {EPSILON: ["q1"]},
                "q1": {EPSILON: ["q2"]},
            },
        )
        closure = nfa.epsilon_closure({"q0"})
        assert closure == frozenset({"q0", "q1", "q2"})

    def test_epsilon_cycle_handled(self):
        """q0 -ε-> q1 -ε-> q0: should not loop infinitely."""
        nfa = NFA(
            states=["q0", "q1"],
            alphabet=["a"],
            start="q0",
            finals=[],
            delta={
                "q0": {EPSILON: ["q1"]},
                "q1": {EPSILON: ["q0"]},
            },
        )
        closure = nfa.epsilon_closure({"q0"})
        assert closure == frozenset({"q0", "q1"})


# ---------------------------------------------------------------------------
# Subset construction: "third symbol from end is 1"
# ---------------------------------------------------------------------------
# Classic example from Sipser / GATE textbooks.
# Language: binary strings where the third-to-last character is 1.
# The minimal NFA has 4 states; the equivalent DFA has exactly 8 reachable states.

def _make_third_from_end_nfa() -> NFA:
    """
    NFA for L = { w ∈ {0,1}* : |w| ≥ 3 and w[|w|-3] = '1' }
    States: q0 (start), q1 (saw the special 1), q2 (one more symbol after),
            q3 (two more symbols after — accept).
    """
    return NFA(
        states=["q0", "q1", "q2", "q3"],
        alphabet=["0", "1"],
        start="q0",
        finals=["q3"],
        delta={
            "q0": {"0": ["q0"], "1": ["q0", "q1"]},
            "q1": {"0": ["q2"], "1": ["q2"]},
            "q2": {"0": ["q3"], "1": ["q3"]},
            "q3": {},
        },
    )


class TestSubsetConstruction:
    def test_third_from_end_dfa_has_8_reachable_states(self):
        nfa = _make_third_from_end_nfa()
        dfa, table = nfa.to_dfa()
        # Remove unreachable states (there should be none for this NFA)
        dfa.remove_unreachable()
        assert len(dfa.states) == 8, (
            f"Expected 8 reachable DFA states, got {len(dfa.states)}: {dfa.states}"
        )

    def test_third_from_end_accepts_correct_strings(self):
        """
        Language: w[|w|-3] = '1'  (third symbol from the end is 1).
        For a string of length n, the third-to-last character is at index n-3.
        """
        nfa = _make_third_from_end_nfa()
        dfa, _ = nfa.to_dfa()
        dfa.complete()

        # Verify via oracle: w[len(w)-3] == '1'
        def oracle(s):
            return len(s) >= 3 and s[len(s) - 3] == '1'

        # Strings that should be accepted (w[-3] == '1')
        for s in ["100", "101", "110", "111", "0100", "1100", "1010"]:
            assert dfa.accepts(s) == oracle(s), (
                f"DFA({'accept' if dfa.accepts(s) else 'reject'}) but oracle "
                f"({'accept' if oracle(s) else 'reject'}) for '{s}'"
            )

        # Strings that should be rejected
        for s in ["", "0", "1", "00", "01", "10", "11", "001", "010", "011",
                  "000", "1000", "1001", "0000"]:
            assert dfa.accepts(s) == oracle(s), (
                f"DFA({'accept' if dfa.accepts(s) else 'reject'}) but oracle "
                f"({'accept' if oracle(s) else 'reject'}) for '{s}'"
            )

    def test_subset_construction_table_rows_match_states(self):
        nfa = _make_third_from_end_nfa()
        dfa, table = nfa.to_dfa()
        assert len(table) == len(dfa.states), (
            "Table rows should equal number of DFA states."
        )


# ---------------------------------------------------------------------------
# NFA with epsilon transitions on "a*(ba)*"
# ---------------------------------------------------------------------------

class TestNFAWithEpsilon:
    def _make_nfa(self) -> NFA:
        """
        NFA for a* (ba)*
        q0: start, accepts ε
        q0 -a-> q0  (loop for a*)
        q0 -ε-> q1  (start of (ba)* part)
        q1 -b-> q2
        q2 -a-> q1
        q1 is accepting (after each ba)
        q0 is also accepting (a* can be empty, then (ba)* is empty)
        """
        return NFA(
            states=["q0", "q1", "q2"],
            alphabet=["a", "b"],
            start="q0",
            finals=["q0", "q1"],
            delta={
                "q0": {"a": ["q0"], EPSILON: ["q1"]},
                "q1": {"b": ["q2"]},
                "q2": {"a": ["q1"]},
            },
        )

    def test_epsilon_nfa_converts_to_dfa(self):
        nfa = self._make_nfa()
        dfa, table = nfa.to_dfa()
        assert dfa is not None
        assert len(dfa.states) > 0

    def test_epsilon_nfa_accepts_correct_strings(self):
        nfa = self._make_nfa()
        dfa, _ = nfa.to_dfa()
        dfa.complete()

        # Should accept
        for s in ["", "a", "aa", "ba", "aba", "baba", "aaba"]:
            assert dfa.accepts(s), f"Should accept '{s}'"

        # Should reject
        for s in ["b", "ab", "bb", "bab", "bba"]:
            assert not dfa.accepts(s), f"Should reject '{s}'"
