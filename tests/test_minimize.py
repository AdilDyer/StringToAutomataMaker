"""
tests/test_minimize.py
======================
Tests for automata/minimize.py:
  - Minimized DFA agrees with original on all strings up to length 8
  - Minimizing an already-minimal DFA changes nothing (same number of states)
  - Unreachable states are removed before minimization
  - A redundant DFA (extra equivalent states) is correctly minimized
"""

import pytest
import itertools
from automata.dfa import DFA
from automata.minimize import minimize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_strings(alphabet, max_len):
    for length in range(max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield "".join(combo)


def _dfa_agrees(dfa_a: DFA, dfa_b: DFA, max_len: int = 8) -> bool:
    """Return True iff both DFAs agree on all strings up to max_len."""
    # Complete both for safe simulation
    from automata.minimize import _copy_dfa
    a = _copy_dfa(dfa_a); a.complete()
    b = _copy_dfa(dfa_b); b.complete()
    for s in _all_strings(a.alphabet, max_len):
        if a.accepts(s) != b.accepts(s):
            return False
    return True


# ---------------------------------------------------------------------------
# Already-minimal DFAs
# ---------------------------------------------------------------------------

class TestAlreadyMinimal:
    def _make_mod3_dfa(self):
        """DFA for #a ≡ 0 (mod 3) — already minimal (3 states)."""
        return DFA(
            states=["q0", "q1", "q2"],
            alphabet=["a", "b"],
            start="q0",
            finals=["q0"],
            delta={
                "q0": {"a": "q1", "b": "q0"},
                "q1": {"a": "q2", "b": "q1"},
                "q2": {"a": "q0", "b": "q2"},
            },
        )

    def test_minimal_dfa_state_count_unchanged(self):
        dfa = self._make_mod3_dfa()
        min_dfa, _ = minimize(dfa)
        # Complete both before counting (dead state may or may not be added)
        from automata.minimize import _copy_dfa
        orig = _copy_dfa(dfa); orig.complete()
        assert len(min_dfa.states) == len(orig.states), (
            f"Already-minimal DFA should stay at {len(orig.states)} states, "
            f"got {len(min_dfa.states)}"
        )

    def test_minimal_dfa_agrees_with_original(self):
        dfa = self._make_mod3_dfa()
        min_dfa, _ = minimize(dfa)
        assert _dfa_agrees(dfa, min_dfa), "Minimized DFA should agree with original."


# ---------------------------------------------------------------------------
# Redundant DFA (extra equivalent states that should be merged)
# ---------------------------------------------------------------------------

class TestRedundantDFA:
    def _make_redundant_dfa(self):
        """
        DFA accepting strings over {a,b} ending in 'a', but with an extra
        redundant copy of q1 (call it q1_copy) that is equivalent to q1.
        Original minimal form has 2 states; this has 3.
        """
        return DFA(
            states=["q0", "q1", "q1_copy"],
            alphabet=["a", "b"],
            start="q0",
            finals=["q1", "q1_copy"],
            delta={
                "q0": {"a": "q1", "b": "q0"},
                "q1": {"a": "q1_copy", "b": "q0"},
                "q1_copy": {"a": "q1", "b": "q0"},
            },
        )

    def test_redundant_dfa_minimizes_to_fewer_states(self):
        dfa = self._make_redundant_dfa()
        min_dfa, _ = minimize(dfa)
        # q1 and q1_copy are equivalent → merged.  Result: 2 states.
        assert len(min_dfa.states) == 2, (
            f"Redundant DFA should minimize to 2 states, got {len(min_dfa.states)}: "
            f"{min_dfa.states}"
        )

    def test_redundant_dfa_minimized_agrees(self):
        dfa = self._make_redundant_dfa()
        min_dfa, _ = minimize(dfa)
        assert _dfa_agrees(dfa, min_dfa)


# ---------------------------------------------------------------------------
# Unreachable states are removed
# ---------------------------------------------------------------------------

class TestUnreachable:
    def test_unreachable_states_removed(self):
        # q_dead_unreachable is defined but never reached from q0
        dfa = DFA(
            states=["q0", "q1", "q_dead_unreachable"],
            alphabet=["a"],
            start="q0",
            finals=["q1"],
            delta={
                "q0": {"a": "q1"},
                "q1": {"a": "q0"},
                "q_dead_unreachable": {"a": "q_dead_unreachable"},
            },
        )
        _, info = minimize(dfa)
        assert "q_dead_unreachable" in info["removed_unreachable"]


# ---------------------------------------------------------------------------
# Minimized DFA agrees with original (various languages)
# ---------------------------------------------------------------------------

class TestAgreement:
    @pytest.mark.parametrize("dfa_fixture,name", [
        (
            # ends in 'a'
            DFA(
                ["q0", "q1"], ["a", "b"], "q0", ["q1"],
                {"q0": {"a": "q1", "b": "q0"}, "q1": {"a": "q1", "b": "q0"}},
            ),
            "ends in a",
        ),
        (
            # contains 'aa'
            DFA(
                ["q0", "q1", "q2"], ["a", "b"], "q0", ["q2"],
                {
                    "q0": {"a": "q1", "b": "q0"},
                    "q1": {"a": "q2", "b": "q0"},
                    "q2": {"a": "q2", "b": "q2"},
                },
            ),
            "contains aa",
        ),
    ])
    def test_minimized_agrees(self, dfa_fixture, name):
        min_dfa, _ = minimize(dfa_fixture)
        assert _dfa_agrees(dfa_fixture, min_dfa), f"Agreement failed for: {name}"
