"""
tests/test_dfa.py
=================
Tests for automata/dfa.py:
  - Validation error reporting
  - Dead-state completion
  - Simulation (empty string, normal string, bad symbol)
  - Unreachable-state removal
"""

import pytest
from automata.dfa import DFA, DFAValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_simple_dfa():
    """DFA accepting strings over {a,b} ending in 'a'."""
    return DFA(
        states=["q0", "q1"],
        alphabet=["a", "b"],
        start="q0",
        finals=["q1"],
        delta={
            "q0": {"a": "q1", "b": "q0"},
            "q1": {"a": "q1", "b": "q0"},
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_dfa_has_no_errors(self):
        dfa = make_simple_dfa()
        assert dfa.validate() == []

    def test_bad_start_state(self):
        dfa = DFA(["q0"], ["a"], start="q99", finals=[], delta={})
        errors = dfa.validate()
        assert any("q99" in e for e in errors), errors

    def test_bad_final_state(self):
        dfa = DFA(["q0"], ["a"], start="q0", finals=["q99"], delta={})
        errors = dfa.validate()
        assert any("q99" in e for e in errors), errors

    def test_bad_transition_target(self):
        dfa = DFA(
            ["q0"], ["a"], start="q0", finals=[],
            delta={"q0": {"a": "q_nonexistent"}},
        )
        errors = dfa.validate()
        assert any("q_nonexistent" in e for e in errors), errors

    def test_symbol_not_in_alphabet(self):
        dfa = DFA(
            ["q0", "q1"], ["a"], start="q0", finals=[],
            delta={"q0": {"z": "q1"}},  # 'z' not in alphabet
        )
        errors = dfa.validate()
        assert any("z" in e for e in errors), errors

    def test_missing_transitions_reported(self):
        dfa = make_simple_dfa()
        # Remove 'b' transition from q0
        del dfa.delta["q0"]["b"]
        missing = dfa.missing_transitions()
        assert ("q0", "b") in missing


# ---------------------------------------------------------------------------
# Completion (dead state)
# ---------------------------------------------------------------------------

class TestCompletion:
    def test_complete_dfa_needs_no_dead_state(self):
        dfa = make_simple_dfa()
        added = dfa.complete()
        assert added is False
        assert "dead" not in dfa.states

    def test_incomplete_dfa_gets_dead_state(self):
        # DFA with only 'a' transitions defined
        dfa = DFA(
            ["q0", "q1"], ["a", "b"], start="q0", finals=["q1"],
            delta={"q0": {"a": "q1"}},  # q0 missing 'b'; q1 missing both
        )
        added = dfa.complete()
        assert added is True
        assert "dead" in dfa.states
        # dead self-loops
        assert dfa.delta["dead"]["a"] == "dead"
        assert dfa.delta["dead"]["b"] == "dead"
        # missing transition routed to dead
        assert dfa.delta["q0"]["b"] == "dead"

    def test_dead_state_is_not_accepting(self):
        dfa = DFA(
            ["q0"], ["a"], start="q0", finals=[],
            delta={},  # no transitions
        )
        dfa.complete()
        assert "dead" not in dfa.finals


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_empty_string_accepted_when_start_is_final(self):
        dfa = DFA(["q0"], ["a"], start="q0", finals=["q0"], delta={"q0": {"a": "q0"}})
        accepted, trace, err = dfa.run("")
        assert accepted is True
        assert trace == []
        assert err is None

    def test_empty_string_rejected_when_start_not_final(self):
        dfa = DFA(["q0"], ["a"], start="q0", finals=[], delta={"q0": {"a": "q0"}})
        accepted, trace, err = dfa.run("")
        assert accepted is False
        assert trace == []

    def test_accepts_string_ending_in_a(self):
        dfa = make_simple_dfa()
        assert dfa.accepts("ba") is True
        assert dfa.accepts("aba") is True

    def test_rejects_string_not_ending_in_a(self):
        dfa = make_simple_dfa()
        assert dfa.accepts("b") is False
        assert dfa.accepts("ab") is False

    def test_bad_symbol_returns_error(self):
        dfa = make_simple_dfa()
        accepted, trace, err = dfa.run("axb")
        assert accepted is False
        assert err is not None
        assert "x" in err

    def test_trace_has_correct_length(self):
        dfa = make_simple_dfa()
        _, trace, _ = dfa.run("aba")
        assert len(trace) == 3
        assert trace[0] == ("q0", "a", "q1")
        assert trace[1] == ("q1", "b", "q0")
        assert trace[2] == ("q0", "a", "q1")


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

class TestReachability:
    def test_remove_unreachable_states(self):
        # q2 is not reachable from q0
        dfa = DFA(
            ["q0", "q1", "q2"], ["a"], start="q0", finals=["q1"],
            delta={
                "q0": {"a": "q1"},
                "q1": {"a": "q0"},
                "q2": {"a": "q2"},  # unreachable
            },
        )
        removed = dfa.remove_unreachable()
        assert "q2" in removed
        assert "q2" not in dfa.states

    def test_all_states_reachable_nothing_removed(self):
        dfa = make_simple_dfa()
        removed = dfa.remove_unreachable()
        assert removed == []
