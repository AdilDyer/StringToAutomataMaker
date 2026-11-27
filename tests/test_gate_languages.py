"""
tests/test_gate_languages.py
============================
Integration tests for the 5 built-in example languages.
No network calls are made — machines come from hardcoded JSON in examples.py.

For each example:
  1. Build the DFA from hardcoded machine_json
  2. Minimize
  3. Assert the minimal state count equals the expected value
  4. Compile the oracle and verify up to length 10
"""

import pytest
from automata.dfa import DFA
from automata.nfa import NFA
from automata.minimize import minimize
from automata.verify import verify
from llm.sandbox import compile_oracle
from examples import EXAMPLES


# Inline build_automaton_from_json to avoid importing app.py (which needs streamlit)
def build_automaton_from_json(mj):
    """Build a DFA (and optional NFA table) from machine_json dict."""
    raw_transitions = mj.get("transitions", {})
    if mj["machine_type"] == "DFA":
        delta = {}
        for src, row in raw_transitions.items():
            delta[src] = {}
            for sym, tgts in row.items():
                delta[src][sym] = tgts[0] if isinstance(tgts, list) else tgts
        dfa = DFA(
            states=mj["states"], alphabet=mj["alphabet"],
            start=mj["start"], finals=mj["finals"], delta=delta,
        )
        return dfa, None
    else:
        nfa_delta = {}
        for src, row in raw_transitions.items():
            nfa_delta[src] = {}
            for sym, tgts in row.items():
                nfa_delta[src][sym] = list(tgts) if isinstance(tgts, list) else [tgts]
        nfa = NFA(
            states=mj["states"], alphabet=mj["alphabet"],
            start=mj["start"], finals=mj["finals"], delta=nfa_delta,
        )
        dfa, nfa_table = nfa.to_dfa()
        return dfa, nfa_table


@pytest.mark.parametrize("example", EXAMPLES, ids=[ex["description"][:40] for ex in EXAMPLES])
class TestGateLanguages:
    def test_state_count_after_minimization(self, example):
        mj = example["machine_json"]
        dfa, _ = build_automaton_from_json(mj)
        errors = dfa.validate()
        assert errors == [], f"Validation errors: {errors}"

        min_dfa, info = minimize(dfa)
        expected = example["expected_min_states"]
        assert len(min_dfa.states) == expected, (
            f"Expected {expected} states for '{example['description']}', "
            f"got {len(min_dfa.states)}: {sorted(min_dfa.states)}"
        )

    def test_oracle_verification_passes(self, example):
        mj = example["machine_json"]
        dfa, _ = build_automaton_from_json(mj)
        dfa.complete()

        oracle_body = mj.get("oracle_body", "")
        if not oracle_body:
            pytest.skip("No oracle defined for this example.")

        oracle_fn, _ = compile_oracle(oracle_body)
        result = verify(dfa, oracle_fn, max_len=10)

        assert result["all_mismatch_count"] == 0, (
            f"Oracle mismatches for '{example['description']}': "
            f"{result['mismatches']}"
        )

    def test_minimized_dfa_matches_oracle(self, example):
        mj = example["machine_json"]
        dfa, _ = build_automaton_from_json(mj)
        min_dfa, _ = minimize(dfa)

        oracle_body = mj.get("oracle_body", "")
        if not oracle_body:
            pytest.skip("No oracle defined for this example.")

        oracle_fn, _ = compile_oracle(oracle_body)
        result = verify(min_dfa, oracle_fn, max_len=10)

        assert result["all_mismatch_count"] == 0, (
            f"Minimized DFA oracle mismatches for '{example['description']}': "
            f"{result['mismatches']}"
        )
