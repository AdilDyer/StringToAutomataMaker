"""
tests/test_lstar.py
===================
Tests for automata/lstar.py (Angluin's L* active DFA learner):
  - Learns minimal DFA matching oracle across all 5 GATE examples
  - Learns 'no consecutive 0s' (3 states) and '3rd symbol from end is 1' (8 states)
  - Raises ValueError cleanly on non-regular languages (e.g. a^n b^n)
"""

import pytest
from automata.lstar import learn_dfa_lstar
from automata.minimize import minimize
from automata.verify import verify
from llm.sandbox import compile_oracle
from examples import EXAMPLES


class TestLStar:
    @pytest.mark.parametrize("example", EXAMPLES, ids=[ex["description"][:35] for ex in EXAMPLES])
    def test_lstar_learns_gate_languages(self, example):
        mj = example["machine_json"]
        oracle_fn, _ = compile_oracle(mj["oracle_body"])
        dfa, rounds = learn_dfa_lstar(example["alphabet"], oracle_fn, max_verify_len=10)
        min_dfa, _ = minimize(dfa)
        assert len(min_dfa.states) == example["expected_min_states"]
        vr = verify(min_dfa, oracle_fn, max_len=10)
        assert vr["all_mismatch_count"] == 0

    def test_no_consecutive_zeros(self):
        oracle_fn, _ = compile_oracle("return '00' not in s")
        dfa, _ = learn_dfa_lstar(["0", "1"], oracle_fn, max_verify_len=10)
        min_dfa, _ = minimize(dfa)
        assert len(min_dfa.states) == 3
        assert verify(min_dfa, oracle_fn, max_len=10)["all_mismatch_count"] == 0

    def test_third_from_end_is_one(self):
        oracle_fn, _ = compile_oracle("return len(s) >= 3 and s[-3] == '1'")
        dfa, _ = learn_dfa_lstar(["0", "1"], oracle_fn, max_verify_len=10)
        min_dfa, _ = minimize(dfa)
        assert len(min_dfa.states) == 8
        assert verify(min_dfa, oracle_fn, max_len=10)["all_mismatch_count"] == 0

    def test_non_regular_language_stops_cleanly(self):
        # L = {a^n b^n} is context-free, not regular
        oracle_fn, _ = compile_oracle(
            "n = len(s)\n"
            "return n % 2 == 0 and s == 'a' * (n // 2) + 'b' * (n // 2)"
        )
        with pytest.raises(ValueError):
            learn_dfa_lstar(["a", "b"], oracle_fn, max_verify_len=10, max_rows=10)
