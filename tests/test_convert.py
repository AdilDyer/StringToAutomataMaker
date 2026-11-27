"""
tests/test_convert.py
=====================
Tests for automata/convert.py:
  - DFA -> Right-Linear Regular Grammar equivalence
  - DFA -> Regular Expression (State Elimination) equivalence
  - Edge cases: empty language, epsilon-only language, uppercase alphabet symbols
"""

import itertools
import re
import pytest

from automata.dfa import DFA
from automata.nfa import NFA
from automata.minimize import minimize
from automata.convert import dfa_to_grammar, dfa_to_regex
from examples import EXAMPLES
from ui.pipeline import build_automaton_from_json


def _all_strings(alphabet, max_len):
    for length in range(max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield "".join(combo)


def _grammar_accepts(grammar: dict, s: str) -> bool:
    """Simulate a right-linear CFG on *s* using BFS over (non_terminal, index)."""
    prods = grammar["productions"]
    if not prods:
        return False
    start = grammar["start"]
    frontier = {(start, 0)}
    visited = set(frontier)

    while frontier:
        nt, idx = frontier.pop()
        for rhs in prods.get(nt, []):
            if rhs == "ε":
                if idx == len(s):
                    return True
            else:
                sym, next_nt = rhs.split(" ", 1)
                if idx < len(s) and s[idx] == sym:
                    nxt = (next_nt, idx + 1)
                    if nxt not in visited:
                        visited.add(nxt)
                        frontier.add(nxt)
    return False


class TestConvertGateLanguages:
    @pytest.mark.parametrize("example", EXAMPLES, ids=[ex["description"][:35] for ex in EXAMPLES])
    def test_regex_matches_minimized_dfa(self, example):
        dfa, _ = build_automaton_from_json(example["machine_json"])
        min_dfa, _ = minimize(dfa)
        rx_info = dfa_to_regex(min_dfa)
        pattern = re.compile(rx_info["python_regex"])

        for s in _all_strings(min_dfa.alphabet, 8):
            assert bool(pattern.fullmatch(s)) == min_dfa.accepts(s), (
                f"Regex {rx_info['regex']!r} disagreed with DFA on {s!r}"
            )

    @pytest.mark.parametrize("example", EXAMPLES, ids=[ex["description"][:35] for ex in EXAMPLES])
    def test_grammar_matches_minimized_dfa(self, example):
        dfa, _ = build_automaton_from_json(example["machine_json"])
        min_dfa, _ = minimize(dfa)
        grammar = dfa_to_grammar(min_dfa)

        for s in _all_strings(min_dfa.alphabet, 8):
            assert _grammar_accepts(grammar, s) == min_dfa.accepts(s), (
                f"Grammar disagreed with DFA on {s!r}\n{grammar['text']}"
            )


class TestConvertEdgeCases:
    def test_empty_language(self):
        dfa = DFA(["q0"], ["a", "b"], "q0", [], {"q0": {"a": "q0", "b": "q0"}})
        rx = dfa_to_regex(dfa)
        gr = dfa_to_grammar(dfa)
        assert rx["regex"] == "∅"
        assert gr["productions"] == {}
        assert not _grammar_accepts(gr, "")
        assert not _grammar_accepts(gr, "a")

    def test_epsilon_only_language(self):
        dfa = DFA(
            ["q0", "dead"], ["a", "b"], "q0", ["q0"],
            {"q0": {"a": "dead", "b": "dead"}, "dead": {"a": "dead", "b": "dead"}},
        )
        rx = dfa_to_regex(dfa)
        gr = dfa_to_grammar(dfa)
        assert rx["regex"] == "ε"
        pat = re.compile(rx["python_regex"])
        for s in _all_strings(["a", "b"], 4):
            assert bool(pat.fullmatch(s)) == (s == "")
            assert _grammar_accepts(gr, s) == (s == "")

    def test_uppercase_alphabet_no_collision(self):
        dfa = DFA(
            ["q0", "q1"], ["S", "A"], "q0", ["q1"],
            {"q0": {"S": "q1", "A": "q0"}, "q1": {"S": "q1", "A": "q0"}},
        )
        gr = dfa_to_grammar(dfa)
        rx = dfa_to_regex(dfa)
        assert set(gr["variables"]).isdisjoint({"S", "A"})
        pat = re.compile(rx["python_regex"])
        for s in _all_strings(["S", "A"], 5):
            assert bool(pat.fullmatch(s)) == dfa.accepts(s)
            assert _grammar_accepts(gr, s) == dfa.accepts(s)

    def test_nfa_to_dfa_to_grammar_and_regex(self):
        nfa = NFA(
            ["q0", "q1", "q2"], ["a", "b"], "q0", ["q0", "q1"],
            {"q0": {"a": ["q0"], "eps": ["q1"]}, "q1": {"b": ["q2"]}, "q2": {"a": ["q1"]}},
        )
        dfa, _ = nfa.to_dfa()
        min_dfa, _ = minimize(dfa)
        gr = dfa_to_grammar(min_dfa)
        rx = dfa_to_regex(min_dfa)
        pat = re.compile(rx["python_regex"])
        for s in _all_strings(["a", "b"], 6):
            assert bool(pat.fullmatch(s)) == min_dfa.accepts(s)
            assert _grammar_accepts(gr, s) == min_dfa.accepts(s)
