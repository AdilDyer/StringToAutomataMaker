"""
automata/nfa.py
===============
NFA class (with ε-transitions) implementing:
  - ε-closure computation
  - Subset construction (NFA → DFA)

An NFA is a 5-tuple (Q, Σ, δ, q0, F) where:
  δ : Q × (Σ ∪ {ε}) → 2^Q  (transitions to *sets* of states)

Subset construction (Rabin-Scott powerset construction):
  Each DFA state corresponds to a *subset* of NFA states.
  The DFA start state = ε-closure({q0_NFA}).
  δ_DFA(S, a) = ε-closure(⋃_{q ∈ S} δ_NFA(q, a))
"""

from __future__ import annotations
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from automata.dfa import DFA

EPSILON = "eps"  # canonical key for ε-transitions


class NFA:
    """
    Non-deterministic Finite Automaton with optional ε-transitions.

    Parameters
    ----------
    states   : list of state names
    alphabet : list of input symbols (must NOT include 'eps')
    start    : the start state name
    finals   : list of accepting state names
    delta    : {state: {symbol_or_"eps": [list of target states]}}
    """

    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        start: str,
        finals: List[str],
        delta: Dict[str, Dict[str, List[str]]],
    ) -> None:
        self.states: List[str] = list(states)
        self.alphabet: List[str] = [a for a in alphabet if a != EPSILON]
        self.start: str = start
        self.finals: Set[str] = set(finals)
        # delta[state][symbol] = list of target states
        self.delta: Dict[str, Dict[str, List[str]]] = {
            s: {sym: list(tgts) for sym, tgts in row.items()}
            for s, row in delta.items()
        }

    # ------------------------------------------------------------------
    # ε-closure
    # ------------------------------------------------------------------

    def epsilon_closure(self, states: Set[str]) -> FrozenSet[str]:
        """
        Compute ε-closure(T) = the set of NFA states reachable from any
        state in T using zero or more ε-transitions.

        Algorithm: BFS/DFS following only ε-edges.

        Textbook definition:
          ε-closure(q) = {q} ∪ ⋃_{r ∈ δ(q,ε)} ε-closure(r)
        """
        closure: Set[str] = set(states)
        stack = list(states)

        while stack:
            q = stack.pop()
            for r in self.delta.get(q, {}).get(EPSILON, []):
                if r not in closure:
                    closure.add(r)
                    stack.append(r)

        return frozenset(closure)

    # ------------------------------------------------------------------
    # Subset construction  (NFA → DFA)
    # ------------------------------------------------------------------

    def to_dfa(self) -> Tuple[DFA, List[Dict]]:
        """
        Convert this NFA to an equivalent DFA using the subset construction
        (Rabin-Scott powerset construction).

        Returns
        -------
        dfa   : the equivalent DFA
        table : list of rows, each a dict:
                  {"dfa_state": str, symbol1: str, symbol2: str, ...}
                used by the UI to display the subset-construction table.

        Algorithm
        ---------
        1. Start DFA state = ε-closure({q0_NFA})
        2. For each unmarked DFA state S and each symbol a:
             T = ε-closure(⋃_{q ∈ S} δ_NFA(q, a))
             Add δ_DFA(S, a) = T
             If T is new, add to worklist.
        3. DFA final states = all S that contain at least one NFA final state.
        """

        def subset_name(fs: FrozenSet[str]) -> str:
            """Canonical name for a subset, e.g. '{q0,q1}' or '{}'."""
            if not fs:
                return "{}"
            return "{" + ",".join(sorted(fs)) + "}"

        start_closure = self.epsilon_closure({self.start})
        start_name = subset_name(start_closure)

        # Worklist: frozensets to process
        worklist: deque[FrozenSet[str]] = deque([start_closure])
        visited: Dict[FrozenSet[str], str] = {start_closure: start_name}

        dfa_delta: Dict[str, Dict[str, str]] = {}
        table_rows: List[Dict] = []

        while worklist:
            current_set = worklist.popleft()
            current_name = visited[current_set]
            dfa_delta[current_name] = {}

            row: Dict = {"dfa_state": current_name}

            for a in self.alphabet:
                # Compute ⋃_{q ∈ current_set} δ_NFA(q, a)
                reachable: Set[str] = set()
                for q in current_set:
                    reachable.update(self.delta.get(q, {}).get(a, []))

                next_closure = self.epsilon_closure(reachable)
                next_name = subset_name(next_closure)

                # Register new subset if unseen
                if next_closure not in visited:
                    visited[next_closure] = next_name
                    worklist.append(next_closure)

                dfa_delta[current_name][a] = next_name
                row[a] = next_name

            table_rows.append(row)

        # DFA states: all subset names
        dfa_states = list(visited.values())

        # DFA finals: subsets that contain at least one NFA final state
        dfa_finals = [
            name
            for fs, name in visited.items()
            if fs & self.finals  # non-empty intersection
        ]

        dfa = DFA(
            states=dfa_states,
            alphabet=self.alphabet,
            start=start_name,
            finals=dfa_finals,
            delta=dfa_delta,
        )
        return dfa, table_rows

    def __repr__(self) -> str:
        return (
            f"NFA(states={self.states}, alphabet={self.alphabet}, "
            f"start={self.start!r}, finals={self.finals})"
        )
