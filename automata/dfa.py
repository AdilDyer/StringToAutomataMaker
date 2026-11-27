"""
automata/dfa.py
===============
DFA class implementing:
  - Validation (textbook definition checks)
  - Completion with a dead/trap state
  - Simulation with step-by-step trace
  - Reachability (remove unreachable states)

A DFA is a 5-tuple (Q, Σ, δ, q0, F) where:
  Q  = finite set of states
  Σ  = finite input alphabet
  δ  = Q × Σ → Q  (total transition function)
  q0 = start state ∈ Q
  F  = set of final/accepting states ⊆ Q
"""

from __future__ import annotations
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


class DFAValidationError(Exception):
    """Raised when a DFA definition violates the formal constraints."""
    pass


class DFA:
    """
    Deterministic Finite Automaton.

    Parameters
    ----------
    states    : list of state names (strings)
    alphabet  : list of input symbols (single-character strings)
    start     : the start state name
    finals    : list of accepting state names
    delta     : dict  {state: {symbol: target_state}}
                (transitions may be partial; call complete() to make them total)
    """

    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        start: str,
        finals: List[str],
        delta: Dict[str, Dict[str, str]],
    ) -> None:
        self.states: List[str] = list(states)
        self.alphabet: List[str] = list(alphabet)
        self.start: str = start
        self.finals: Set[str] = set(finals)
        # delta[state][symbol] = target_state
        self.delta: Dict[str, Dict[str, str]] = {
            s: dict(row) for s, row in delta.items()
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Check the DFA against the formal definition and return a list of
        human-readable error strings.  An empty list means the DFA is valid.

        Checks:
          1. Start state must be in Q.
          2. Every final state must be in Q.
          3. Every source state in δ must be in Q.
          4. Every symbol in δ rows must be in Σ.
          5. Every target state in δ must be in Q.
          6. Report missing transitions (not an error, but noted).
        """
        errors: List[str] = []
        state_set = set(self.states)
        alpha_set = set(self.alphabet)

        # Check 1: start state
        if self.start not in state_set:
            errors.append(
                f"Start state '{self.start}' is not in the states list {self.states}."
            )

        # Check 2: final states
        for f in self.finals:
            if f not in state_set:
                errors.append(
                    f"Final state '{f}' is not in the states list {self.states}."
                )

        # Checks 3-5: transitions
        for src, row in self.delta.items():
            if src not in state_set:
                errors.append(
                    f"Transition source '{src}' is not in the states list."
                )
            for sym, tgt in row.items():
                if sym not in alpha_set:
                    errors.append(
                        f"Symbol '{sym}' in δ({src}, ·) is not in alphabet {self.alphabet}."
                    )
                if tgt not in state_set:
                    errors.append(
                        f"Transition target '{tgt}' in δ({src},{sym}) is not in states."
                    )

        return errors

    def missing_transitions(self) -> List[Tuple[str, str]]:
        """Return (state, symbol) pairs that have no defined transition."""
        missing = []
        for s in self.states:
            for a in self.alphabet:
                if a not in self.delta.get(s, {}):
                    missing.append((s, a))
        return missing

    # ------------------------------------------------------------------
    # Completion (add dead/trap state)
    # ------------------------------------------------------------------

    def complete(self) -> bool:
        """
        Make the DFA total by adding a 'dead' trap state for every undefined
        transition.  The dead state has self-loops on every symbol.

        Returns True if a dead state was added, False if the DFA was already
        complete.

        Textbook: a complete DFA requires δ to be a total function Q×Σ→Q.
        The dead state is a non-accepting state that absorbs all missing
        transitions.
        """
        missing = self.missing_transitions()
        if not missing:
            return False  # already complete

        DEAD = "dead"
        # Ensure the name does not clash with existing states
        while DEAD in self.states:
            DEAD = "_" + DEAD

        self.states.append(DEAD)
        self.delta.setdefault(DEAD, {})

        # Route all missing transitions to dead
        for (s, a) in missing:
            self.delta.setdefault(s, {})[a] = DEAD

        # dead state: self-loop on all symbols
        for a in self.alphabet:
            self.delta[DEAD][a] = DEAD

        return True

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(
        self, string: str
    ) -> Tuple[bool, List[Tuple[str, str, str]], Optional[str]]:
        """
        Simulate the DFA on *string*.

        Returns
        -------
        accepted : bool
        trace    : list of (state_before, symbol, state_after)
        error    : str | None — set if the string contains a symbol not in Σ

        Handles the empty string: trace is [], accepted = (start in finals).
        """
        current = self.start
        trace: List[Tuple[str, str, str]] = []

        for ch in string:
            if ch not in self.alphabet:
                return False, trace, f"Symbol '{ch}' is not in the alphabet {self.alphabet}."
            nxt = self.delta.get(current, {}).get(ch)
            if nxt is None:
                # Incomplete DFA — transition undefined; treat as rejection
                trace.append((current, ch, "undefined"))
                return False, trace, None
            trace.append((current, ch, nxt))
            current = nxt

        accepted = current in self.finals
        return accepted, trace, None

    def accepts(self, string: str) -> bool:
        """Return True iff the DFA accepts *string*."""
        ok, _, _ = self.run(string)
        return ok

    # ------------------------------------------------------------------
    # Reachability
    # ------------------------------------------------------------------

    def remove_unreachable(self) -> List[str]:
        """
        Remove all states not reachable from the start state via BFS/BFS.

        Returns the list of removed state names.

        Textbook: a state q is reachable if there exists a string w such that
        δ*(q0, w) = q.
        """
        reachable: Set[str] = set()
        queue: deque[str] = deque([self.start])
        reachable.add(self.start)

        while queue:
            s = queue.popleft()
            for a in self.alphabet:
                tgt = self.delta.get(s, {}).get(a)
                if tgt is not None and tgt not in reachable:
                    reachable.add(tgt)
                    queue.append(tgt)

        removed = [s for s in self.states if s not in reachable]

        self.states = [s for s in self.states if s in reachable]
        self.finals = self.finals & reachable
        self.delta = {
            s: {a: t for a, t in row.items() if t in reachable}
            for s, row in self.delta.items()
            if s in reachable
        }
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the DFA to the canonical JSON dict format."""
        return {
            "machine_type": "DFA",
            "states": self.states,
            "alphabet": self.alphabet,
            "start": self.start,
            "finals": sorted(self.finals),
            "transitions": {s: {a: [t] for a, t in row.items()} for s, row in self.delta.items()},
        }

    def __repr__(self) -> str:
        return (
            f"DFA(states={self.states}, alphabet={self.alphabet}, "
            f"start={self.start!r}, finals={self.finals})"
        )
