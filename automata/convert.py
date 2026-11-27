"""
automata/convert.py
===================
Exact, from-scratch conversions from a DFA to:
  1. Right-Linear Regular Grammar: G = (V, T, P, S)
  2. Regular Expression via State Elimination (Brzozowski-McCluskey / GNFA)
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import re as _re
from typing import Dict, List, Set, Tuple, Union

from automata.dfa import DFA


def _co_accessible_states(dfa: DFA) -> Set[str]:
    """Return states that can reach at least one final state (reverse BFS)."""
    rev: Dict[str, Set[str]] = {s: set() for s in dfa.states}
    for src, row in dfa.delta.items():
        for tgt in row.values():
            if tgt in rev:
                rev[tgt].add(src)
    live: Set[str] = set(dfa.finals & set(dfa.states))
    queue: deque[str] = deque(live)
    while queue:
        cur = queue.popleft()
        for prev in rev.get(cur, ()):
            if prev not in live:
                live.add(prev)
                queue.append(prev)
    return live


def _assign_non_terminals(dfa: DFA, live_states: Set[str]) -> Dict[str, str]:
    """Map each live DFA state to a CFG non-terminal disjoint from alphabet."""
    ordered = [dfa.start] + [s for s in dfa.states if s in live_states and s != dfa.start]
    alpha_set = set(dfa.alphabet)
    candidates = ["S"] + [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) != "S"]
    safe_pool = [c for c in candidates if c not in alpha_set]

    mapping: Dict[str, str] = {}
    for idx, state in enumerate(ordered):
        if idx < len(safe_pool):
            mapping[state] = safe_pool[idx]
        else:
            nt = f"Q{idx}"
            while nt in alpha_set:
                nt = f"_{nt}"
            mapping[state] = nt
    return mapping


def dfa_to_grammar(dfa: DFA) -> dict:
    """
    Convert a DFA into an equivalent right-linear regular grammar G = (V, T, P, S):
      - δ(p, a) = q (live) -> p -> a q
      - f ∈ F              -> f -> ε
    """
    live = _co_accessible_states(dfa)
    if dfa.start not in live:
        return {
            "variables": ["S"], "terminals": list(dfa.alphabet), "start": "S",
            "productions": {}, "state_map": {},
            "text": "S -> ∅  (empty language — no accepting state is reachable)",
        }

    state_map = _assign_non_terminals(dfa, live)
    ordered_states = list(state_map.keys())
    productions: Dict[str, List[str]] = {}
    lines: List[str] = []

    for state in ordered_states:
        lhs = state_map[state]
        rhs_list: List[str] = []
        for sym in dfa.alphabet:
            tgt = dfa.delta.get(state, {}).get(sym)
            if tgt is not None and tgt in live:
                rhs_list.append(f"{sym} {state_map[tgt]}")
        if state in dfa.finals:
            rhs_list.append("ε")
        if rhs_list:
            productions[lhs] = rhs_list
            lines.append(f"{lhs} -> {' | '.join(rhs_list)}")

    return {
        "variables": [state_map[s] for s in ordered_states],
        "terminals": list(dfa.alphabet),
        "start": state_map[dfa.start],
        "productions": productions,
        "state_map": state_map,
        "text": "\n".join(lines),
    }


@dataclass(frozen=True)
class _Empty: pass

@dataclass(frozen=True)
class _Eps: pass

@dataclass(frozen=True)
class _Lit:
    ch: str

@dataclass(frozen=True)
class _Star:
    sub: "_Rx"

@dataclass(frozen=True)
class _Seq:
    parts: Tuple["_Rx", ...]

@dataclass(frozen=True)
class _Alt:
    branches: Tuple["_Rx", ...]


_Rx = Union[_Empty, _Eps, _Lit, _Star, _Seq, _Alt]
_EMPTY, _EPS = _Empty(), _Eps()


def _rx_alt(a: _Rx, b: _Rx) -> _Rx:
    """Construct simplified alternation (a | b)."""
    if a == _EMPTY:
        return b
    if b == _EMPTY or a == b:
        return a
    raw: List[_Rx] = []
    for node in (a, b):
        raw.extend(node.branches if isinstance(node, _Alt) else (node,))
    unique: List[_Rx] = []
    for item in raw:
        if item != _EMPTY and item not in unique:
            unique.append(item)
    if _EPS in unique and any(isinstance(x, _Star) for x in unique):
        unique = [x for x in unique if x != _EPS]
    if not unique:
        return _EMPTY
    return unique[0] if len(unique) == 1 else _Alt(tuple(unique))


def _rx_seq(a: _Rx, b: _Rx) -> _Rx:
    """Construct simplified concatenation (a · b)."""
    if a == _EMPTY or b == _EMPTY:
        return _EMPTY
    if a == _EPS:
        return b
    if b == _EPS:
        return a
    parts: List[_Rx] = []
    for node in (a, b):
        parts.extend(node.parts if isinstance(node, _Seq) else (node,))
    return parts[0] if len(parts) == 1 else _Seq(tuple(parts))


def _rx_star(a: _Rx) -> _Rx:
    """Construct simplified Kleene star (a*)."""
    if a == _EMPTY or a == _EPS:
        return _EPS
    if isinstance(a, _Star):
        return a
    if isinstance(a, _Alt):
        non_eps = [b for b in a.branches if b != _EPS]
        if len(non_eps) < len(a.branches):
            return _rx_star(non_eps[0] if len(non_eps) == 1 else _Alt(tuple(non_eps)))
    return _Star(a)


def _format_rx(node: _Rx, python_mode: bool = False) -> str:
    """Format a regex AST node into TOC notation or Python `re` syntax."""
    if isinstance(node, _Empty):
        return "(?!)" if python_mode else "∅"
    if isinstance(node, _Eps):
        return "" if python_mode else "ε"
    if isinstance(node, _Lit):
        return _re.escape(node.ch) if python_mode else node.ch
    if isinstance(node, _Star):
        s = _format_rx(node.sub, python_mode)
        if isinstance(node.sub, _Lit):
            return f"{s}*"
        return f"(?:{s})*" if python_mode else f"({s})*"
    if isinstance(node, _Seq):
        chunks = []
        for p in node.parts:
            s = _format_rx(p, python_mode)
            chunks.append((f"(?:{s})" if python_mode else f"({s})") if isinstance(p, _Alt) else s)
        return "".join(chunks)
    return "|".join(_format_rx(b, python_mode) for b in node.branches)


def dfa_to_regex(dfa: DFA) -> dict:
    """
    Convert a DFA to an equivalent Regular Expression using State Elimination:
      R[i][j] <- R[i][j] | R[i][k] · (R[k][k])* · R[k][j]
    """
    live = _co_accessible_states(dfa)
    if dfa.start not in live:
        return {"regex": "∅", "python_regex": "(?!)", "elimination_order": [], "too_large": False}

    live_list = [s for s in dfa.states if s in live]
    if len(live_list) > 15:
        return {
            "regex": f"(DFA has {len(live_list)} live states; state elimination skipped to avoid blowup)",
            "python_regex": None, "elimination_order": [], "too_large": True,
        }

    S_NEW, F_NEW = "__GNFA_START__", "__GNFA_FINAL__"
    edges: Dict[Tuple[str, str], _Rx] = {}

    def add_edge(u: str, v: str, rx: _Rx) -> None:
        if rx != _EMPTY:
            merged = _rx_alt(edges.get((u, v), _EMPTY), rx)
            if merged != _EMPTY:
                edges[(u, v)] = merged

    add_edge(S_NEW, dfa.start, _EPS)
    for f in dfa.finals:
        if f in live:
            add_edge(f, F_NEW, _EPS)
    for src in live_list:
        for sym in dfa.alphabet:
            tgt = dfa.delta.get(src, {}).get(sym)
            if tgt is not None and tgt in live:
                add_edge(src, tgt, _Lit(sym))

    remaining = set(live_list)
    elim_order: List[str] = []
    while remaining:
        def score(k: str) -> Tuple[int, int, str]:
            inc = sum(1 for (u, v) in edges if v == k and u != k)
            out = sum(1 for (u, v) in edges if u == k and v != k)
            return (inc * out, inc + out, k)

        k = min(remaining, key=score)
        remaining.remove(k)
        elim_order.append(k)
        loop = _rx_star(edges.get((k, k), _EMPTY))
        preds = [(u, rx) for (u, v), rx in list(edges.items()) if v == k and u != k]
        succs = [(v, rx) for (u, v), rx in list(edges.items()) if u == k and v != k]
        for u, rx_uk in preds:
            for v, rx_kv in succs:
                add_edge(u, v, _rx_seq(_rx_seq(rx_uk, loop), rx_kv))
        for e in [e for e in edges if e[0] == k or e[1] == k]:
            del edges[e]

    final_rx = edges.get((S_NEW, F_NEW), _EMPTY)
    return {
        "regex": _format_rx(final_rx, python_mode=False),
        "python_regex": _format_rx(final_rx, python_mode=True),
        "elimination_order": elim_order,
        "too_large": False,
    }
