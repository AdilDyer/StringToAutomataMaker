"""
automata/draw.py
================
Generate Graphviz DOT strings for DFA visualisation.

Rendering strategy: return a raw DOT string and let the caller pass it to
st.graphviz_chart().  This avoids needing a system Graphviz binary.

Styling conventions:
  - rankdir=LR  (left-to-right layout)
  - Invisible "init" node with arrow into start state
  - Final states: doublecircle shape
  - Dead/trap state: dashed grey
  - Parallel edges (same source + target, different symbols) are merged into
    one edge label: "a,b"
  - Trace highlighting: visited states/edges in orange; final state green
    (accepted) or red (rejected); edge labels include step number
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from automata.dfa import DFA


def _escape(s: str) -> str:
    """Escape a state name for use in a DOT node identifier."""
    return '"' + s.replace('"', '\\"') + '"'


def generate_dot(
    dfa: DFA,
    title: str = "",
    trace: Optional[List[Tuple[str, str, str]]] = None,
    accepted: Optional[bool] = None,
) -> str:
    """
    Generate a Graphviz DOT string for *dfa*.

    Parameters
    ----------
    dfa      : the DFA to draw
    title    : optional graph label
    trace    : list of (state_before, symbol, state_after) from dfa.run()
    accepted : True/False — determines the colour of the final trace state

    Returns
    -------
    A DOT string suitable for st.graphviz_chart() or graphviz.Source().
    """

    # --- Collect trace information ------------------------------------------
    # visited_states: set of states that appear in the trace
    # edge_steps: {(src, tgt): [(step_num, symbol), ...]}
    visited_states: set = set()
    edge_steps: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    final_trace_state: Optional[str] = None

    if trace is not None:
        visited_states.add(dfa.start)
        for step_idx, (src, sym, tgt) in enumerate(trace):
            visited_states.add(src)
            visited_states.add(tgt)
            edge_steps[(src, tgt)].append((step_idx + 1, sym))
        final_trace_state = trace[-1][2] if trace else dfa.start

    # --- Merge parallel edges -----------------------------------------------
    # normal_edges: {(src, tgt): [symbol, ...]}
    normal_edges: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for src, row in dfa.delta.items():
        for sym, tgt in row.items():
            normal_edges[(src, tgt)].append(sym)

    # --- Build DOT -----------------------------------------------------------
    lines: List[str] = []
    lines.append("digraph DFA {")
    lines.append('  rankdir=LR;')
    lines.append('  node [fontname="Helvetica" fontsize=12];')
    lines.append('  edge [fontname="Helvetica" fontsize=11];')

    if title:
        escaped_title = title.replace('"', '\\"')
        lines.append(f'  label="{escaped_title}";')
        lines.append('  labelloc="t";')

    # Invisible init node
    lines.append('  __init__ [shape=point style=invis];')

    # State nodes
    for s in dfa.states:
        is_dead = (s == "dead" or s.startswith("_dead") or s == "{}")
        is_final = s in dfa.finals
        shape = "doublecircle" if is_final else "circle"

        attrs = [f'shape={shape}']

        if trace is not None and s in visited_states:
            if s == final_trace_state and accepted is not None:
                border = "darkgreen" if accepted else "firebrick"
                fill = "#c8e6c9" if accepted else "#ffcdd2"
                style = "filled,dashed" if is_dead else "filled"
                attrs += [f'style="{style}"', f'color={border}', f'fillcolor="{fill}"', 'penwidth=2']
            else:
                style = "filled,dashed" if is_dead else "filled"
                attrs += [f'style="{style}"', 'fillcolor="#ffe0b2"', 'color=darkorange', 'penwidth=2']
        elif is_dead:
            attrs += ['style="dashed"', 'color=grey', 'fontcolor=grey']

        label = s.replace('"', '\\"')
        attr_str = " ".join(attrs) + f' label="{label}"'
        lines.append(f"  {_escape(s)} [{attr_str}];")

    # Arrow into start state
    start_color = ""
    if trace is not None:
        start_color = ' color=darkorange penwidth=2'
    lines.append(f'  __init__ -> {_escape(dfa.start)} [arrowhead=vee{start_color}];')

    # Edges
    for (src, tgt), symbols in normal_edges.items():
        sorted_symbols = sorted(symbols)
        is_dead_edge = (tgt == "dead" or tgt.startswith("_dead") or tgt == "{}")
        is_dead_src = (src == "dead" or src.startswith("_dead") or src == "{}")

        if trace and (src, tgt) in edge_steps:
            # Highlighted edge: show step numbers
            steps_info = edge_steps[(src, tgt)]
            step_nums = ",".join(str(n) for n, _ in steps_info)
            syms_shown = ",".join(sym for _, sym in steps_info)
            # Also append any non-traced symbols
            traced_syms = {sym for _, sym in steps_info}
            extra = [s for s in sorted_symbols if s not in traced_syms]
            extra_part = (",".join(extra)) if extra else ""
            full_label = f"{step_nums}: {syms_shown}"
            if extra_part:
                full_label += f"\\n({extra_part})"
            escaped_lbl = full_label.replace('"', '\\"')
            lines.append(
                f"  {_escape(src)} -> {_escape(tgt)} "
                f'[label="{escaped_lbl}" color=darkorange penwidth=2 fontcolor=darkorange];'
            )
        else:
            label = ",".join(sorted_symbols)
            escaped_lbl = label.replace('"', '\\"')
            if is_dead_edge or is_dead_src:
                lines.append(
                    f"  {_escape(src)} -> {_escape(tgt)} "
                    f'[label="{escaped_lbl}" color=grey fontcolor=grey style=dashed];'
                )
            else:
                lines.append(
                    f"  {_escape(src)} -> {_escape(tgt)} "
                    f'[label="{escaped_lbl}"];'
                )

    lines.append("}")
    return "\n".join(lines)
