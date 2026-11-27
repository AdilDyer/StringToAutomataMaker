"""
ui/tabs.py
==========
Render each of the five result tabs in the AutomataGPT Streamlit UI.
Called from app.py after a successful pipeline run.
"""

from __future__ import annotations

import streamlit as st

from automata.dfa import DFA
from automata.draw import generate_dot


# ---------------------------------------------------------------------------
# Tab 1: Result (DFA + Regular Grammar + Regular Expression)
# ---------------------------------------------------------------------------

def _render_grammar_and_regex(result: dict) -> None:
    """Display the equivalent Regular Expression and Right-Linear Grammar."""
    regex_info = result.get("regex")
    grammar_info = result.get("grammar")
    if not regex_info and not grammar_info:
        return

    st.divider()
    col_rx, col_gr = st.columns(2)

    with col_rx:
        st.subheader("🔤 Regular Expression (Regex)")
        st.caption("Derived from the minimized DFA via State Elimination.")
        if regex_info:
            st.code(regex_info["regex"], language="text")
            py_rx = regex_info.get("python_regex")
            if py_rx is not None and py_rx != regex_info["regex"]:
                st.caption(f"Python `re` pattern: `{py_rx}`")
            order = regex_info.get("elimination_order", [])
            if order:
                st.caption(f"State elimination order: `{' → '.join(order)}`")

    with col_gr:
        st.subheader("📝 Regular Grammar (Right-Linear CFG)")
        st.caption("Derived directly from the minimized DFA (G = (V, T, P, S)).")
        if grammar_info:
            st.code(grammar_info["text"], language="text")
            smap = grammar_info.get("state_map", {})
            if smap:
                legend = ", ".join(f"{nt} = {st_name}" for st_name, nt in smap.items())
                st.caption(f"Start symbol: `{grammar_info['start']}` · State mapping: `{legend}`")
            st.download_button(
                "⬇️ Download Grammar (.txt)",
                grammar_info["text"],
                "regular_grammar.txt",
                "text/plain",
            )


def render_result_tab(result: dict) -> None:
    """Render the main Result tab showing DFA, Grammar, and Regex."""
    original_dfa = result.get("original_dfa")
    min_dfa = result.get("min_dfa")
    machine_json = result.get("machine_json", {})
    min_info = result.get("min_info", {})
    verify_result = result.get("verify_result")

    n_orig = len(original_dfa.states) if original_dfa else "?"
    n_min = len(min_dfa.states)
    if verify_result:
        n_checked = verify_result["total_checked"]
        n_mm = verify_result["all_mismatch_count"]
        verify_badge = f"✅ {n_checked:,} strings" if n_mm == 0 else f"❌ {n_mm} mismatches"
    else:
        verify_badge = "Not verified (no oracle provided)"

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("States (original DFA)", n_orig)
    mc2.metric(
        "States (minimized DFA)", n_min,
        delta=f"{n_min - n_orig}" if isinstance(n_orig, int) else None,
    )
    mc3.metric("Verification", verify_badge)

    if min_info.get("dead_added"):
        st.info(
            "ℹ️ **Dead state included**: the minimized DFA is *complete* — a trap/dead state "
            "was added for undefined transitions (standard GATE convention)."
        )

    explanation = machine_json.get("explanation", "")
    if explanation:
        st.subheader("💡 Explanation")
        st.write(explanation)

    st.subheader("📐 1. DFA Diagrams")
    orig_dot = generate_dot(original_dfa, title="Original DFA") if original_dfa else ""
    min_dot = generate_dot(min_dfa, title="Minimized DFA")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.caption("**Original DFA**")
        if orig_dot:
            st.graphviz_chart(orig_dot)
        st.download_button("⬇️ Download Original DOT", orig_dot, "original_dfa.dot", "text/plain")
    with dcol2:
        st.caption("**Minimized DFA**")
        st.graphviz_chart(min_dot)
        st.download_button("⬇️ Download Minimized DOT", min_dot, "minimized_dfa.dot", "text/plain")

    st.subheader("📋 Transition Table (Minimized DFA)")
    rows = []
    for s in min_dfa.states:
        label = ("→" if s == min_dfa.start else "") + ("*" if s in min_dfa.finals else "")
        row = {"State": f"{label}{s}"}
        for a in min_dfa.alphabet:
            row[a] = min_dfa.delta.get(s, {}).get(a, "—")
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("→ = start state, * = accepting state")

    # Show 2. Regex & 3. Regular Grammar right on the main Result output
    _render_grammar_and_regex(result)


# ---------------------------------------------------------------------------
# Tab 2: Minimization Steps
# ---------------------------------------------------------------------------

def render_minimization_tab(result: dict) -> None:
    """Render the Minimization Steps tab."""
    min_info = result.get("min_info")
    if not min_info:
        st.info("No minimization info available.")
        return

    removed = min_info.get("removed_unreachable", [])
    if removed:
        st.warning(f"Removed unreachable states: {removed}")
    else:
        st.success("All states are reachable from the start state.")

    if min_info.get("dead_added"):
        st.info("A dead/trap state was added to make the DFA complete before minimization.")

    st.subheader("🔗 Equivalence Classes")
    for i, cls in enumerate(sorted(min_info.get("classes", []), key=lambda c: sorted(c)[0])):
        members = sorted(cls)
        merged = members[0] if len(members) == 1 else "{" + ",".join(members) + "}"
        st.write(f"**Class {i+1}**: {members} → merged as `{merged}`")

    st.subheader("📝 Marking Log (Table-Filling Algorithm)")
    for entry in min_info.get("round_log", []):
        label = "Base case" if entry["round"] == 0 else f"Round {entry['round']}"
        pairs_str = ", ".join(f"({p},{q})" for p, q in entry["pairs"])
        with st.expander(f"{label}: {len(entry['pairs'])} pair(s) marked"):
            st.write(entry["reason"])
            st.write(f"Pairs marked: {pairs_str}")

    st.subheader("🔢 Distinguishability Table")
    all_pairs = min_info.get("all_pairs", [])
    states_shown = min_info.get("states_after_completion", [])
    table = min_info.get("table", {})

    if all_pairs and states_shown:
        import pandas as pd
        matrix = {s: {t: "" for t in states_shown} for s in states_shown}
        for pair_key, round_marked in table.items():
            p_str, q_str = sorted(pair_key)
            val = "≡" if round_marked is None else f"X{round_marked}"
            matrix[p_str][q_str] = val
            matrix[q_str][p_str] = val
        df = pd.DataFrame(matrix).reindex(index=states_shown, columns=states_shown)
        st.dataframe(df, use_container_width=True)
        st.caption("X0 = base case. Xn = marked round n. ≡ = equivalent.")


# ---------------------------------------------------------------------------
# Tab 3: NFA → DFA
# ---------------------------------------------------------------------------

def render_nfa_tab(result: dict) -> None:
    """Render the NFA → DFA subset construction tab."""
    nfa_table = result.get("nfa_table")
    if nfa_table is None:
        st.info("The LLM returned a **DFA** directly — no NFA-to-DFA conversion was needed.")
        return

    st.subheader("🔄 Subset Construction Table")
    st.caption("Each row is a DFA state (a subset of NFA states).")
    import pandas as pd
    df_nfa = pd.DataFrame(nfa_table).rename(columns={"dfa_state": "DFA State"})
    st.dataframe(df_nfa, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: Test a String
# ---------------------------------------------------------------------------

def render_test_tab(result: dict) -> None:
    """Render the Test a String tab."""
    min_dfa = result.get("min_dfa")
    original_dfa = result.get("original_dfa")

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        test_string = st.text_input(
            "Enter a string to test", value="",
            placeholder="e.g. ababba (or leave empty for ε)",
            key="test_string_input",
        )
    with tc2:
        use_min = st.checkbox("Use minimized DFA", value=True, key="test_use_min_checkbox")

    active_dfa = min_dfa if use_min else (original_dfa or min_dfa)
    active_copy = DFA(
        states=list(active_dfa.states), alphabet=list(active_dfa.alphabet),
        start=active_dfa.start, finals=set(active_dfa.finals),
        delta={s: dict(r) for s, r in active_dfa.delta.items()},
    )
    active_copy.complete()
    accepted, trace, err = active_copy.run(test_string)

    if err:
        st.error(f"⚠️ {err}")
        return

    color = "green" if accepted else "red"
    label = "✅ ACCEPTED" if accepted else "❌ REJECTED"
    st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
    if test_string == "":
        st.info("Testing the empty string ε.")
    if trace:
        st.subheader("Step-by-step trace")
        st.dataframe(
            [{"Step": i + 1, "State before": s, "Symbol": a, "State after": t}
             for i, (s, a, t) in enumerate(trace)],
            use_container_width=True, hide_index=True,
        )
    trace_dot = generate_dot(
        active_copy,
        title="Test: " + (repr(test_string) if test_string else "ε"),
        trace=trace, accepted=accepted,
    )
    st.subheader("Highlighted DFA")
    st.graphviz_chart(trace_dot)


# ---------------------------------------------------------------------------
# Tab 5: Verification
# ---------------------------------------------------------------------------

def render_verification_tab(result: dict, attempts_log: list) -> None:
    """Render the Verification tab."""
    verify_result = result.get("verify_result")
    cv_result = result.get("cross_verify_result")
    oracle_code = result.get("oracle_code")

    if verify_result:
        total = verify_result["total_checked"]
        n_mm = verify_result["all_mismatch_count"]
        max_len_used = verify_result["max_len_used"]
        if verify_result.get("capped"):
            st.warning(f"⚠️ Max length auto-reduced to {max_len_used} to stay under safety cap.")
        if verify_result.get("timed_out"):
            st.warning("⚠️ Verification timed out (20 s limit).")
        if n_mm == 0:
            st.success(f"✅ Verified on {total:,} strings (length 0–{max_len_used}). No mismatches.")
        else:
            st.error(f"❌ Found {n_mm} mismatch(es) on {total:,} strings.")
            import pandas as pd
            st.dataframe(pd.DataFrame(verify_result["mismatches"]), use_container_width=True, hide_index=True)
        if verify_result.get("shortest_counterexample") is not None:
            st.write(f"**Shortest counterexample**: `{verify_result['shortest_counterexample']!r}`")
    else:
        st.info("Not verified (no oracle provided).")

    if cv_result:
        st.subheader("🔁 Original ↔ Minimized Consistency")
        if cv_result["agree"]:
            st.success(f"✅ Both DFAs agree on all {cv_result['total_checked']:,} tested strings.")
        else:
            st.error(f"❌ Found {cv_result['all_mismatch_count']} disagreements (bug in minimizer).")

    if attempts_log:
        st.subheader("📋 Attempt History")
        import pandas as pd
        st.dataframe(pd.DataFrame(attempts_log), use_container_width=True, hide_index=True)

    if oracle_code:
        with st.expander("🔍 AI-written oracle checker (shown for transparency)"):
            st.code(oracle_code, language="python")
            st.caption("AST-vetted and runs in a restricted sandbox with no filesystem/network access.")
