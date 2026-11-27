"""
app.py
======
AutomataGPT — Streamlit UI entry point.
Supports 3 modes: Gemini Cloud AI, Offline Local LLM (Ollama), and Manual JSON.
"""

from __future__ import annotations
import inspect
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
sys.path.insert(0, os.path.dirname(__file__))

from llm.client import generate_machine
from llm.sandbox import compile_oracle, OracleSecurityError
from ui.pipeline import (
    build_counterexample_feedback,
    parse_alphabet,
    prepare_manual_machine,
    run_pipeline,
    synthesize_from_oracle,
)
from ui.sidebar import render_sidebar
from ui.tabs import (
    render_result_tab, render_minimization_tab,
    render_nfa_tab, render_test_tab, render_verification_tab,
)
from examples import get_example_names, get_example

st.set_page_config(page_title="AutomataGPT", page_icon="🤖", layout="wide")

_DEFAULTS = {
    "result": None, "attempts_log": [], "last_example": "— choose —",
    "desc_input": "", "alpha_input": "a,b",
    "manual_json_input": "", "manual_oracle_input": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


@st.cache_data(show_spinner=False)
def _cached_initial_generate(
    backend: str, desc: str, alpha_tuple: tuple[str, ...], model: str
) -> dict:
    """Cache initial machine generation by (backend, description, alphabet, model)."""
    mj, _ = generate_machine(
        desc, list(alpha_tuple), model, max_retries=1, feedback=None, backend=backend
    )
    return mj


mode, active_model, max_retries, max_verify_len, api_key, use_lstar_fallback = render_sidebar()
manual_mode = "Manual" in mode
offline_mode = "Offline" in mode
backend = "ollama" if offline_mode else "gemini"

# ---------------------------------------------------------------------------
# Main page inputs
# ---------------------------------------------------------------------------

st.title("🤖 AutomataGPT: English → DFA Designer")
st.caption("Describe a formal language in plain English and get a verified, minimized DFA, Grammar & Regex.")

example_names = get_example_names()
example_choice = st.selectbox("📚 Load a built-in example", ["— choose —"] + example_names)
if example_choice != st.session_state["last_example"]:
    st.session_state["last_example"] = example_choice
    if example_choice != "— choose —":
        ex = get_example(example_names.index(example_choice))
        mj = ex["machine_json"]
        st.session_state["desc_input"] = ex["description"]
        st.session_state["alpha_input"] = ",".join(ex["alphabet"])
        st.session_state["manual_json_input"] = json.dumps(mj, indent=2)
        st.session_state["manual_oracle_input"] = mj.get("oracle_body", "")

col1, col2 = st.columns([3, 1])
with col1:
    description = st.text_area(
        "Language description", height=100, key="desc_input",
        placeholder="e.g. 'Binary strings that contain at least two 1s'",
    )
with col2:
    alphabet_text = st.text_input(
        "Alphabet (comma-separated, 1-4 symbols)", key="alpha_input", placeholder="a,b",
    )
    st.caption("Single characters, e.g. `a,b` or `0,1`")

manual_json_text, manual_oracle_text = "", ""
if manual_mode:
    st.divider()
    st.subheader("📋 Manual JSON Input")
    st.caption("Uses the `alphabet` defined inside your JSON. `oracle_body` and `explanation` are optional.")
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        manual_json_text = st.text_area(
            "Machine JSON", height=260, key="manual_json_input",
            placeholder='{"machine_type":"DFA","states":["q0"],"alphabet":["0","1"],...}',
        )
    with mc2:
        manual_oracle_text = st.text_area(
            "Oracle body (optional — leave blank to skip verification)",
            height=260, key="manual_oracle_input",
            placeholder="return s.endswith('101')",
        )

st.divider()
btn_label = "▶️ Run Pipeline" if manual_mode else (
    "💻 Generate Offline & Verify" if offline_mode else "🚀 Generate & Verify"
)
generate_btn = st.button(btn_label, type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

status_slot = st.empty()

if generate_btn:
    st.session_state["result"] = None
    st.session_state["attempts_log"] = []
    st.session_state["result_tabs"] = "📊 Result"

    if manual_mode:
        mj, oracle_fn, oracle_code, err, warn = prepare_manual_machine(
            manual_json_text, manual_oracle_text, alphabet_text
        )
        if err:
            st.error(err); st.stop()
        if warn:
            st.warning(warn)
        with status_slot.status("Running pipeline…", expanded=True) as status:
            st.write("Building automaton, minimizing, and verifying…")
            result = run_pipeline(mj, max_verify_len, oracle_fn)
            result["oracle_code"] = oracle_code
            status.update(label="✅ Pipeline complete", state="complete")
        st.session_state["result"] = result

    else:
        alphabet, alpha_err = parse_alphabet(alphabet_text)
        if alpha_err:
            st.error(f"Alphabet error: {alpha_err}"); st.stop()
        if not description.strip():
            st.error("Please enter a language description."); st.stop()
        if not offline_mode and not api_key:
            st.error("❌ No GEMINI_API_KEY. Switch to Offline AI or Manual JSON mode."); st.stop()

        best_result, best_mismatch, feedback = None, float("inf"), None
        attempts_log = []
        provider_label = f"Local Ollama ({active_model})" if offline_mode else f"Gemini ({active_model})"
        with status_slot.status(f"Calling {provider_label}…", expanded=True) as status:
            for attempt in range(1, max_retries + 1):
                label = f"Attempt {attempt}/{max_retries}"
                if feedback and feedback.get("type") == "counterexample":
                    st.write(f"🔁 {label}: fixing counterexample {feedback['counterexample']!r}…")
                elif feedback:
                    st.write(f"🔁 {label}: fixing error…")
                else:
                    st.write(f"🤖 {label}: generating machine…")

                try:
                    if attempt == 1 and feedback is None:
                        mj = _cached_initial_generate(backend, description, tuple(alphabet), active_model)
                    else:
                        mj, _ = generate_machine(
                            description, alphabet, active_model, 1, feedback, backend=backend
                        )
                except Exception as exc:
                    attempts_log.append({"attempt": str(attempt), "outcome": "Error", "counterexample": "—", "detail": str(exc)})
                    feedback = {"type": "error", "error_message": str(exc)}
                    if attempt == max_retries:
                        status.update(label="❌ Failed", state="error")
                    continue

                try:
                    oracle_fn_local, oracle_code_local = compile_oracle(mj.get("oracle_body", ""))
                except (OracleSecurityError, SyntaxError) as e:
                    attempts_log.append({"attempt": str(attempt), "outcome": "Oracle Error", "counterexample": "—", "detail": str(e)})
                    feedback = {"type": "error", "error_message": f"Oracle error: {e}"}
                    if attempt == max_retries:
                        status.update(label="❌ Failed", state="error")
                    continue

                result = run_pipeline(mj, max_verify_len, oracle_fn_local)
                result["oracle_code"] = oracle_code_local
                vr = result.get("verify_result")
                n_mm = vr["all_mismatch_count"] if vr else 0
                if n_mm < best_mismatch:
                    best_mismatch, best_result = n_mm, result

                if n_mm == 0:
                    attempts_log.append({"attempt": str(attempt), "outcome": "✅ Verified", "counterexample": "—",
                                         "detail": f"Verified on {vr['total_checked']:,} strings."})
                    status.update(label="✅ Verified!", state="complete"); break
                else:
                    cex, feedback = build_counterexample_feedback(result, mj)
                    attempts_log.append({"attempt": str(attempt), "outcome": f"❌ {n_mm} mismatches", "counterexample": repr(cex), "detail": str(cex)})
                    if attempt == max_retries:
                        # Auto-fallback: if LLM wrote a valid oracle, learn minimal DFA with L*
                        if use_lstar_fallback and oracle_fn_local is not None:
                            try:
                                best_result, l_rounds = synthesize_from_oracle(
                                    alphabet, oracle_fn_local, mj.get("oracle_body", ""),
                                    oracle_code_local, max_verify_len,
                                )
                                best_mismatch = 0
                                l_total = best_result["verify_result"]["total_checked"]
                                attempts_log.append({
                                    "attempt": "L*", "outcome": "✅ Verified (Angluin L*)",
                                    "counterexample": "—",
                                    "detail": f"Learned exact minimal DFA from oracle in {l_rounds} round(s); verified on {l_total:,} strings.",
                                })
                                status.update(label="✅ Synthesized & Verified via Angluin L*!", state="complete")
                            except Exception as lstar_exc:
                                attempts_log.append({"attempt": "L*", "outcome": "L* Error", "counterexample": "—", "detail": str(lstar_exc)})
                                status.update(label="⚠️ Could not verify", state="error")
                        else:
                            status.update(label="⚠️ Could not verify", state="error")

        st.session_state["attempts_log"] = attempts_log
        if best_result:
            st.session_state["result"] = best_result
        else:
            last_err = attempts_log[-1]["detail"] if attempts_log else "Unknown error"
            st.error(f"❌ No valid machine was produced after {max_retries} attempt(s).\n\n**Last error:** {last_err}")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

result = st.session_state.get("result")
if result is None:
    st.info("👆 Enter a language description (or paste Manual JSON) and click the button above to start.")
    st.stop()

if result.get("validation_errors"):
    st.error("Validation errors:\n" + "\n".join(f"- {e}" for e in result["validation_errors"]))
    st.stop()

if result.get("min_dfa") is None:
    st.error("Minimization failed. Something went wrong."); st.stop()

vr = result.get("verify_result")
if result.get("lstar_used"):
    st.info(
        f"🧠 **Angluin L\\* synthesis used:** the AI model wrote the Python oracle, and "
        f"our from-scratch L\\* learner synthesized the exact minimal DFA from it in "
        f"{result.get('lstar_rounds', '?')} round(s)."
    )
elif vr and vr.get("all_mismatch_count", 0) > 0:
    st.error("🚨 **Could not be verified; do not trust this machine.** Showing the best attempt below.")
    ob_rec = result.get("machine_json", {}).get("oracle_body", "")
    if ob_rec and st.button("🧠 Synthesize exact DFA from Oracle (Angluin L*)", key="lstar_recover_btn"):
        try:
            o_fn, o_code = compile_oracle(ob_rec)
            synth_res, l_rounds = synthesize_from_oracle(
                result["min_dfa"].alphabet, o_fn, ob_rec, o_code, max_verify_len,
            )
            st.session_state["result"] = synth_res
            st.rerun()
        except Exception as exc:
            st.error(f"Could not synthesize DFA from oracle: {exc}")

_TAB_LABELS = [
    "📊 Result", "🔬 Minimization Steps", "🔄 NFA → DFA",
    "🧪 Test a String", "✅ Verification",
]
if "key" in inspect.signature(st.tabs).parameters:
    tab_result, tab_min, tab_nfa, tab_test, tab_verify = st.tabs(
        _TAB_LABELS, key="result_tabs", on_change="rerun",
    )
else:
    tab_result, tab_min, tab_nfa, tab_test, tab_verify = st.tabs(_TAB_LABELS)

with tab_result:
    render_result_tab(result)
with tab_min:
    render_minimization_tab(result)
with tab_nfa:
    render_nfa_tab(result)
with tab_test:
    render_test_tab(result)
with tab_verify:
    render_verification_tab(result, st.session_state.get("attempts_log", []))
