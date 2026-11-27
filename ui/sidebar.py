"""
ui/sidebar.py
=============
Sidebar controls for AutomataGPT:
  Mode selection (Gemini Cloud, Offline Local LLM via Ollama, Manual JSON),
  model selection, retry count, verification length, and provider status.
"""

from __future__ import annotations
import os
from typing import Tuple

import streamlit as st

from automata.verify import count_strings
from llm.client import AVAILABLE_MODELS, DEFAULT_MODEL
from llm.local import LOCAL_MODELS, DEFAULT_LOCAL_MODEL, check_ollama_status


def render_sidebar() -> Tuple[str, str, int, int, str, bool]:
    """
    Render the sidebar and return:
      (mode, active_model, max_retries, max_verify_len, api_key, use_lstar_fallback)
    """
    with st.sidebar:
        st.title("⚙️ Settings")
        mode = st.radio(
            "Mode",
            ["🤖 AI mode (Gemini)", "💻 Offline AI (Local LLM)", "✏️ Manual JSON mode"],
        )
        manual_mode = "Manual" in mode
        offline_mode = "Offline" in mode

        st.divider()
        st.subheader("Offline Local LLM (Ollama)" if offline_mode else "AI Settings")

        ollama_ok, installed_local, ollama_err = (False, [], "")
        if offline_mode:
            ollama_ok, installed_local, ollama_err = check_ollama_status()
            local_opts = list(dict.fromkeys(installed_local + LOCAL_MODELS)) + ["Custom…"]
            sel_local = st.selectbox("Local Model", local_opts)
            if sel_local == "Custom…":
                active_model = st.text_input(
                    "Custom Ollama model", placeholder="qwen2.5-coder:3b"
                ).strip() or DEFAULT_LOCAL_MODEL
            else:
                active_model = sel_local
        else:
            sel = st.selectbox("Model", AVAILABLE_MODELS + ["Custom…"], disabled=manual_mode)
            if sel == "Custom…":
                active_model = st.text_input(
                    "Custom model name",
                    placeholder="models/gemini-3.5-flash-lite",
                    disabled=manual_mode,
                ).strip() or DEFAULT_MODEL
            else:
                active_model = sel

        max_retries = st.slider("Max retries", 1, 5, 3, disabled=manual_mode)
        use_lstar_fallback = st.checkbox(
            "Auto-learn DFA via Angluin L* if LLM DFA fails",
            value=offline_mode,
            disabled=manual_mode,
            help=(
                "Small local LLMs often write a correct Python oracle even when their "
                "DFA transition table has bugs. When enabled, Angluin's L* algorithm "
                "synthesizes the exact minimal DFA directly from that oracle."
            ),
        )

        st.divider()
        st.subheader("Verification")
        max_verify_len = st.slider("Max string length to verify", 1, 12, 10)
        st.caption(f"≈ {count_strings(['a','b'], max_verify_len):,} strings for 2-symbol alphabet.")

        st.divider()
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if offline_mode:
            if ollama_ok:
                st.success(f"✅ Ollama running ({len(installed_local)} model(s) installed)")
                if not installed_local:
                    st.warning("No models pulled yet. Run:\n`ollama pull qwen2.5-coder:3b`")
            else:
                st.warning(
                    f"⚠️ {ollama_err}\n\n"
                    "**Quick setup (100% offline):**\n"
                    "1. `brew install ollama` (or ollama.com)\n"
                    "2. `ollama serve`\n"
                    "3. `ollama pull qwen2.5-coder:3b`"
                )
        elif api_key:
            st.success("✅ GEMINI_API_KEY found")
        else:
            st.error("❌ GEMINI_API_KEY not set.\nUse **Offline AI** or **Manual JSON mode**.")

    return mode, active_model, max_retries, max_verify_len, api_key, use_lstar_fallback
