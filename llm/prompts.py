"""
llm/prompts.py
==============
All prompt templates as module-level constants.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in Theory of Computation and Formal Languages.

Your task is to design a finite automaton (DFA or NFA) that precisely accepts
the language described by the user, and to write a Python oracle function that
implements the same language.

STRICT OUTPUT RULES:
- Output ONLY a single JSON object — no prose, no markdown, no code fences.
- Write "explanation" FIRST so you plan every state's exact meaning before
  writing "finals" and "transitions".
- Check what happens to the EMPTY STRING "" (start state q0 in finals iff "" is accepted).
- Use only symbols that appear in the user-supplied alphabet; never invent new symbols.
- Prefer a DFA when a small DFA exists naturally; use an NFA only when it results
  in significantly fewer states.
- For "no consecutive X" / "must not contain X" languages, include a non-accepting
  trap/dead state (e.g. q2) once X has been seen!
- Name states q0, q1, q2, … in order; q0 is always the start state.
- ALL transitions must map to a LIST of states (even for DFAs, use a one-element list).
- The oracle_body must be pure Python returning a bool for any string s over the
  alphabet (including ""). Do NOT import anything or use while loops.
- The alphabet field in JSON must exactly equal the alphabet provided by the user.

REQUIRED JSON SCHEMA:
{
  "explanation": "<plan what each state q0, q1, ... means and which are accepting>",
  "machine_type": "DFA" | "NFA",
  "states": ["q0", "q1", ...],
  "alphabet": ["a", "b"],
  "start": "q0",
  "finals": ["q2"],
  "transitions": {
    "q0": {"a": ["q1"], "b": ["q0"]},
    ...
  },
  "oracle_body": "<python statements; the body of def oracle(s):>"
}

FEW-SHOT EXAMPLES:

--- EXAMPLE 1: "Binary strings ending in 01" (alphabet: 0, 1) ---
{
  "explanation": "q0 is start/reset (empty or ends in 1). q1 means last symbol seen was 0. q2 means ends in 01 (accepting).",
  "machine_type": "DFA",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start": "q0",
  "finals": ["q2"],
  "transitions": {
    "q0": {"0": ["q1"], "1": ["q0"]},
    "q1": {"0": ["q1"], "1": ["q2"]},
    "q2": {"0": ["q1"], "1": ["q0"]}
  },
  "oracle_body": "return s.endswith('01')"
}

--- EXAMPLE 2: "Strings over {a,b} where the number of a's is divisible by 3" ---
{
  "explanation": "q0 means #a ≡ 0 (mod 3) — accepting (including empty string). q1 means #a ≡ 1 (mod 3). q2 means #a ≡ 2 (mod 3). b leaves count unchanged.",
  "machine_type": "DFA",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["a", "b"],
  "start": "q0",
  "finals": ["q0"],
  "transitions": {
    "q0": {"a": ["q1"], "b": ["q0"]},
    "q1": {"a": ["q2"], "b": ["q1"]},
    "q2": {"a": ["q0"], "b": ["q2"]}
  },
  "oracle_body": "count = s.count('a')\\nreturn count % 3 == 0"
}

--- EXAMPLE 3: "Strings over {a,b} with no two consecutive a's" ---
{
  "explanation": "q0 is start or last symbol was b (accepting). q1 means last symbol was a (still valid, accepting). q2 is the trap/dead state after seeing 'aa' (rejected forever).",
  "machine_type": "DFA",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["a", "b"],
  "start": "q0",
  "finals": ["q0", "q1"],
  "transitions": {
    "q0": {"a": ["q1"], "b": ["q0"]},
    "q1": {"a": ["q2"], "b": ["q0"]},
    "q2": {"a": ["q2"], "b": ["q2"]}
  },
  "oracle_body": "return 'aa' not in s"
}
"""

# ---------------------------------------------------------------------------
# User prompt template (first attempt)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Design a finite automaton for the following language:

DESCRIPTION: {description}

ALPHABET: {alphabet_list}

Remember:
- Write "explanation" first to carefully plan each state and transition.
- The "alphabet" field in your JSON must be exactly: {alphabet_list}
- Use only the symbols listed above in all transitions.
- Output ONLY the JSON object.
"""

# ---------------------------------------------------------------------------
# Retry prompt (with feedback from counterexamples)
# ---------------------------------------------------------------------------

RETRY_PROMPT_TEMPLATE = """\
Your previous DFA transitions/finals were INCORRECT. DO NOT return the same transitions again — fix the machine!

ORIGINAL DESCRIPTION: {description}
ALPHABET: {alphabet_list}

COUNTEREXAMPLES FOUND (Oracle = expected, DFA = what your machine returned):
{counterexamples_block}

Trace of your DFA on shortest counterexample {counterexample!r}: {trace_summary}

PREVIOUS MACHINE (contains a bug — modify states, finals, or transitions to fix ALL counterexamples above):
{prev_json}

Check:
1. Should empty string "" be accepted or rejected? (Controls if q0 is in finals.)
2. Do you need an extra state or trap/dead state for forbidden patterns?
3. Trace every counterexample above step-by-step on your new transitions before outputting.
Output ONLY the corrected JSON object.
"""

# ---------------------------------------------------------------------------
# Error feedback prompt (JSON/schema/sandbox failure)
# ---------------------------------------------------------------------------

ERROR_FEEDBACK_TEMPLATE = """\
Your previous response could not be parsed or was invalid.

ORIGINAL DESCRIPTION: {description}
ALPHABET: {alphabet_list}

ERROR: {error_message}

Please output a corrected JSON object that matches the required schema exactly.
Output ONLY the JSON object.
"""
