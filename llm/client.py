"""
llm/client.py
=============
Unified LLM client (Gemini Cloud API & Offline Local Ollama) for AutomataGPT.
"""

from __future__ import annotations
import json
import os
import re
from typing import List, Optional, Tuple

from llm.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    RETRY_PROMPT_TEMPLATE,
    ERROR_FEEDBACK_TEMPLATE,
)
from llm.local import call_ollama

# Configurable Gemini model list — edit here if models are added or retired.
AVAILABLE_MODELS: List[str] = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
API_TIMEOUT_MS = 45_000  # 45-second HTTP timeout per API call

REQUIRED_KEYS = {
    "machine_type", "states", "alphabet", "start", "finals", "transitions",
    "oracle_body", "explanation",
}


def validate_machine_json(data: dict, expected_alphabet: List[str]) -> List[str]:
    """Validate a parsed machine JSON dict and return human-readable error strings."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["Root JSON value must be an object."]

    for k in REQUIRED_KEYS:
        if k not in data:
            errors.append(f"Missing required key: '{k}'")
    if errors:
        return errors

    if data["machine_type"] not in ("DFA", "NFA"):
        errors.append(f"machine_type must be 'DFA' or 'NFA', got {data['machine_type']!r}")

    states = data.get("states")
    if not isinstance(states, list) or not states or not all(isinstance(s, str) for s in states):
        errors.append("'states' must be a non-empty list of strings.")
        return errors
    state_set = set(states)

    raw_alpha = data.get("alphabet")
    if not isinstance(raw_alpha, list) or not all(isinstance(a, str) for a in raw_alpha):
        errors.append("'alphabet' must be a list of strings.")
        return errors
    if sorted(raw_alpha) != sorted(expected_alphabet):
        errors.append(
            f"Alphabet mismatch: expected {sorted(expected_alphabet)}, got {sorted(raw_alpha)}."
        )

    if not isinstance(data.get("start"), str) or data.get("start") not in state_set:
        errors.append(f"start state {data.get('start')!r} is not in states list.")

    finals = data.get("finals")
    if not isinstance(finals, list):
        errors.append("'finals' must be a list of state names.")
    else:
        for f in finals:
            if not isinstance(f, str) or f not in state_set:
                errors.append(f"Final state {f!r} is not in states list.")

    trans = data.get("transitions")
    if not isinstance(trans, dict):
        errors.append("'transitions' must be a dict.")
        return errors

    alpha_set = set(raw_alpha)
    for src, row in trans.items():
        if src not in state_set:
            errors.append(f"Transition source '{src}' is not in states.")
        if not isinstance(row, dict):
            errors.append(f"Transition row for '{src}' must be a dict.")
            continue
        for sym, tgts in row.items():
            if sym != "eps" and sym not in alpha_set:
                errors.append(f"Symbol '{sym}' in δ({src}, ·) is not in alphabet.")
            if not isinstance(tgts, list):
                errors.append(
                    f"Transition targets δ({src},{sym}) must be a list, got {type(tgts).__name__}."
                )
                continue
            for t in tgts:
                if not isinstance(t, str) or t not in state_set:
                    errors.append(f"Transition target '{t}' in δ({src},{sym}) is not in states.")

    if not isinstance(data.get("oracle_body"), str):
        errors.append("'oracle_body' must be a string.")
    if not isinstance(data.get("explanation"), str):
        errors.append("'explanation' must be a string.")
    return errors


def _extract_json(text: str) -> str:
    """Strip ```json ... ``` fences and return raw JSON."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        return brace_match.group(0)
    return text


def _friendly_api_error(exc: Exception) -> str:
    """Map raw SDK/HTTP exceptions to concise user-friendly messages."""
    msg = str(exc)
    low = msg.lower()
    if "api_key" in low or "api key" in low or "401" in msg or "403" in msg:
        return "Invalid or unauthorized GEMINI_API_KEY. Check your .env file."
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return "API quota/rate limit reached (429). Try another model or Offline/Manual mode."
    if "503" in msg or "unavailable" in low:
        return "Selected Gemini model is temporarily overloaded (503). Try another model."
    if "404" in msg or "not_found" in low:
        return "Selected Gemini model was not found / retired (404). Pick another model or custom name."
    if "504" in msg or "timeout" in low or "deadline" in low:
        return "Gemini request timed out. Please retry or try a faster model."
    if "safety" in low or "blocked" in low:
        return "Gemini blocked the response due to safety filters."
    return f"Gemini API error: {msg[:300]}"


def _build_user_prompt(
    description: str, alphabet: List[str], attempt: int, feedback: Optional[dict]
) -> str:
    """Build initial or feedback prompt depending on attempt and error/counterexample."""
    alphabet_list = str(alphabet)
    if attempt == 1 and feedback is None:
        return USER_PROMPT_TEMPLATE.format(description=description, alphabet_list=alphabet_list)
    if feedback and feedback.get("type") == "counterexample":
        trace_summary = " → ".join(
            f"δ({s},{a})={t}" for s, a, t in feedback.get("trace", [])[:5]
        )
        cex_list = feedback.get("counterexamples") or [
            {"string": feedback["counterexample"], "oracle": feedback["oracle_answer"], "dfa": feedback["dfa_answer"]}
        ]
        cex_lines = "\n".join(
            f"  - {m['string']!r}: Oracle={m['oracle']}, Your DFA={m['dfa']}"
            for m in cex_list[:4]
        )
        return RETRY_PROMPT_TEMPLATE.format(
            description=description, alphabet_list=alphabet_list,
            counterexample=feedback["counterexample"],
            counterexamples_block=cex_lines,
            trace_summary=trace_summary or "(empty — start state q0)",
            prev_json=json.dumps(feedback.get("prev_machine", {}), indent=2),
        )
    return ERROR_FEEDBACK_TEMPLATE.format(
        description=description, alphabet_list=alphabet_list,
        error_message=(feedback or {}).get("error_message", "Unknown error"),
    )


def _call_gemini(user_msg: str, model: str, temperature: float = 0.0) -> str:
    """Call Google Gemini API and return raw response text."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        raise ImportError("google-genai is not installed. Run: pip install google-genai") from exc

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to .env or switch to Offline AI / Manual JSON mode."
        )
    client = genai.Client(api_key=api_key, http_options={"timeout": API_TIMEOUT_MS})
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_msg,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise RuntimeError(_friendly_api_error(exc)) from exc
    if not response.text:
        raise RuntimeError("Empty response from Gemini (possibly safety-blocked).")
    return response.text


def generate_machine(
    description: str,
    alphabet: List[str],
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    feedback: Optional[dict] = None,
    backend: str = "gemini",
) -> Tuple[dict, List[dict]]:
    """
    Generate and validate a machine JSON using either 'gemini' (cloud) or 'ollama' (offline).
    """
    attempt_log: List[dict] = []

    for attempt in range(1, max_retries + 1):
        user_msg = _build_user_prompt(description, alphabet, attempt, feedback)
        temp = 0.0 if (attempt == 1 and feedback is None) else 0.25
        try:
            raw_text = (
                call_ollama(SYSTEM_PROMPT, user_msg, model=model, temperature=temp)
                if backend == "ollama"
                else _call_gemini(user_msg, model=model, temperature=temp)
            )
        except Exception as exc:
            err_str = str(exc)
            attempt_log.append({"attempt": attempt, "outcome": "Backend error", "detail": err_str})
            feedback = {"type": "error", "error_message": err_str}
            if attempt == max_retries:
                raise RuntimeError(err_str) from exc
            continue

        try:
            data = json.loads(_extract_json(raw_text))
        except (json.JSONDecodeError, ValueError) as exc:
            err_str = f"JSON parse error: {exc}\nRaw response:\n{raw_text[:400]}"
            attempt_log.append({"attempt": attempt, "outcome": "Parse error", "detail": err_str})
            feedback = {"type": "error", "error_message": err_str}
            if attempt == max_retries:
                raise ValueError(err_str) from exc
            continue

        errors = validate_machine_json(data, alphabet)
        if errors:
            err_str = "Schema validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            attempt_log.append({"attempt": attempt, "outcome": "Validation error", "detail": err_str})
            feedback = {"type": "error", "error_message": err_str}
            if attempt == max_retries:
                raise ValueError(err_str)
            continue

        attempt_log.append({"attempt": attempt, "outcome": "Success", "detail": "Schema validated."})
        return data, attempt_log

    raise RuntimeError("generate_machine: exhausted all retries without success.")
