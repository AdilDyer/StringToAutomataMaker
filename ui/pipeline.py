"""
ui/pipeline.py
==============
Core pipeline runner and mode orchestration:
  - build_automaton_from_json: parses DFA or NFA JSON (running subset construction if NFA)
  - run_pipeline: validate -> minimize -> verify -> cross-verify
  - run_manual_pipeline: handles Manual JSON mode validation and optional oracle
  - parse_alphabet: validates 1-4 distinct single-char alphabet symbols
"""

from __future__ import annotations
import json
from typing import Callable, Dict, List, Optional, Tuple

from automata.dfa import DFA
from automata.nfa import NFA
from automata.minimize import minimize
from automata.verify import verify, cross_verify
from automata.convert import dfa_to_grammar, dfa_to_regex
from automata.lstar import learn_dfa_lstar
from llm.client import validate_machine_json
from llm.sandbox import compile_oracle, OracleSecurityError


def parse_alphabet(text: str) -> Tuple[List[str], str]:
    """Validate comma-separated alphabet (1–4 distinct single-character symbols)."""
    raw = [s.strip() for s in text.split(",") if s.strip()]
    if not raw:
        return [], "Alphabet cannot be empty."
    if len(raw) > 4:
        return [], f"Maximum 4 symbols allowed; got {len(raw)}."
    if len(set(raw)) != len(raw):
        return [], "Alphabet symbols must be distinct."
    for sym in raw:
        if len(sym) != 1:
            return [], f"Each symbol must be 1 character; '{sym}' is not."
    return raw, ""


def build_automaton_from_json(mj: dict) -> Tuple[DFA, Optional[list]]:
    """
    Parse machine_json and return (dfa, nfa_table).
    nfa_table is the subset-construction table if the machine was an NFA,
    else None. Raises ValueError for structural errors.
    """
    raw_transitions = mj.get("transitions", {})

    if mj["machine_type"] == "DFA":
        delta: Dict[str, Dict[str, str]] = {}
        for src, row in raw_transitions.items():
            delta[src] = {}
            for sym, tgts in row.items():
                if isinstance(tgts, list):
                    if len(tgts) != 1:
                        raise ValueError(
                            f"DFA transition δ({src},{sym}) must have exactly 1 target, got {tgts}."
                        )
                    delta[src][sym] = tgts[0]
                elif isinstance(tgts, str):
                    delta[src][sym] = tgts
                else:
                    raise ValueError(f"Invalid transition format for δ({src},{sym}).")

        dfa = DFA(
            states=mj["states"], alphabet=mj["alphabet"],
            start=mj["start"], finals=mj["finals"], delta=delta,
        )
        return dfa, None

    nfa_delta: Dict[str, Dict[str, List[str]]] = {}
    for src, row in raw_transitions.items():
        nfa_delta[src] = {}
        for sym, tgts in row.items():
            nfa_delta[src][sym] = list(tgts) if isinstance(tgts, list) else [tgts]

    nfa = NFA(
        states=mj["states"], alphabet=mj["alphabet"],
        start=mj["start"], finals=mj["finals"], delta=nfa_delta,
    )
    return nfa.to_dfa()


def run_pipeline(
    machine_json: dict,
    max_verify_len: int,
    oracle_fn: Optional[Callable] = None,
) -> dict:
    """Run build -> validate -> minimize -> verify -> cross-verify on machine_json."""
    result: dict = {
        "machine_json": machine_json,
        "nfa_table": None,
        "original_dfa": None,
        "min_dfa": None,
        "min_info": None,
        "validation_errors": [],
        "dead_state_added": False,
        "verify_result": None,
        "cross_verify_result": None,
        "oracle_code": None,
        "grammar": None,
        "regex": None,
    }

    try:
        dfa, nfa_table = build_automaton_from_json(machine_json)
    except ValueError as exc:
        result["validation_errors"] = [str(exc)]
        return result

    result["nfa_table"] = nfa_table
    result["original_dfa"] = dfa

    errors = dfa.validate()
    if errors:
        result["validation_errors"] = errors
        return result

    min_dfa, min_info = minimize(dfa)
    result["min_dfa"] = min_dfa
    result["min_info"] = min_info
    result["dead_state_added"] = min_info["dead_added"]
    result["grammar"] = dfa_to_grammar(min_dfa)
    result["regex"] = dfa_to_regex(min_dfa)

    if oracle_fn is not None:
        dfa_for_verify = DFA(
            states=list(dfa.states), alphabet=list(dfa.alphabet),
            start=dfa.start, finals=set(dfa.finals),
            delta={s: dict(r) for s, r in dfa.delta.items()},
        )
        dfa_for_verify.complete()
        result["verify_result"] = verify(dfa_for_verify, oracle_fn, max_len=max_verify_len)

    orig_complete = DFA(
        states=list(dfa.states), alphabet=list(dfa.alphabet),
        start=dfa.start, finals=set(dfa.finals),
        delta={s: dict(r) for s, r in dfa.delta.items()},
    )
    orig_complete.complete()
    result["cross_verify_result"] = cross_verify(
        orig_complete, min_dfa, max_len=min(max_verify_len, 8)
    )
    return result


def prepare_manual_machine(
    manual_json_text: str,
    manual_oracle_text: str,
    fallback_alphabet_text: str,
) -> Tuple[Optional[dict], Optional[Callable], Optional[str], Optional[str], Optional[str]]:
    """
    Parse and validate Manual JSON mode inputs.
    Returns (machine_json, oracle_fn, oracle_code, error_msg, warning_msg).
    """
    if not manual_json_text.strip():
        return None, None, None, "Please paste a machine JSON (or select a built-in example above).", None
    try:
        machine_json = json.loads(manual_json_text)
    except json.JSONDecodeError as exc:
        return None, None, None, f"JSON parse error: {exc}", None
    if not isinstance(machine_json, dict):
        return None, None, None, "Root JSON value must be an object `{...}`.", None

    machine_json.setdefault("oracle_body", "")
    machine_json.setdefault("explanation", "")

    if isinstance(machine_json.get("alphabet"), list) and machine_json["alphabet"]:
        alphabet, alpha_err = parse_alphabet(",".join(str(x) for x in machine_json["alphabet"]))
    else:
        alphabet, alpha_err = parse_alphabet(fallback_alphabet_text)
        if not alpha_err:
            machine_json["alphabet"] = alphabet
    if alpha_err:
        return None, None, None, f"Alphabet error: {alpha_err}", None

    errors = validate_machine_json(machine_json, alphabet)
    if errors:
        return None, None, None, "Schema errors:\n" + "\n".join(f"- {e}" for e in errors), None

    oracle_fn, oracle_code, warn = None, None, None
    ob = manual_oracle_text.strip() or str(machine_json.get("oracle_body", "")).strip()
    if ob:
        try:
            oracle_fn, oracle_code = compile_oracle(ob)
        except (OracleSecurityError, SyntaxError) as exc:
            warn = f"Oracle rejected (skipping verification): {exc}"
    return machine_json, oracle_fn, oracle_code, None, warn


def build_counterexample_feedback(result: dict, machine_json: dict) -> Tuple[str, dict]:
    """Build counterexample feedback dict with trace for the next LLM retry attempt."""
    vr = result["verify_result"]
    first_mm = vr["mismatches"][0]
    cex = vr["shortest_counterexample"] if vr["shortest_counterexample"] is not None else first_mm["string"]
    od = result.get("original_dfa")
    trace_fb = []
    if od:
        oc = DFA(
            list(od.states), list(od.alphabet), od.start,
            set(od.finals), {s: dict(r) for s, r in od.delta.items()},
        )
        oc.complete()
        _, trace_fb, _ = oc.run(cex)
    return cex, {
        "type": "counterexample",
        "counterexample": cex,
        "oracle_answer": first_mm["oracle"],
        "dfa_answer": first_mm["dfa"],
        "counterexamples": vr["mismatches"][:4],
        "trace": trace_fb,
        "prev_machine": machine_json,
    }


def synthesize_from_oracle(
    alphabet: List[str],
    oracle_fn: Callable[[str], bool],
    oracle_body: str,
    oracle_code: str,
    max_verify_len: int,
) -> Tuple[dict, int]:
    """
    Use Angluin's L* algorithm to synthesize the exact minimal DFA directly
    from the compiled oracle, then run it through the full pipeline.
    Returns (result_dict, lstar_rounds).
    """
    learned_dfa, lstar_rounds = learn_dfa_lstar(alphabet, oracle_fn, max_verify_len=max_verify_len)
    synth_mj = learned_dfa.to_dict()
    synth_mj["oracle_body"] = oracle_body
    synth_mj["explanation"] = (
        f"Synthesized directly from the verified Python oracle via Angluin's L* "
        f"active learning algorithm in {lstar_rounds} equivalence round(s)."
    )
    res = run_pipeline(synth_mj, max_verify_len, oracle_fn)
    res["oracle_code"] = oracle_code
    res["lstar_used"] = True
    res["lstar_rounds"] = lstar_rounds
    return res, lstar_rounds
