"""
examples.py
===========
Five built-in example language descriptions for AutomataGPT.

Each example stores:
  description          : human-readable language description
  alphabet             : list of symbols
  expected_min_states  : expected number of states in the MINIMIZED DFA
                         (complete DFA, including dead state if reachable)
  machine_json         : hardcoded correct machine JSON for offline testing
  oracle_body          : oracle function body for verification
"""

from __future__ import annotations
from typing import List, Dict, Any

EXAMPLES: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # 1. Number of a's divisible by 3
    # -----------------------------------------------------------------------
    {
        "description": "Strings over {a,b} where the number of a's is divisible by 3",
        "alphabet": ["a", "b"],
        "expected_min_states": 3,
        "machine_json": {
            "machine_type": "DFA",
            "states": ["q0", "q1", "q2"],
            "alphabet": ["a", "b"],
            "start": "q0",
            "finals": ["q0"],
            "transitions": {
                "q0": {"a": ["q1"], "b": ["q0"]},
                "q1": {"a": ["q2"], "b": ["q1"]},
                "q2": {"a": ["q0"], "b": ["q2"]},
            },
            "oracle_body": "return s.count('a') % 3 == 0",
            "explanation": (
                "q0: #a ≡ 0 (mod 3) — accepting. "
                "q1: #a ≡ 1 (mod 3). "
                "q2: #a ≡ 2 (mod 3). b's don't change the residue."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # 2. Binary strings ending in 101
    # -----------------------------------------------------------------------
    {
        "description": "Binary strings ending in 101",
        "alphabet": ["0", "1"],
        "expected_min_states": 4,
        "machine_json": {
            "machine_type": "DFA",
            "states": ["q0", "q1", "q2", "q3"],
            "alphabet": ["0", "1"],
            "start": "q0",
            "finals": ["q3"],
            "transitions": {
                "q0": {"0": ["q0"], "1": ["q1"]},
                "q1": {"0": ["q2"], "1": ["q1"]},
                "q2": {"0": ["q0"], "1": ["q3"]},
                "q3": {"0": ["q2"], "1": ["q1"]},
            },
            "oracle_body": "return s.endswith('101')",
            "explanation": (
                "q0: initial/reset (last seen not 1). "
                "q1: last symbol was 1 (possible start of 101). "
                "q2: seen 10 as suffix. "
                "q3: seen 101 as suffix — accepting."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # 3. Binary strings containing substring 11
    # -----------------------------------------------------------------------
    {
        "description": "Binary strings that contain the substring 11",
        "alphabet": ["0", "1"],
        "expected_min_states": 3,
        "machine_json": {
            "machine_type": "DFA",
            "states": ["q0", "q1", "q2"],
            "alphabet": ["0", "1"],
            "start": "q0",
            "finals": ["q2"],
            "transitions": {
                "q0": {"0": ["q0"], "1": ["q1"]},
                "q1": {"0": ["q0"], "1": ["q2"]},
                "q2": {"0": ["q2"], "1": ["q2"]},
            },
            "oracle_body": "return '11' in s",
            "explanation": (
                "q0: haven't seen a 1 yet (or last was 0). "
                "q1: last symbol was 1 (but not two consecutive 1s yet). "
                "q2: have seen '11' — trap/accept state."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # 4. Binary strings with no two consecutive 1s
    # -----------------------------------------------------------------------
    {
        "description": "Binary strings with no two consecutive 1s",
        "alphabet": ["0", "1"],
        "expected_min_states": 3,  # q0, q1, dead (after '11')
        "machine_json": {
            "machine_type": "DFA",
            "states": ["q0", "q1", "q2"],
            "alphabet": ["0", "1"],
            "start": "q0",
            "finals": ["q0", "q1"],
            "transitions": {
                "q0": {"0": ["q0"], "1": ["q1"]},
                "q1": {"0": ["q0"], "1": ["q2"]},
                "q2": {"0": ["q2"], "1": ["q2"]},
            },
            "oracle_body": "return '11' not in s",
            "explanation": (
                "q0: last symbol was 0 or start. "
                "q1: last symbol was 1 (OK so far). "
                "q2: dead state — '11' was seen, string is rejected."
            ),
        },
    },
    # -----------------------------------------------------------------------
    # 5. Binary strings representing numbers divisible by 4
    #    (number = binary integer; leading zeros allowed; empty string = 0 → accept)
    # -----------------------------------------------------------------------
    {
        "description": "Binary strings representing numbers divisible by 4 (including 0 represented as empty string or '0')",
        "alphabet": ["0", "1"],
        "expected_min_states": 3,
        "machine_json": {
            "machine_type": "DFA",
            "states": ["q0", "q1", "q2"],
            "alphabet": ["0", "1"],
            "start": "q0",
            "finals": ["q0"],
            "transitions": {
                "q0": {"0": ["q0"], "1": ["q1"]},
                "q1": {"0": ["q2"], "1": ["q1"]},
                "q2": {"0": ["q0"], "1": ["q1"]},
            },
            "oracle_body": (
                "if s == '':\n"
                "    return True\n"
                "return int(s, 2) % 4 == 0"
            ),
            "explanation": (
                "Tracks the last two bits (the value mod 4). "
                "q0: last two bits are 00 (or fewer than 2 bits seen, treating prefix as 0) — accepting. "
                "q1: last bit is 1 (value mod 4 is 1 or 3). "
                "q2: last two bits are 10 (value mod 4 is 2)."
            ),
        },
    },
]


def get_example_names() -> List[str]:
    """Return display names for the example dropdown."""
    return [ex["description"] for ex in EXAMPLES]


def get_example(index: int) -> Dict[str, Any]:
    """Return the example dict at *index*."""
    return EXAMPLES[index]
