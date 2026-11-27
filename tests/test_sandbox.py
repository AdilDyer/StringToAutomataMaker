"""
tests/test_sandbox.py
=====================
Tests for llm/sandbox.py:
  - Dangerous oracle bodies are REJECTED (import, __class__, open, while, exec)
  - Normal oracle bodies work correctly
  - The whitelist allows common builtins and re methods
"""

import pytest
from llm.sandbox import compile_oracle, OracleSecurityError


# ---------------------------------------------------------------------------
# Should be REJECTED
# ---------------------------------------------------------------------------

class TestRejected:
    def test_import_statement_rejected(self):
        body = "import os\nreturn True"
        with pytest.raises((OracleSecurityError, SyntaxError)):
            compile_oracle(body)

    def test_from_import_rejected(self):
        body = "from os import path\nreturn True"
        with pytest.raises((OracleSecurityError, SyntaxError)):
            compile_oracle(body)

    def test_dunder_name_rejected(self):
        body = "x = __class__\nreturn True"
        with pytest.raises((OracleSecurityError, SyntaxError, NameError)):
            fn, _ = compile_oracle(body)
            fn("test")

    def test_dunder_attribute_access_rejected(self):
        body = "return s.__class__.__name__ == 'str'"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_open_call_rejected(self):
        body = "f = open('/etc/passwd')\nreturn True"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_while_loop_rejected(self):
        body = "while True:\n    pass"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_exec_call_rejected(self):
        body = "exec('import os')\nreturn True"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_eval_call_rejected(self):
        body = "return eval('1+1') == 2"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_globals_call_rejected(self):
        body = "return len(globals()) > 0"
        with pytest.raises(OracleSecurityError):
            compile_oracle(body)

    def test_dunder_import_rejected(self):
        body = "m = __import__('os')\nreturn True"
        with pytest.raises((OracleSecurityError, SyntaxError)):
            compile_oracle(body)


# ---------------------------------------------------------------------------
# Should WORK correctly
# ---------------------------------------------------------------------------

class TestAllowed:
    def test_simple_endswith(self):
        fn, code = compile_oracle("return s.endswith('101')")
        assert fn("101") is True
        assert fn("0101") is True
        assert fn("100") is False
        assert fn("") is False

    def test_count_modulo(self):
        fn, _ = compile_oracle("return s.count('a') % 3 == 0")
        assert fn("") is True       # 0 a's
        assert fn("aaa") is True    # 3 a's
        assert fn("a") is False
        assert fn("aa") is False

    def test_re_fullmatch_allowed(self):
        fn, _ = compile_oracle("return bool(re.fullmatch('[01]*1[01][01]', s))")
        assert fn("100") is True
        assert fn("0100") is True
        assert fn("") is False

    def test_len_and_range_allowed(self):
        fn, _ = compile_oracle(
            "count = 0\n"
            "for i in range(len(s)):\n"
            "    if s[i] == 'a':\n"
            "        count += 1\n"
            "return count % 2 == 0"
        )
        assert fn("") is True
        assert fn("aa") is True
        assert fn("a") is False

    def test_set_and_all_allowed(self):
        fn, _ = compile_oracle(
            "return all(c in {'a', 'b'} for c in s) and len(s) > 0"
        )
        assert fn("ab") is True
        assert fn("") is False

    def test_int_conversion_allowed(self):
        fn, _ = compile_oracle(
            "if s == '':\n    return True\nreturn int(s, 2) % 4 == 0"
        )
        assert fn("") is True
        assert fn("100") is True  # 4 in binary
        assert fn("101") is False

    def test_empty_string_handled(self):
        fn, _ = compile_oracle("return len(s) == 0")
        assert fn("") is True
        assert fn("a") is False
