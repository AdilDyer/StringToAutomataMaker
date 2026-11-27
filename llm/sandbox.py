"""
llm/sandbox.py
==============
Safe execution environment for oracle code supplied by the LLM.

The oracle is the BODY of a function  def oracle(s): → bool
We wrap it in a def, perform static AST safety checks, then exec it in a
restricted namespace.

Security model
--------------
1. AST check (structural): rejects dangerous constructs BEFORE any code runs.
2. Restricted builtins: __builtins__ replaced by an explicit whitelist.
3. Limited re module: only re.fullmatch / re.match / re.search / re.compile
   are exposed (via a small wrapper object).
4. No While-loop (prevents infinite loops; use for+range instead).
5. AST node count capped at 200 (prevents obfuscated large scripts).
"""

from __future__ import annotations
import ast
import textwrap
import types
import re as _re_module
from typing import Callable, Optional, Tuple


# ---------------------------------------------------------------------------
# Whitelist of allowed built-in names
# ---------------------------------------------------------------------------

_ALLOWED_BUILTINS = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "all": all,
    "any": any,
    "range": range,
    "sorted": sorted,
    "set": set,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "str": str,
    "int": int,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "True": True,
    "False": False,
    "None": None,
    "print": print,  # harmless; useful for debugging
}

# ---------------------------------------------------------------------------
# Limited re wrapper
# ---------------------------------------------------------------------------

class _SafeRe:
    """Expose only the safe subset of the re module."""
    fullmatch = staticmethod(_re_module.fullmatch)
    match = staticmethod(_re_module.match)
    search = staticmethod(_re_module.search)
    compile = staticmethod(_re_module.compile)


_SAFE_RE = _SafeRe()

# ---------------------------------------------------------------------------
# Forbidden AST node types
# ---------------------------------------------------------------------------

_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.While,
    ast.Delete,
)

# Names that must never appear as identifiers
_FORBIDDEN_NAME_PREFIXES = ("__",)

# Call names that are forbidden even if not imported
_FORBIDDEN_CALL_NAMES = frozenset({
    "exec", "eval", "open", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "__import__", "breakpoint",
})


# ---------------------------------------------------------------------------
# AST safety checker
# ---------------------------------------------------------------------------

class OracleSecurityError(Exception):
    """Raised when oracle code fails the security check."""
    pass


def _check_ast(tree: ast.AST) -> None:
    """
    Walk the AST and raise OracleSecurityError if any forbidden construct
    is found.
    """
    nodes = list(ast.walk(tree))

    # Cap node count
    if len(nodes) > 200:
        raise OracleSecurityError(
            f"Oracle AST has {len(nodes)} nodes (max 200). "
            "The oracle code is too complex."
        )

    for node in nodes:
        # Forbidden statement types
        if isinstance(node, _FORBIDDEN_NODES):
            raise OracleSecurityError(
                f"Forbidden construct in oracle: {type(node).__name__}"
            )

        # Attribute access starting with '_' (e.g. obj.__class__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise OracleSecurityError(
                f"Forbidden attribute access: .{node.attr}"
            )

        # Names starting with '__'
        if isinstance(node, ast.Name):
            for prefix in _FORBIDDEN_NAME_PREFIXES:
                if node.id.startswith(prefix):
                    raise OracleSecurityError(
                        f"Forbidden name: {node.id}"
                    )

        # Forbidden function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALL_NAMES:
                    raise OracleSecurityError(
                        f"Forbidden call: {node.func.id}()"
                    )
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_CALL_NAMES:
                    raise OracleSecurityError(
                        f"Forbidden call: .{node.func.attr}()"
                    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_oracle(oracle_body: str) -> Tuple[Callable[[str], bool], str]:
    """
    Validate and compile the oracle body into a callable.

    Parameters
    ----------
    oracle_body : the BODY of  def oracle(s):  (indentation will be fixed)

    Returns
    -------
    oracle_fn  : a Python callable oracle(s: str) -> bool
    full_code  : the full def string (for display in the UI)

    Raises
    ------
    OracleSecurityError : if the code fails the safety check
    SyntaxError         : if the code has a syntax error
    """
    # Dedent the body and wrap in a function definition
    body = textwrap.dedent(oracle_body).rstrip()
    # Indent body by 4 spaces for the function body
    indented = textwrap.indent(body, "    ")
    full_code = f"def oracle(s):\n{indented}\n"

    # Parse
    try:
        tree = ast.parse(full_code, mode="exec")
    except SyntaxError as exc:
        raise SyntaxError(f"Oracle syntax error: {exc}") from exc

    # Security check
    _check_ast(tree)

    # Compile and exec in restricted namespace
    namespace: dict = {
        "__builtins__": _ALLOWED_BUILTINS,
        "re": _SAFE_RE,
    }
    exec(compile(tree, "<oracle>", "exec"), namespace)  # noqa: S102

    oracle_fn = namespace.get("oracle")
    if oracle_fn is None or not callable(oracle_fn):
        raise OracleSecurityError("Oracle function not found after exec.")

    return oracle_fn, full_code
