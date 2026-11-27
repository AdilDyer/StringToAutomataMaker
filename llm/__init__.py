"""
llm/__init__.py
"""
from llm.client import generate_machine, validate_machine_json, AVAILABLE_MODELS, DEFAULT_MODEL
from llm.local import (
    LOCAL_MODELS,
    DEFAULT_LOCAL_MODEL,
    check_ollama_status,
    call_ollama,
)
from llm.sandbox import compile_oracle, OracleSecurityError

__all__ = [
    "generate_machine",
    "validate_machine_json",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "LOCAL_MODELS",
    "DEFAULT_LOCAL_MODEL",
    "check_ollama_status",
    "call_ollama",
    "compile_oracle",
    "OracleSecurityError",
]
