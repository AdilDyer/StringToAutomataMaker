"""
llm/local.py
============
Offline local LLM backend via Ollama HTTP API (no extra pip dependencies).

Talks to a locally running Ollama server (default: http://localhost:11434)
using Python's standard library urllib.request.
"""

from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import List, Tuple

LOCAL_MODELS: List[str] = [
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "llama3.2:3b",
    "gemma3:4b",
]
DEFAULT_LOCAL_MODEL = LOCAL_MODELS[0]
LOCAL_TIMEOUT_S = 90


def get_ollama_url() -> str:
    """Return the configured Ollama base URL without a trailing slash."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def check_ollama_status(base_url: str | None = None) -> Tuple[bool, List[str], str]:
    """
    Probe the local Ollama server (/api/tags).

    Returns
    -------
    (is_running, installed_models, error_message)
    """
    url = (base_url or get_ollama_url()) + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in payload.get("models", []) if isinstance(m, dict) and "name" in m]
        return True, models, ""
    except urllib.error.URLError:
        return False, [], "Ollama server is not reachable at " + (base_url or get_ollama_url())
    except Exception as exc:
        return False, [], f"Error checking Ollama: {exc}"


def call_ollama(
    system_prompt: str,
    user_msg: str,
    model: str = DEFAULT_LOCAL_MODEL,
    timeout_s: int = LOCAL_TIMEOUT_S,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> str:
    """
    Call Ollama's /api/chat endpoint with JSON output format.
    Raises RuntimeError with actionable instructions on connection/model errors.
    """
    host = base_url or get_ollama_url()
    url = f"{host}/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404 or "not found" in detail.lower():
            raise RuntimeError(
                f"Local model '{model}' is not pulled yet. Run: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. Start it with `ollama serve` (or open Ollama app)."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout_s}s. Try a smaller model like `qwen2.5-coder:3b`."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama error: {exc}") from exc

    content = payload.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content
