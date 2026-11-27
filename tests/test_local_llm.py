"""
tests/test_local_llm.py
======================
Offline unit tests for the Ollama local LLM backend (llm/local.py + llm/client.py).
Uses mocked HTTP responses — no live Ollama server or network required.
"""

from io import BytesIO
import json
import urllib.error
from unittest.mock import patch

import pytest

from llm.client import generate_machine
from llm.local import call_ollama, check_ollama_status
from examples import EXAMPLES


class _FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestOllamaStatus:
    def test_status_when_running(self):
        payload = {"models": [{"name": "qwen2.5-coder:3b"}, {"name": "llama3.2:3b"}]}
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            ok, models, err = check_ollama_status("http://localhost:11434")
        assert ok is True
        assert models == ["qwen2.5-coder:3b", "llama3.2:3b"]
        assert err == ""

    def test_status_when_not_running(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            ok, models, err = check_ollama_status("http://localhost:11434")
        assert ok is False
        assert models == []
        assert "not reachable" in err.lower()


class TestCallOllama:
    def test_returns_message_content(self):
        payload = {"message": {"role": "assistant", "content": '{"ok": true}'}}
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            out = call_ollama("sys", "user", model="qwen2.5-coder:3b")
        assert out == '{"ok": true}'

    def test_404_gives_pull_hint(self):
        err = urllib.error.HTTPError(
            "http://localhost:11434/api/chat", 404, "Not Found", {},
            BytesIO(b'{"error":"model not found"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="ollama pull qwen2.5-coder:3b"):
                call_ollama("sys", "user", model="qwen2.5-coder:3b")

    def test_connection_error_gives_serve_hint(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError, match="ollama serve"):
                call_ollama("sys", "user", model="qwen2.5-coder:3b")


class TestGenerateMachineOffline:
    def test_offline_backend_success(self):
        ex = EXAMPLES[0]
        mj = ex["machine_json"]
        with patch("llm.client.call_ollama", return_value=json.dumps(mj)):
            data, log = generate_machine(
                ex["description"], ex["alphabet"],
                model="qwen2.5-coder:3b", max_retries=2, backend="ollama",
            )
        assert data["states"] == mj["states"]
        assert len(log) == 1
        assert log[0]["outcome"] == "Success"

    def test_offline_backend_retries_on_bad_schema(self):
        ex = EXAMPLES[1]
        good_json = json.dumps(ex["machine_json"])
        bad_json = json.dumps({"machine_type": "DFA"})  # missing keys
        with patch("llm.client.call_ollama", side_effect=[bad_json, good_json]):
            data, log = generate_machine(
                ex["description"], ex["alphabet"],
                model="qwen2.5-coder:3b", max_retries=2, backend="ollama",
            )
        assert data["finals"] == ex["machine_json"]["finals"]
        assert len(log) == 2
        assert log[0]["outcome"] == "Validation error"
        assert log[1]["outcome"] == "Success"
