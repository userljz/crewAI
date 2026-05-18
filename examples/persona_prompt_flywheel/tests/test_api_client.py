from __future__ import annotations

from persona_prompt_flywheel.api_client import _normalize_model_id, _openai_compatible_endpoint, _openrouter_content


def test_openrouter_content_reads_chat_completion_payload() -> None:
    payload = {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    assert _openrouter_content(payload) == "{\"ok\": true}"


def test_openrouter_model_prefix_is_normalized() -> None:
    assert _normalize_model_id("openrouter/deepseek/deepseek-v3.2") == "deepseek/deepseek-v3.2"


def test_openai_compatible_endpoint_uses_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1/")

    assert _openai_compatible_endpoint() == "http://127.0.0.1:8000/v1/chat/completions"

