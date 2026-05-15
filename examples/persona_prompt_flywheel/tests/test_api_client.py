from __future__ import annotations

from persona_prompt_flywheel.api_client import _normalize_model_id, _openrouter_content


def test_openrouter_content_reads_chat_completion_payload() -> None:
    payload = {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    assert _openrouter_content(payload) == "{\"ok\": true}"


def test_openrouter_model_prefix_is_normalized() -> None:
    assert _normalize_model_id("openrouter/deepseek/deepseek-v3.2") == "deepseek/deepseek-v3.2"

