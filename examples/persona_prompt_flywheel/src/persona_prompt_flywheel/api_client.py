from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import os
from typing import Any
from urllib import request


class UnifiedAgentAPI(ABC):
    """Single adapter used by every LLM-backed agent in this prototype."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


class MockAgentAPI(UnifiedAgentAPI):
    """Deterministic stand-in for the provided production Agent API."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del temperature, response_format, max_tokens
        metadata = metadata or {}
        operation = metadata.get("operation", "generic")
        content = "\n".join(str(message.get("content", "")) for message in messages)

        if operation == "feedback_labeling":
            return json.dumps(self._mock_feedback_label(messages, metadata), ensure_ascii=False)
        if operation == "objective_feature_extraction":
            return json.dumps(self._mock_objective_features(metadata), ensure_ascii=False)
        if operation == "subjective_feature_scoring":
            return json.dumps(self._mock_subjective_scores(metadata), ensure_ascii=False)
        if operation == "prompt_patch_generation":
            return json.dumps(self._mock_prompt_patch(metadata), ensure_ascii=False)
        if operation == "agent_handoff_review":
            return json.dumps(self._mock_agent_handoff(metadata), ensure_ascii=False)

        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        return json.dumps({"mock_response": digest}, ensure_ascii=False)

    def _mock_feedback_label(self, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        payload = self._last_json_payload(messages)
        record = payload.get("record", {}) if isinstance(payload, dict) else {}
        labels = payload.get("label_config", {}).get("labels", {}) if isinstance(payload, dict) else {}
        trace_label = {
            "21d18731177769015138389081300": "neutral_continuation_no_feedback",
            "t-positive-001": "positive_feedback",
            "t-negative-001": "negative_feedback",
            "t-uncertain-001": "uncertain_needs_human_review",
            "t-nosignal-001": "no_signal",
        }
        trace_id = str(record.get("trace_id") or metadata.get("trace_id") or "")
        text = str(record.get("subsequent_user_query") or metadata.get("subsequent_user_query", "")).strip()
        label = trace_label.get(trace_id)
        if label == "uncertain_needs_human_review":
            return {
                "semantic_feedback_label": self._mock_label(labels, label),
                "confidence": 0.55,
                "needs_human_review": True,
                "uncertainty_reason": "Mock LLM marks this trace as ambiguous for review.",
                "evidence": text[:120],
            }
        if label:
            return {
                "semantic_feedback_label": self._mock_label(labels, label),
                "confidence": 0.9,
                "needs_human_review": False,
                "uncertainty_reason": "",
                "evidence": text[:120] or "用户没有后续交互。",
            }
        if not text:
            return {
                "semantic_feedback_label": self._mock_label(labels, "no_signal"),
                "confidence": 0.99,
                "needs_human_review": False,
                "uncertainty_reason": "",
                "evidence": "用户没有后续交互。",
            }
        return {
            "semantic_feedback_label": self._mock_label(labels, "neutral_continuation_no_feedback"),
            "confidence": 0.72,
            "needs_human_review": False,
            "uncertainty_reason": "",
            "evidence": text[:120],
        }

    @staticmethod
    def _mock_label(labels: dict[str, Any], preferred: str) -> str:
        if preferred in labels:
            return preferred
        return next(iter(labels), "")

    @staticmethod
    def _last_json_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
        for message in reversed(messages):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _mock_objective_features(self, metadata: dict[str, Any]) -> dict[str, Any]:
        objective_features = metadata.get("objective_features", {})
        defaults = {
            "response_length_chars": 42,
            "response_length_words_or_tokens": 18,
            "paragraph_count": 1,
            "sentence_count": 2,
            "question_count": 3,
            "clarifying_question_count": 3,
            "numbered_list_count": 0,
            "bullet_count": 0,
            "markdown_bold_count": 0,
            "link_count": 0,
            "code_block_count": 0,
            "has_table": 0,
            "starts_with_acknowledgement": 1,
            "starts_with_direct_answer": 0,
            "contains_medical_safety_note": 1,
            "hedging_phrase_count": 1,
            "action_item_count": 1,
        }
        return {
            name: {
                "value": defaults.get(name, 0),
                "confidence": 0.82,
                "evidence": "mock LLM objective extraction",
            }
            for name in objective_features
        }

    def _mock_subjective_scores(self, metadata: dict[str, Any]) -> dict[str, Any]:
        subjective_features = metadata.get("subjective_features", {})
        scores = {name: 3 for name in subjective_features}
        return {
            name: {
                "score": score,
                "confidence": 0.82,
                "evidence": f"mock rubric score={score}",
            }
            for name, score in scores.items()
        }

    def _mock_prompt_patch(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "patch_text": metadata.get(
                "default_patch",
                "先给出简短方向，再提出必要澄清问题，并保留安全边界。",
            )
        }

    def _mock_agent_handoff(self, metadata: dict[str, Any]) -> dict[str, Any]:
        agent_name = metadata.get("agent_name", "unknown_agent")
        return {
            "agent_name": agent_name,
            "status": "ready_for_next_stage",
            "summary": f"{agent_name} completed controlled tool execution in mock mode.",
            "checks": ["required output files were produced", "handoff summary generated"],
            "warnings": [],
            "next_agent_guidance": "Continue with the next configured task.",
        }


class ConfigurableAgentAPI(UnifiedAgentAPI):
    """HTTP adapter for OpenRouter or a user-provided unified API."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "unified").strip().lower()
        self.model = _normalize_model_id(os.getenv("LLM_MODEL"))
        if self.provider == "openrouter":
            self.endpoint = endpoint or os.getenv(
                "OPENROUTER_API_URL",
                "https://openrouter.ai/api/v1/chat/completions",
            )
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        elif self.provider in {"openai", "openai_compatible", "vllm"}:
            self.endpoint = endpoint or _openai_compatible_endpoint()
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        else:
            self.endpoint = endpoint or os.getenv("UNIFIED_AGENT_API_URL")
            self.api_key = api_key or os.getenv("UNIFIED_AGENT_API_KEY")

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.provider == "openrouter":
            return self._generate_openai_compatible(
                messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
                metadata=metadata,
            )
        if self.provider in {"openai", "openai_compatible", "vllm"}:
            return self._generate_openai_compatible(
                messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
                metadata=metadata,
            )
        return self._generate_unified(
            messages,
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens,
            metadata=metadata,
        )

    def _generate_unified(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.endpoint:
            raise RuntimeError(
                "Real API mode requires UNIFIED_AGENT_API_URL or --mock-api for deterministic local runs."
            )

        body = json.dumps(
            {
                "messages": messages,
                "temperature": temperature,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "metadata": metadata or {},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=60) as response:  # noqa: S310 - endpoint is user configured.
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("text") or payload.get("content") or payload.get("result") or "")

    def _generate_openai_compatible(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del metadata
        if not self.endpoint:
            raise RuntimeError("OpenAI-compatible mode requires an API endpoint.")
        if self.provider == "openrouter" and not self.api_key:
            raise RuntimeError("OpenRouter mode requires OPENROUTER_API_KEY.")
        if not self.model:
            raise RuntimeError("OpenAI-compatible mode requires LLM_MODEL.")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost/persona_prompt_flywheel")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "Persona Prompt Flywheel")
        req = request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:  # noqa: S310 - endpoint is user configured.
            payload = json.loads(response.read().decode("utf-8"))
        return _openrouter_content(payload)


def _openrouter_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content)
            text = first.get("text")
            if text is not None:
                return str(text)
    return str(payload.get("text") or payload.get("content") or payload.get("result") or "")


def _openai_compatible_endpoint() -> str | None:
    endpoint = os.getenv("OPENAI_API_URL") or os.getenv("LLM_API_URL")
    if endpoint:
        return endpoint
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/chat/completions"


def _normalize_model_id(model: str | None) -> str | None:
    if not model:
        return model
    prefix = "openrouter/"
    return model[len(prefix) :] if model.startswith(prefix) else model

