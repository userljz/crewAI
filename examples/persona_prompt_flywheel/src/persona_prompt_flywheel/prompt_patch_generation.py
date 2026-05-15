from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.api_client import UnifiedAgentAPI
from persona_prompt_flywheel.io_utils import read_yaml, write_json, write_jsonl
from persona_prompt_flywheel.registry import PromptPatchRegistry


def generate_prompt_patches(
    *,
    segment_insights: dict[str, Any],
    model_report: dict[str, Any],
    feature_catalog: dict[str, Any],
    candidate_features: list[dict[str, Any]],
    api: UnifiedAgentAPI,
    output_dir: str | Path,
    modeling_config_path: str | Path,
    llm_execution_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    patch_config = read_yaml(modeling_config_path).get("prompt_patch_generation", {})
    status_map = patch_config.get("status_by_model_status", {})
    status = status_map.get(model_report.get("status"), status_map.get("default"))
    insights = segment_insights.get("segment_insights", [])
    max_workers = _max_workers(llm_execution_config)
    if not insights:
        patches = [_fallback_patch(model_report, feature_catalog, patch_config)]
    elif max_workers == 1 or len(insights) <= 1:
        patches = [
            _patch_for_insight(
                insight, segment_insights, model_report, feature_catalog, candidate_features, api, status, patch_config
            )
            for insight in insights
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            patches = list(
                executor.map(
                    lambda insight: _patch_for_insight(
                        insight,
                        segment_insights,
                        model_report,
                        feature_catalog,
                        candidate_features,
                        api,
                        status,
                        patch_config,
                    ),
                    insights,
                )
            )

    write_jsonl(output / "prompt_patches.jsonl", patches)
    write_json(output / "prompt_patch_registry.json", {"patches": patches})
    _write_markdown(output / "prompt_patches.md", patches)
    _write_runtime_example(output / "runtime_prompt_example.md", patches, patch_config)
    return patches


def _max_workers(llm_execution_config: dict[str, Any] | None = None) -> int:
    raw_value = (llm_execution_config or {}).get("max_workers", 1)
    return max(1, int(raw_value or 1))


def _patch_for_insight(
    insight: dict[str, Any],
    all_insights: dict[str, Any],
    report: dict[str, Any],
    catalog: dict[str, Any],
    candidate_features: list[dict[str, Any]],
    api: UnifiedAgentAPI,
    status: str,
    patch_config: dict[str, Any],
) -> dict[str, Any]:
    segment = insight.get("segment", "unknown")
    intent = insight.get("profile_intent") or _intent_from_segment(segment)
    positive = [driver["feature"] for driver in _drivers_for(insight, all_insights, "positive_drivers")]
    negative = [driver["feature"] for driver in _drivers_for(insight, all_insights, "negative_drivers")]
    default_patch = _build_patch_text(patch_config)
    api_response = api.generate(
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": patch_config.get("task", "generate_prompt_patch_from_model_insight"),
                        "insight": insight,
                        "positive_drivers": positive,
                        "negative_drivers": negative,
                        "patch_config": patch_config,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        response_format="json",
        metadata={"operation": "prompt_patch_generation", "default_patch": default_patch},
    )
    return {
        "patch_id": _patch_id(f"patch_{segment}", patch_config),
        "profile_intent": intent,
        "profile_composite": segment,
        "trigger_condition": f"当 profile_composite={segment} 且 intention_code={intent} 时触发",
        "prompt_patch": json.loads(api_response).get("patch_text", default_patch),
        "do_rules": _configured_rules(patch_config.get("do_rules", {})),
        "dont_rules": _configured_rules(patch_config.get("dont_rules", {})),
        "evidence": {
            "sample_size": insight.get("sample_size", 0),
            "positive_drivers": positive,
            "negative_drivers": negative,
            "model_status": report.get("status"),
            "model_metrics": report.get("metrics", {}),
            "excluded_label_counts": report.get("excluded_label_counts", {}),
            "raw_is_positive_feedback_used": False,
            "candidate_features": [item.get("feature_name") for item in candidate_features],
        },
        "confidence": _patch_confidence(report, insight, patch_config),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_report": patch_config.get("source_model_report"),
        "labeling_rule_version": 1,
        "feature_catalog_version": catalog.get("version", 1),
    }


def _fallback_patch(report: dict[str, Any], catalog: dict[str, Any], patch_config: dict[str, Any]) -> dict[str, Any]:
    fallback = patch_config.get("fallback", {})
    return {
        "patch_id": _patch_id(str(fallback.get("patch_id_prefix")), patch_config),
        "profile_intent": fallback.get("profile_intent"),
        "profile_composite": fallback.get("profile_composite"),
        "trigger_condition": fallback.get("trigger_condition"),
        "prompt_patch": fallback.get("prompt_patch"),
        "do_rules": fallback.get("do_rules", []),
        "dont_rules": fallback.get("dont_rules", []),
        "evidence": {"model_status": report.get("status"), "raw_is_positive_feedback_used": False},
        "confidence": fallback.get("confidence"),
        "status": fallback.get("status"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_report": patch_config.get("source_model_report"),
        "labeling_rule_version": 1,
        "feature_catalog_version": catalog.get("version", 1),
    }


def _drivers_for(insight: dict[str, Any], all_insights: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return insight.get(key) or all_insights.get("global_insights", {}).get(key, [])


def _build_patch_text(patch_config: dict[str, Any]) -> str:
    text_config = patch_config.get("patch_text", {})
    return str(text_config.get("base", ""))


def _configured_rules(rule_config: dict[str, Any]) -> list[str]:
    rules = [str(item) for item in rule_config.get("base", [])]
    return [rule for rule in rules if rule]


def _patch_confidence(report: dict[str, Any], insight: dict[str, Any], patch_config: dict[str, Any]) -> float:
    config = patch_config.get("confidence_by_model_status", {})
    value = config.get(report.get("status"), config.get("default", 0.0))
    if isinstance(value, dict):
        return float(value.get("skipped" if insight.get("skipped") else "default", 0.0))
    return float(value)


def _patch_id(prefix: str, patch_config: dict[str, Any]) -> str:
    date_format = str(patch_config.get("patch_id_date_format", "%Y%m%d"))
    version = str(patch_config.get("patch_version", "v1"))
    return f"{prefix}_{datetime.now(timezone.utc).strftime(date_format)}_{version}"


def _intent_from_segment(segment: str) -> str:
    return segment.split("_", 1)[0] if segment and segment != "unknown" else "unknown"


def _write_markdown(path: str | Path, patches: list[dict[str, Any]]) -> None:
    lines = ["# Prompt Patch Candidates", ""]
    for patch in patches:
        lines += [
            f"## {patch['patch_id']}",
            "",
            f"- status: {patch['status']}",
            f"- confidence: {patch['confidence']}",
            f"- trigger: {patch['trigger_condition']}",
            "",
            patch["prompt_patch"],
            "",
        ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_runtime_example(path: str | Path, patches: list[dict[str, Any]], patch_config: dict[str, Any]) -> None:
    registry = PromptPatchRegistry.from_patches(patches)
    first = patches[0]
    patch = registry.select_best_patch(
        {"profile_composite": first["profile_composite"]},
        intent=first["profile_intent"],
    )
    runtime_prompt = registry.render_runtime_prompt(str(patch_config.get("runtime_example", {}).get("base_prompt", "")), patch)
    Path(path).write_text(f"# Runtime Prompt Example\n\n{runtime_prompt}\n", encoding="utf-8")

