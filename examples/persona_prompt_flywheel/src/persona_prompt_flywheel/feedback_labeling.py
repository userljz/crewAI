from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.api_client import UnifiedAgentAPI
from persona_prompt_flywheel.io_utils import read_yaml, write_csv, write_jsonl, write_yaml
from persona_prompt_flywheel.profile_builder import profile_fields


def label_feedback(
    records: list[dict[str, Any]],
    *,
    api: UnifiedAgentAPI,
    config: dict[str, Any],
    rules_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_config = read_yaml(rules_path)
    label_specs = _label_specs(label_config)
    labeling_config = config.get("labeling", {})
    decision_policy = _decision_policy(label_config, labeling_config)
    profile_output_fields = profile_fields(config)
    labeled: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    max_workers = _max_workers(config)

    print(f"[progress] feedback_labeling: records={len(records)} max_workers={max_workers}", flush=True)
    if max_workers == 1 or len(records) <= 1:
        results = [
            _label_feedback_record(
                record,
                api,
                label_config,
                label_specs,
                decision_policy,
                labeling_config,
                profile_output_fields,
            )
            for record in records
        ]
    else:
        results = [None] * len(records)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _label_feedback_record,
                        record,
                        api,
                        label_config,
                        label_specs,
                        decision_policy,
                        labeling_config,
                        profile_output_fields,
                ): index
                for index, record in enumerate(records)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if completed == len(records) or completed % 250 == 0:
                    print(f"[progress] feedback_labeling: completed {completed}/{len(records)}", flush=True)
    results = [result for result in results if result is not None]
    labeled = [output for output, _audit in results]
    audit_rows = [audit for _output, audit in results]
    review_queue = [
        _human_review_record(record, output, label_specs)
        for record, output in zip(records, labeled, strict=False)
        if output["needs_human_review"]
    ]

    output = Path(output_dir)
    write_jsonl(output / "labeled_feedback.jsonl", labeled)
    write_jsonl(output / "human_review_queue.jsonl", review_queue)
    write_jsonl(output / "feedback_labeling_audit.jsonl", audit_rows)
    write_yaml(output / "feedback_labeling_rules.snapshot.yaml", label_config)
    write_user_feedback_stats(output / "user_feedback_stats.csv", labeled)
    return labeled, review_queue


def _label_feedback_record(
    record: dict[str, Any],
    api: UnifiedAgentAPI,
    label_config: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
    decision_policy: dict[str, Any],
    labeling_config: dict[str, Any],
    profile_output_fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _api_decision(record, api, label_config, label_specs, decision_policy)
    label_source = "llm_label"
    output = _build_labeled_record(record, decision, label_specs, labeling_config, label_source, profile_output_fields)
    audit = {
        "trace_id": record.get("trace_id"),
        "label_source": label_source,
        "raw_is_positive_feedback": record.get("raw_is_positive_feedback"),
        "raw_label_used_for_decision": False,
        "semantic_feedback_label": output["semantic_feedback_label"],
        "reason": output["feedback_evidence"],
    }
    return output, audit


def _label_specs(label_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_labels = label_config.get("labels")
    if not isinstance(raw_labels, dict) or not raw_labels:
        raise ValueError("feedback labeling config must define a non-empty labels mapping.")

    specs: dict[str, dict[str, Any]] = {}
    for label, raw_spec in raw_labels.items():
        if isinstance(raw_spec, str):
            spec = {"description": raw_spec}
        elif isinstance(raw_spec, dict):
            spec = dict(raw_spec)
        else:
            raise ValueError(f"label {label!r} must be a string description or mapping.")
        if not str(spec.get("description", "")).strip():
            raise ValueError(f"label {label!r} must define a description.")
        specs[str(label)] = spec
    return specs


def _decision_policy(label_config: dict[str, Any], labeling_config: dict[str, Any]) -> dict[str, Any]:
    policy = dict(labeling_config)
    yaml_policy = label_config.get("decision_policy", {})
    if isinstance(yaml_policy, dict):
        policy.update(yaml_policy)
    return policy


def _feedback_labeling_payload(
    record: dict[str, Any],
    label_config: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
    decision_policy: dict[str, Any],
) -> dict[str, Any]:
    allowed_labels = list(label_specs)
    prompt_config = label_config.get("prompt", {})
    output_contract = dict(prompt_config.get("output_contract", {})) if isinstance(prompt_config, dict) else {}
    output_contract["semantic_feedback_label"] = allowed_labels
    return {
        "task": prompt_config.get("task"),
        "label_config": {
            "version": label_config.get("version"),
            "labels": label_specs,
        },
        "decision_policy": {
            "ignore_raw_is_positive_feedback": decision_policy.get("ignore_raw_is_positive_feedback", True),
            "confidence_threshold_for_auto_label": decision_policy.get("confidence_threshold_for_auto_label", 0.70),
            "human_review_enabled": decision_policy.get("human_review_enabled", True),
            "allowed_labels": allowed_labels,
            "uncertainty_label": _resolve_uncertainty_label(label_specs, decision_policy),
            "when_confidence_below_threshold": prompt_config.get("low_confidence_instruction"),
        },
        "record": {
            "trace_id": record.get("trace_id"),
            "user_query": record.get("user_query"),
            "response_content": record.get("response_content"),
            "subsequent_user_query": record.get("subsequent_user_query"),
            "raw_is_positive_feedback": record.get("raw_is_positive_feedback"),
        },
        "output_contract": output_contract,
    }


def write_user_feedback_stats(path: str | Path, labeled: list[dict[str, Any]]) -> None:
    by_user: dict[str, Counter[str]] = defaultdict(Counter)
    label_names = sorted({str(row.get("semantic_feedback_label")) for row in labeled})
    for row in labeled:
        user_key = row.get("app_user_id_hash") or row.get("user_id_hash") or "unknown"
        label = str(row.get("semantic_feedback_label"))
        by_user[user_key]["total_rows"] += 1
        if row.get("has_user_interaction_after_response"):
            by_user[user_key]["valid_interaction_rows"] += 1
        if row.get("needs_human_review"):
            by_user[user_key]["human_review_rows"] += 1
        by_user[user_key][label] += 1
        if row.get("training_target_positive") == 1:
            by_user[user_key]["positive_rows"] += 1
        elif row.get("usable_for_training"):
            by_user[user_key]["non_positive_rows"] += 1

    rows: list[dict[str, Any]] = []
    for user_key, counts in by_user.items():
        valid = counts["valid_interaction_rows"]
        total = counts["total_rows"]
        row = {
            "user_key_hash": user_key,
            "total_rows": total,
            "valid_interaction_rows": valid,
            "positive_rows": counts["positive_rows"],
            "non_positive_rows": counts["non_positive_rows"],
            "valid_interaction_rate": round(valid / total, 4) if total else 0,
            "positive_rate": round(counts["positive_rows"] / valid, 4) if valid else 0,
            "human_review_rate": round(counts["human_review_rows"] / valid, 4) if valid else 0,
        }
        for label in label_names:
            row[f"{label}_rows"] = counts[label]
        rows.append(row)
    write_csv(path, rows)


def _max_workers(config: dict[str, Any]) -> int:
    raw_value = config.get("llm_execution", {}).get("max_workers", 1)
    return max(1, int(raw_value or 1))


def _resolve_uncertainty_label(label_specs: dict[str, dict[str, Any]], labeling_config: dict[str, Any]) -> str:
    configured = str(labeling_config.get("uncertainty_label", ""))
    if configured in label_specs:
        return configured
    for label, spec in label_specs.items():
        if spec.get("needs_human_review"):
            return label
    raise ValueError("No uncertainty label configured. Set labeling.uncertainty_label or needs_human_review: true in labels.")


def _api_decision(
    record: dict[str, Any],
    api: UnifiedAgentAPI,
    label_config: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
    decision_policy: dict[str, Any],
) -> dict[str, Any]:
    prompt = _feedback_labeling_payload(record, label_config, label_specs, decision_policy)
    response = api.generate(
        [
            {"role": "system", "content": _system_prompt(label_config)},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.0,
        response_format="json",
        metadata={
            "operation": "feedback_labeling",
            "trace_id": record.get("trace_id"),
            "subsequent_user_query": record.get("subsequent_user_query", ""),
        },
    )
    data = json.loads(response)
    label = str(data["semantic_feedback_label"])
    if label not in label_specs:
        label = _resolve_uncertainty_label(label_specs, decision_policy)
        data["needs_human_review"] = True
        data["uncertainty_reason"] = f"LLM returned unsupported label: {data.get('semantic_feedback_label')}"
    return _decision(
        label,
        float(data.get("confidence", 0.5)),
        bool(data.get("needs_human_review", False)),
        str(data.get("evidence") or ""),
        uncertainty_reason=str(data.get("uncertainty_reason") or ""),
    )


def _system_prompt(label_config: dict[str, Any]) -> str:
    prompt_config = label_config.get("prompt", {})
    if not isinstance(prompt_config, dict) or not str(prompt_config.get("system", "")).strip():
        raise ValueError("feedback labeling config must define prompt.system.")
    return str(prompt_config["system"])


def _build_labeled_record(
    record: dict[str, Any],
    decision: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
    labeling_config: dict[str, Any],
    label_source: str,
    profile_output_fields: list[str],
) -> dict[str, Any]:
    label = decision["semantic_feedback_label"]
    target = _training_target(label, label_specs)
    needs_human_review = bool(decision["needs_human_review"] or label_specs.get(label, {}).get("needs_human_review"))
    usable = target in {0, 1} and not needs_human_review
    output = {
        "trace_id": record.get("trace_id"),
        "chat_id": record.get("chat_id"),
        "app_user_id_hash": _hash_id(record.get("app_user_id")),
        "user_id_hash": _hash_id(record.get("user_id")),
        "user_query": record.get("user_query"),
        "response_content": record.get("response_content"),
        "subsequent_user_query": record.get("subsequent_user_query"),
        "has_user_interaction_after_response": bool(str(record.get("subsequent_user_query", "")).strip()),
        "raw_is_positive_feedback": record.get("raw_is_positive_feedback"),
        "raw_label_used_for_decision": False,
        "semantic_feedback_label": label,
        "feedback_confidence": decision["feedback_confidence"],
        "needs_human_review": needs_human_review,
        "uncertainty_reason": decision["uncertainty_reason"],
        "usable_for_training": usable,
        "training_target_positive": target,
        "training_label_policy": labeling_config.get("training_label_policy", "positive_feedback_observed"),
        "feedback_evidence": decision["feedback_evidence"],
        "label_description": str(label_specs.get(label, {}).get("description", "")),
        "suggested_label_update": _suggested_label_update(label, record, label_specs),
        "label_source": label_source,
        "label_confidence": decision["feedback_confidence"],
    }
    for field in profile_output_fields:
        output[field] = record.get(field, "unknown")
    return output


def _training_target(label: str, label_specs: dict[str, dict[str, Any]]) -> int | None:
    value = label_specs.get(label, {}).get("training_target_positive")
    if value in {0, 1}:
        return int(value)
    if isinstance(value, str) and value in {"0", "1"}:
        return int(value)
    return None


def _human_review_record(
    record: dict[str, Any],
    labeled: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "trace_id": record.get("trace_id"),
        "user_query": record.get("user_query"),
        "response_content": record.get("response_content"),
        "subsequent_user_query": record.get("subsequent_user_query"),
        "candidate_labels": list(label_specs),
        "label_descriptions": {label: spec.get("description") for label, spec in label_specs.items()},
        "uncertainty_reason": labeled["uncertainty_reason"],
        "suggested_label_update": labeled["suggested_label_update"],
    }


def _decision(
    label: str,
    confidence: float,
    needs_human_review: bool,
    evidence: str,
    *,
    uncertainty_reason: str = "",
) -> dict[str, Any]:
    return {
        "semantic_feedback_label": label,
        "feedback_confidence": confidence,
        "needs_human_review": needs_human_review,
        "uncertainty_reason": uncertainty_reason if needs_human_review else "",
        "feedback_evidence": evidence,
    }


def _hash_id(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def _suggested_label_update(
    label: str,
    record: dict[str, Any],
    label_specs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not label_specs.get(label, {}).get("needs_human_review"):
        return None
    return {
        "label_config_path": "config/feedback_labeling_rules.yaml",
        "note": (
            f"人工复核用户后续表达“{str(record.get('subsequent_user_query', ''))[:40]}”后，"
            "如发现稳定的新标签或描述边界，请直接更新 labels 配置。"
        ),
    }

