from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
from typing import Any

from persona_prompt_flywheel.api_client import UnifiedAgentAPI
from persona_prompt_flywheel.io_utils import read_yaml, write_json, write_jsonl, write_yaml


def extract_features(
    labeled: list[dict[str, Any]],
    *,
    api: UnifiedAgentAPI,
    feature_config_path: str | Path,
    output_dir: str | Path,
    llm_execution_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_config = read_yaml(feature_config_path)
    output = Path(output_dir)
    max_workers = _max_workers(feature_config, llm_execution_config)
    print(f"[progress] feature_extraction: records={len(labeled)} max_workers={max_workers} mode=llm_semantic_extraction", flush=True)
    if max_workers == 1 or len(labeled) <= 1:
        results = [_extract_feature_row(record, api, feature_config, max_workers) for record in labeled]
    else:
        results = [None] * len(labeled)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_extract_feature_row, record, api, feature_config, max_workers): index
                for index, record in enumerate(labeled)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if completed == len(labeled) or completed % 250 == 0:
                    print(f"[progress] feature_extraction: completed {completed}/{len(labeled)}", flush=True)
    results = [result for result in results if result is not None]
    matrix = [row for row, _audit in results]
    audit = [audit_row for _row, audit_row in results]

    catalog = {
        "version": 1,
        "objective_features": _all_objective_features(feature_config),
        "subjective_features": feature_config.get("subjective_features", {}),
    }
    write_yaml(output / "feature_config.resolved.yaml", feature_config)
    write_json(output / "feature_catalog.json", catalog)
    write_jsonl(output / "feature_matrix.jsonl", matrix)
    _write_feature_matrix_csv(output / "feature_matrix.csv", matrix, feature_config)
    write_jsonl(output / "feature_extraction_audit.jsonl", audit)
    return matrix, catalog


def _extract_feature_row(
    record: dict[str, Any],
    api: UnifiedAgentAPI,
    feature_config: dict[str, Any],
    max_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = str(record.get("response_content") or "")
    features, evidence, confidence, source = _objective_features(response, api, feature_config)
    deterministic_features, deterministic_evidence, deterministic_confidence, deterministic_source = (
        _deterministic_objective_features(response, feature_config)
    )
    features = {**deterministic_features, **features}
    evidence = {**deterministic_evidence, **evidence}
    confidence = {**deterministic_confidence, **confidence}
    source = {**deterministic_source, **source}
    subjective = _subjective_features(response, api, feature_config)
    for name, payload in subjective.items():
        features[name] = payload["score"]
        evidence[name] = payload["evidence"]
        confidence[name] = payload["confidence"]
        source[name] = "llm_rubric_v1"
    is_unreadable_response = _is_unreadable_response(features, feature_config)
    row = {
        **{field: record.get(field) for field in feature_config.get("feature_matrix_record_fields", [])},
        "features": features,
        "feature_evidence": evidence,
        "feature_confidence": confidence,
        "feature_source": source,
        "response_content_quality": "unreadable_or_system_noise" if is_unreadable_response else "readable",
    }
    if is_unreadable_response:
        row["usable_for_training"] = False
        row["training_exclusion_reason"] = "unreadable_response_content"
    audit = {
        "trace_id": row["trace_id"],
        "objective_feature_count": len(_all_objective_features(feature_config)),
        "subjective_feature_count": len(feature_config.get("subjective_features", {})),
        "llm_adapter": type(api).__name__,
        "max_workers": max_workers,
        "feature_extraction_mode": "llm_semantic_extraction",
        "response_content_quality": row["response_content_quality"],
    }
    return row, audit


def _max_workers(feature_config: dict[str, Any], llm_execution_config: dict[str, Any] | None = None) -> int:
    merged_config = {**(llm_execution_config or {}), **feature_config.get("llm_execution", {})}
    raw_value = merged_config.get("max_workers", 1)
    return max(1, int(raw_value or 1))


def _deterministic_objective_features(
    response: str,
    feature_config: dict[str, Any],
) -> tuple[dict[str, float | int], dict[str, str], dict[str, float], dict[str, str]]:
    features: dict[str, float | int] = {}
    evidence: dict[str, str] = {}
    confidence: dict[str, float] = {}
    source: dict[str, str] = {}
    extractors = {
        "char_count": lambda text: len(text),
        "token_or_word_count": _token_or_word_count,
        "paragraph_count": _paragraph_count,
        "question_count": _question_count,
        "numbered_list_count": _numbered_list_count,
        "bullet_count": _bullet_count,
        "markdown_bold_count": _markdown_bold_count,
        "link_count": _link_count,
        "code_block_count": _code_block_count,
        "has_table": _has_table,
    }
    for name, spec in feature_config.get("deterministic_objective_features", {}).items():
        extractor_name = str(spec.get("extractor", ""))
        extractor = extractors.get(extractor_name)
        if extractor is None:
            raise ValueError(f"Unknown deterministic objective extractor: {extractor_name}")
        value = extractor(response)
        features[name] = _coerce_feature_value(value, spec)
        evidence[name] = f"{extractor_name}={features[name]}"
        confidence[name] = 1.0
        source[name] = "deterministic_python"
    return features, evidence, confidence, source


def _objective_features(
    response: str,
    api: UnifiedAgentAPI,
    feature_config: dict[str, Any],
) -> tuple[dict[str, float | int], dict[str, str], dict[str, float], dict[str, str]]:
    prompt_config = feature_config.get("objective_extraction_prompt", {})
    response_text = api.generate(
        [
            {"role": "system", "content": str(prompt_config.get("system", ""))},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": prompt_config.get("task"),
                        "output_contract": prompt_config.get("output_contract"),
                        "objective_features": feature_config.get("objective_features", {}),
                        "response_content": response,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.0,
        response_format="json",
        metadata={
            "operation": "objective_feature_extraction",
            "response_content": response,
            "objective_features": feature_config.get("objective_features", {}),
        },
    )
    payload = json.loads(response_text)
    features: dict[str, float | int] = {}
    evidence: dict[str, str] = {}
    confidence: dict[str, float] = {}
    source: dict[str, str] = {}
    for name, spec in feature_config.get("objective_features", {}).items():
        value = payload.get(name, {}) if isinstance(payload, dict) else {}
        if isinstance(value, dict):
            features[name] = _coerce_feature_value(value.get("value", 0), spec)
            evidence[name] = str(value.get("evidence", ""))
            confidence[name] = float(value.get("confidence", 0.5))
        else:
            features[name] = _coerce_feature_value(value, spec)
            evidence[name] = ""
            confidence[name] = 0.5
        source[name] = "llm_objective_extractor"
    return features, evidence, confidence, source


def _all_objective_features(feature_config: dict[str, Any]) -> dict[str, Any]:
    return {
        **feature_config.get("deterministic_objective_features", {}),
        **feature_config.get("objective_features", {}),
    }


def _is_unreadable_response(features: dict[str, Any], feature_config: dict[str, Any]) -> bool:
    llm_feature_names = [
        *feature_config.get("objective_features", {}),
        *feature_config.get("subjective_features", {}),
    ]
    if not llm_feature_names:
        return False
    return all(_to_float(features.get(name)) < 0 for name in llm_feature_names)


def _subjective_features(
    response: str,
    api: UnifiedAgentAPI,
    feature_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    prompt_config = feature_config.get("subjective_scoring_prompt", {})
    response_text = api.generate(
        [
            {"role": "system", "content": str(prompt_config.get("system", ""))},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": prompt_config.get("task"),
                        "score_range": prompt_config.get("score_range"),
                        "subjective_features": feature_config.get("subjective_features", {}),
                        "response_content": response,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.0,
        response_format="json",
        metadata={
            "operation": "subjective_feature_scoring",
            "response_content": response,
            "subjective_features": feature_config.get("subjective_features", {}),
        },
    )
    return json.loads(response_text)


def _coerce_feature_value(value: Any, spec: dict[str, Any]) -> float | int:
    number = float(value or 0)
    if number < 0:
        return -1
    if spec.get("type") == "binary":
        return 1 if number >= 0.5 else 0
    return int(number) if number.is_integer() else number


def _token_or_word_count(text: str) -> int:
    count = 0
    in_ascii_token = False
    for char in text:
        if char.isascii() and char.isalnum():
            if not in_ascii_token:
                count += 1
                in_ascii_token = True
            continue
        in_ascii_token = False
        if _is_cjk(char):
            count += 1
    return count


def _paragraph_count(text: str) -> int:
    return len([block for block in re.split(r"\n\s*\n+", text.strip()) if block.strip()])


def _question_count(text: str) -> int:
    return len(re.findall(r"[?？]", text))


def _numbered_list_count(text: str) -> int:
    pattern = re.compile(r"^\s*(?:\d+[.)、]|[一二三四五六七八九十]+[、.、]|第[一二三四五六七八九十\d]+步[:：])", re.MULTILINE)
    return len(pattern.findall(text))


def _bullet_count(text: str) -> int:
    return len(re.findall(r"^\s*(?:[-*+•]|✅)\s+", text, flags=re.MULTILINE))


def _markdown_bold_count(text: str) -> int:
    return len(re.findall(r"\*\*[^*\n][\s\S]*?[^*\n]\*\*", text))


def _link_count(text: str) -> int:
    markdown_links = re.findall(r"\[[^\]]+\]\((?:https?://|www\.)[^)]+\)", text)
    without_markdown_links = re.sub(r"\[[^\]]+\]\((?:https?://|www\.)[^)]+\)", "", text)
    bare_links = re.findall(r"(?<![\w/])(?:https?://|www\.)\S+", without_markdown_links)
    return len(markdown_links) + len(bare_links)


def _code_block_count(text: str) -> int:
    return text.count("```") // 2


def _has_table(text: str) -> int:
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if "|" not in line or "|" not in lines[index + 1]:
            continue
        if re.search(r"\|\s*:?-{3,}:?\s*(?:\||$)", lines[index + 1]):
            return 1
    return 0


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_feature_matrix_csv(path: Path, matrix: list[dict[str, Any]], feature_config: dict[str, Any]) -> None:
    feature_names = sorted({name for row in matrix for name in row["features"]})
    fieldnames = [*feature_config.get("feature_matrix_csv_fields", []), *feature_names]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix:
            flat = {key: row.get(key) for key in fieldnames}
            flat.update(row["features"])
            writer.writerow(flat)

