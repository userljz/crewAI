from __future__ import annotations

from typing import Any


def profile_fields(config: dict[str, Any]) -> list[str]:
    fields = config.get("profile", {}).get("output_fields", [])
    if not isinstance(fields, list) or not fields:
        raise ValueError("profile.output_fields must be a non-empty list.")
    return [str(field) for field in fields]


def attach_profiles(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    profile_columns = config.get("profile_columns") or []
    profile_config = config.get("profile", {})
    unknown = str(profile_config.get("unknown_value", "unknown"))
    use_real_profile = profile_columns and all(
        column in records[0] for column in profile_columns
    ) if records else False

    profiled: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if use_real_profile:
            for column in profile_columns:
                item[f"profile_{column}"] = _safe_segment(record.get(column, unknown), unknown)
            item["profile_intent"] = _safe_segment(record.get(profile_config.get("intent_source_column"), unknown), unknown)
            item["profile_input_channel"] = _safe_segment(record.get(profile_config.get("input_channel_source_column"), unknown), unknown)
            item["profile_session_stage"] = _bucket(
                record.get(profile_config.get("session_stage_source_column")),
                profile_config.get("session_stage_bins", []),
                unknown,
            )
            item["profile_interaction_band"] = _bucket(
                record.get(profile_config.get("interaction_count_source_column")),
                profile_config.get("interaction_band_bins", []),
                unknown,
                boundary_key="max_exclusive",
            )
            item["profile_feedback_style"] = str(profile_config.get("default_feedback_style", unknown))
            item["profile_composite"] = "_".join(
                _safe_segment(record.get(column, unknown), unknown) for column in profile_columns
            )
        else:
            item["profile_intent"] = _safe_segment(record.get(profile_config.get("intent_source_column"), unknown), unknown)
            item["profile_input_channel"] = _safe_segment(record.get(profile_config.get("input_channel_source_column"), unknown), unknown)
            item["profile_session_stage"] = _bucket(
                record.get(profile_config.get("session_stage_source_column")),
                profile_config.get("session_stage_bins", []),
                unknown,
            )
            item["profile_interaction_band"] = _bucket(
                record.get(profile_config.get("interaction_count_source_column")),
                profile_config.get("interaction_band_bins", []),
                unknown,
                boundary_key="max_exclusive",
            )
            item["profile_feedback_style"] = str(profile_config.get("default_feedback_style", unknown))
            composite_fields = profile_config.get("fallback_composite_fields", [])
            item["profile_composite"] = "_".join(
                str(item.get(field, unknown)) for field in composite_fields
            )
        profiled.append(item)
    return profiled


def _bucket(value: Any, bins: Any, unknown: str, *, boundary_key: str = "max_inclusive") -> str:
    number = _to_int(value, default=0)
    if not isinstance(bins, list) or not bins:
        return unknown
    for item in bins:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", unknown))
        if boundary_key in item and number < int(item[boundary_key]):
            return label
        if "max_inclusive" in item and number <= int(item["max_inclusive"]):
            return label
        if "max_exclusive" not in item and "max_inclusive" not in item:
            return label
    return unknown


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _safe_segment(value: Any, unknown: str) -> str:
    text = str(value or unknown).strip()
    return "".join(char if char.isalnum() else "_" for char in text) or unknown

