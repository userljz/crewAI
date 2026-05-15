from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.io_utils import write_csv, write_jsonl


def normalize_interactions(
    input_path: str | Path,
    output_dir: str | Path,
    schema_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(input_path)
    output = Path(output_dir)
    required_columns = _required_columns(schema_config)
    delimiter = _detect_delimiter(path, schema_config)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header: {path}")
        field_map = _build_field_map(reader.fieldnames, schema_config, warnings)
        missing = [column for column in required_columns if column not in field_map.values()]
        if missing:
            raise ValueError(f"Input file missing required columns after normalization: {missing}")

        for line_number, raw_row in enumerate(reader, start=2):
            normalized = {canonical: "" for canonical in required_columns}
            passthrough: dict[str, Any] = {}
            for raw_name, value in raw_row.items():
                if raw_name is None:
                    continue
                canonical = field_map.get(raw_name, raw_name)
                if canonical in normalized:
                    normalized[canonical] = "" if value is None else value
                else:
                    passthrough[canonical] = "" if value is None else value
            normalized["source_line_number"] = line_number
            normalized["normalization_warnings"] = list(warnings)
            normalized.update(passthrough)
            rows.append(normalized)

    write_jsonl(output / "normalized_interactions.jsonl", rows)
    write_csv(output / "normalized_interactions.csv", rows)
    return rows, warnings


def _required_columns(schema_config: dict[str, Any]) -> list[str]:
    columns = schema_config.get("required_canonical_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("input_schema.required_canonical_columns must be a non-empty list.")
    return [str(column) for column in columns]


def _detect_delimiter(path: Path, schema_config: dict[str, Any]) -> str:
    by_extension = schema_config.get("delimiter_by_extension", {})
    delimiter = by_extension.get(path.suffix.lower()) if isinstance(by_extension, dict) else None
    if delimiter is not None:
        return str(delimiter)
    sample = path.read_text(encoding="utf-8")[:4096]
    candidates = str(schema_config.get("delimiter_candidates", ",\t"))
    try:
        return csv.Sniffer().sniff(sample, delimiters=candidates).delimiter
    except csv.Error:
        return str(schema_config.get("fallback_delimiter", "\t"))


def _build_field_map(
    fieldnames: list[str],
    schema_config: dict[str, Any],
    warnings: list[str],
) -> dict[str, str]:
    aliases = schema_config.get("column_aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}
    mapping: dict[str, str] = {}
    for raw_name in fieldnames:
        cleaned_name = _clean_header(raw_name)
        canonical = str(aliases.get(cleaned_name, cleaned_name))
        if canonical != raw_name:
            warnings.append(f"column '{raw_name}' normalized to '{canonical}'")
        mapping[raw_name] = canonical
    return mapping


def _clean_header(value: str) -> str:
    return value.lstrip("\ufeff").strip()

