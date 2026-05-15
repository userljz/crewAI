from __future__ import annotations

from pathlib import Path
from typing import Any

from persona_prompt_flywheel.io_utils import read_jsonl, read_yaml, write_yaml


def merge_human_review_labels(labels_path: str | Path, rules_path: str | Path) -> dict[str, Any]:
    label_config_file = Path(rules_path)
    label_config = read_yaml(label_config_file)
    records = read_jsonl(labels_path)
    if not isinstance(label_config.get("labels"), dict):
        label_config["labels"] = {}
    labels = label_config["labels"]
    merged = 0

    for record in records:
        label_update = record.get("label_update") or {}
        label_name = str(label_update.get("label") or record.get("resolved_label") or "")
        if not label_name:
            continue

        existing = labels.get(label_name)
        spec = dict(existing) if isinstance(existing, dict) else {"description": str(existing or "")}
        original = dict(spec)
        if label_update.get("description"):
            spec["description"] = label_update["description"]
        elif not spec.get("description"):
            spec["description"] = f"Human resolved trace {record.get('trace_id')} as {label_name}."
        for field in ("training_target_positive", "needs_human_review"):
            if field in label_update:
                spec[field] = label_update[field]
        labels[label_name] = spec
        if spec != original:
            merged += 1

    label_config["version"] = int(label_config.get("version", 1)) + (1 if merged else 0)
    write_yaml(label_config_file, label_config)
    return {"merged_labels": merged, "label_config_path": str(label_config_file), "version": label_config["version"]}

