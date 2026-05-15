from __future__ import annotations

from pathlib import Path

from persona_prompt_flywheel.api_client import MockAgentAPI
from persona_prompt_flywheel.feedback_labeling import label_feedback
from persona_prompt_flywheel.human_review import merge_human_review_labels
from persona_prompt_flywheel.io_utils import read_jsonl, read_yaml, write_jsonl, write_yaml
from persona_prompt_flywheel.profile_builder import attach_profiles
from persona_prompt_flywheel.schema import normalize_interactions


def test_feedback_labeling_ignores_raw_label_and_queues_uncertain(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_yaml(root / "config" / "pipeline_config.yaml")
    records, _warnings = normalize_interactions(root / "data" / "sample_interactions.tsv", tmp_path, config["input_schema"])
    profiled = attach_profiles(records, config)
    labeled, review = label_feedback(
        profiled,
        api=MockAgentAPI(),
        config=config,
        rules_path=root / "config" / "feedback_labeling_rules.yaml",
        output_dir=tmp_path,
    )

    sample = next(row for row in labeled if row["trace_id"] == "21d18731177769015138389081300")
    assert sample["semantic_feedback_label"] == "neutral_continuation_no_feedback"
    assert {row["label_source"] for row in labeled} == {"llm_label"}
    assert sample["raw_label_used_for_decision"] is False
    assert sample["training_target_positive"] == 0
    positive = next(row for row in labeled if row["trace_id"] == "t-positive-001")
    negative = next(row for row in labeled if row["trace_id"] == "t-negative-001")
    assert positive["semantic_feedback_label"] == "positive_feedback"
    assert negative["semantic_feedback_label"] == "negative_feedback"
    assert any(row["semantic_feedback_label"] == "uncertain_needs_human_review" for row in labeled)
    assert review
    assert read_jsonl(tmp_path / "human_review_queue.jsonl")


def test_merge_human_review_label_update(tmp_path: Path) -> None:
    rules_path = tmp_path / "labels.yaml"
    labels_path = tmp_path / "labels.jsonl"
    write_yaml(rules_path, {"version": 1, "labels": {}})
    write_jsonl(
        labels_path,
        [
            {
                "trace_id": "x",
                "resolved_label": "positive_feedback",
                "label_update": {
                    "description": "x means positive",
                    "training_target_positive": 1,
                },
            }
        ],
    )

    result = merge_human_review_labels(labels_path, rules_path)

    label_config = read_yaml(rules_path)
    assert result["merged_labels"] == 1
    assert label_config["version"] == 2
    assert label_config["labels"]["positive_feedback"]["description"] == "x means positive"
    assert label_config["labels"]["positive_feedback"]["training_target_positive"] == 1

