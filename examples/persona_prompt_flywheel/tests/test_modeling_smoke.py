from __future__ import annotations

import argparse
from pathlib import Path

from run_flywheel import run_pipeline
from persona_prompt_flywheel.modeling_tools import build_design_matrix
from persona_prompt_flywheel.prompt_patch_generation import _patch_text_from_api_response, _quality_gate_reason


def test_full_mock_pipeline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        input=root / "data" / "sample_interactions.tsv",
        output=tmp_path,
        config=root / "config" / "pipeline_config.yaml",
        rules=root / "config" / "feedback_labeling_rules.yaml",
        feature_config=root / "config" / "feature_config.yaml",
        modeling_config=root / "config" / "modeling_config.yaml",
        mock_api=True,
        merge_human_labels=None,
    )

    summary = run_pipeline(args)

    assert summary["total_rows"] == 5
    assert summary["neutral_continuation_no_feedback"] >= 1
    assert summary["human_review_queue_count"] == 1
    assert summary["agent_debug_output_dir"] == str(tmp_path / "04_debug_bundles" / "output")
    assert summary["model_insights_path"] == str(tmp_path / "01_core_results" / "model_insights.md")
    assert (tmp_path / "01_core_results" / "model_report.json").exists()
    assert (tmp_path / "01_core_results" / "model_insights.json").exists()
    assert (tmp_path / "01_core_results" / "model_insights.md").exists()
    assert (tmp_path / "01_core_results" / "prompt_patches.jsonl").exists()
    assert (tmp_path / "01_core_results" / "run_summary.md").exists()
    assert (tmp_path / "02_stage_artifacts" / "feature_matrix.jsonl").exists()
    for agent_name in [
        "feedback_labeling_agent",
        "feature_discovery_agent",
        "modeling_code_agent",
        "prompt_patch_generator_agent",
    ]:
        assert (tmp_path / "04_debug_bundles" / "output" / agent_name / "output_manifest.json").exists()
        assert (tmp_path / "03_agent_handoffs" / f"{agent_name}_handoff.json").exists()
        assert (tmp_path / "04_debug_bundles" / "output" / agent_name / f"{agent_name}_handoff.md").exists()
    assert (tmp_path / "04_debug_bundles" / "output" / "modeling_code_agent" / "model_insights.md").exists()


def test_interaction_terms_respect_min_category_count(tmp_path: Path) -> None:
    rows = [
        {"trace_id": "a", "score": 1, "segment": "rare", "training_target_positive": 1},
        {"trace_id": "b", "score": 2, "segment": "rare", "training_target_positive": 0},
        {"trace_id": "c", "score": 3, "segment": "other", "training_target_positive": 1},
    ]
    plan = {
        "numeric_features": ["score"],
        "categorical_features": ["segment"],
        "interaction_terms_enabled": True,
        "min_category_count_for_interaction": 20,
        "max_interaction_features": 100,
    }

    _x, _y, feature_names, _metadata = build_design_matrix(rows, plan, tmp_path / "design_matrix.csv")

    assert not any(" x " in name for name in feature_names)


def test_patch_text_from_api_response_accepts_nested_dict() -> None:
    text = _patch_text_from_api_response('{"patch_text": {"instruction": "先给结论。"}}', "fallback")

    assert text == "先给结论。"


def test_patch_quality_gate_blocks_low_precision_segment_patches() -> None:
    reason = _quality_gate_reason(
        {"metrics": {"precision": 0.19}},
        {"quality_gates": {"min_precision_for_segment_patches": 0.30}},
    )

    assert "below required" in str(reason)

