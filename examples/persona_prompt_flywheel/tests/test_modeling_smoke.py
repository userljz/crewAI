from __future__ import annotations

import argparse
from pathlib import Path

from run_flywheel import run_pipeline


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

