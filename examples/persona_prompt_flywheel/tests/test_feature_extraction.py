from __future__ import annotations

from pathlib import Path

from persona_prompt_flywheel.api_client import MockAgentAPI
from persona_prompt_flywheel.feature_extraction import _coerce_feature_value, extract_features


def test_extract_features_from_single_response(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    labeled = [
        {
            "trace_id": "t1",
            "response_content": "好的，先确认：CRP 是多少？PCT 是多少？低烧变化吗？建议联系医生复诊。",
            "profile_composite": "yiliaoqita_edit_voice_high_mid",
            "profile_intent": "yiliaoqita",
            "profile_input_channel": "edit_voice",
            "profile_session_stage": "mid",
            "profile_interaction_band": "high",
            "profile_feedback_style": "unknown",
            "semantic_feedback_label": "neutral_continuation_no_feedback",
            "usable_for_training": True,
            "training_target_positive": 0,
            "label_source": "auto_rule",
            "label_confidence": 0.75,
        }
    ]

    matrix, catalog = extract_features(
        labeled,
        api=MockAgentAPI(),
        feature_config_path=root / "config" / "feature_config.yaml",
        output_dir=tmp_path,
    )

    assert matrix[0]["features"]["response_length_chars"] == len(labeled[0]["response_content"])
    assert matrix[0]["feature_source"]["response_length_chars"] == "deterministic_python"
    assert matrix[0]["features"]["question_count"] == 3
    assert "clarification_burden_score" in matrix[0]["feature_evidence"]
    assert catalog["objective_features"]
    assert (tmp_path / "feature_matrix.jsonl").exists()


def test_negative_missing_value_is_preserved_for_binary_feature() -> None:
    assert _coerce_feature_value(-1, {"type": "binary"}) == -1

