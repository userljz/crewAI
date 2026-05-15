from __future__ import annotations

from pathlib import Path

from persona_prompt_flywheel.io_utils import read_yaml
from persona_prompt_flywheel.schema import normalize_interactions


def test_normalize_tsv_with_markdown_and_raw_label(tmp_path: Path) -> None:
    input_path = Path(__file__).resolve().parents[1] / "data" / "sample_interactions.tsv"
    config = read_yaml(Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml")
    rows, warnings = normalize_interactions(input_path, tmp_path, config["input_schema"])

    assert rows[0]["response_content"].startswith("好的")
    assert rows[0]["user_query"] == "但是又重新置了管。"
    assert rows[0]["subsequent_user_query"] == "跟之前差不多。"
    assert rows[0]["raw_is_positive_feedback"] == "0"
    assert "normalized_interactions.jsonl" in {path.name for path in tmp_path.iterdir()}
    assert any("content*" in warning for warning in warnings)


def test_normalize_strips_utf8_bom_from_header(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "\ufefftrace_id,chat_id,app_user_id,user_id,intention_code,total_session_id,query,query_from,content*,rn_in_session,next_query,next_query_from,next_query_cate,dt,interaction_cnt,is_positive_feedback\n"
        "t1,c1,a1,u1,intent,s1,hello,edit_text,response,1,next,edit_text,,20260502,1,0\n",
        encoding="utf-8",
    )
    config = read_yaml(Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml")

    rows, warnings = normalize_interactions(input_path, tmp_path, config["input_schema"])

    assert rows[0]["trace_id"] == "t1"
    assert any("trace_id" in warning for warning in warnings)

