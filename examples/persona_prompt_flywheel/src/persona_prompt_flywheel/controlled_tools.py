from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

from persona_prompt_flywheel.api_client import UnifiedAgentAPI
from persona_prompt_flywheel.debug_outputs import write_agent_debug_bundle
from persona_prompt_flywheel.feedback_labeling import label_feedback
from persona_prompt_flywheel.feature_extraction import extract_features
from persona_prompt_flywheel.io_utils import ensure_dir, read_yaml
from persona_prompt_flywheel.modeling import run_modeling
from persona_prompt_flywheel.profile_builder import attach_profiles
from persona_prompt_flywheel.prompt_patch_generation import generate_prompt_patches
from persona_prompt_flywheel.schema import normalize_interactions


HANDOFF_SYSTEM_PROMPT = """你是当前 CrewAI agent 的阶段检查员。你必须基于 agent 配置、task 配置、受控工具输出和上一阶段交接信息，检查本阶段是否可以交给下一个 agent。
如果输入快照或工具输出中出现明显的系统协议残片、模板字段、转义后的内部结构、乱码，或完全无法形成自然语言回答的内容，不要强行分析其语义；应在 warnings 中指出这是数据质量问题，并建议后续阶段低置信处理或人工复核。
返回严格 JSON，字段包含：agent_name、status、summary、checks、warnings、next_agent_guidance。
不要改写文件，不要提出执行受控工具之外的动作。"""


class ControlledPipelineTool:
    """Deterministic tool boundary used by the CrewAI-style pipeline runner."""

    name = "controlled_pipeline_tool"
    agent_name = "unknown_agent"
    output_files: list[str] = []

    def run(self, state: dict[str, Any], *, index: int, total: int, task_config: dict[str, Any]) -> dict[str, Any]:
        _print_agent_start(
            index=index,
            total=total,
            agent_name=self.agent_name,
            task_config=task_config,
            input_files=self.input_files(state),
        )
        self._run(state)
        metadata = self.metadata(state)
        handoff = _agent_handoff_review(
            state=state,
            agent_name=self.agent_name,
            task_config=task_config,
            tool_name=self.name,
            output_files=self.output_files,
            metadata=metadata,
        )
        state.setdefault("agent_handoffs", {})[self.agent_name] = handoff
        state["last_handoff"] = handoff
        _write_handoff_files(state["output_dir"], self.agent_name, handoff)
        output_files = [*self.output_files, *_handoff_file_names(self.agent_name)]
        snapshot = self.input_snapshot(state, task_config)
        snapshot["previous_agent_handoff"] = state.get("previous_handoff_for_snapshot")
        snapshot["current_agent_handoff"] = handoff
        debug_dir = write_agent_debug_bundle(
            run_dir=state["output_dir"],
            agent_name=self.agent_name,
            input_snapshot=snapshot,
            output_files=output_files,
            metadata={**metadata, "handoff_status": handoff.get("status")},
        )
        state["previous_handoff_for_snapshot"] = handoff
        _print_agent_done(self.agent_name, state["output_dir"], output_files, debug_dir)
        return state

    def input_files(self, state: dict[str, Any]) -> list[str | Path]:
        return []

    def input_snapshot(self, state: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
        return {"task_config": task_config}

    def metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _run(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


class FeedbackLabelingTool(ControlledPipelineTool):
    name = "feedback_labeling_tool"
    agent_name = "feedback_labeling_agent"
    output_files = [
        "labeled_feedback.jsonl",
        "human_review_queue.jsonl",
        "feedback_labeling_rules.snapshot.yaml",
        "feedback_labeling_audit.jsonl",
        "user_feedback_stats.csv",
    ]

    def input_files(self, state: dict[str, Any]) -> list[str | Path]:
        return [state["input_path"], state["pipeline_config_path"], state["label_config_path"]]

    def _run(self, state: dict[str, Any]) -> None:
        config = state["pipeline_config"]
        normalized, warnings = normalize_interactions(
            state["input_path"],
            state["output_dir"],
            config.get("input_schema", {}),
        )
        profiled = attach_profiles(normalized, config)
        labeled, review_queue = label_feedback(
            profiled,
            api=state["api"],
            config=config,
            rules_path=state["label_config_path"],
            output_dir=state["output_dir"],
        )
        state.update(
            {
                "normalization_warnings": warnings,
                "profiled_records": profiled,
                "labeled": labeled,
                "review_queue": review_queue,
            }
        )

    def input_snapshot(self, state: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_config": task_config,
            "normalized_and_profiled_records": state.get("profiled_records", []),
            "label_config_path": state["label_config_path"],
            "labeling_config": state["pipeline_config"].get("labeling", {}),
        }

    def metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_role": "semantic feedback relabeling",
            "controlled_tool": self.name,
            "raw_is_positive_feedback_used": False,
            "records_in": len(state.get("profiled_records", [])),
            "records_out": len(state.get("labeled", [])),
            "human_review_queue_count": len(state.get("review_queue", [])),
        }


class FeatureExtractionTool(ControlledPipelineTool):
    name = "feature_extraction_tool"
    agent_name = "feature_discovery_agent"
    output_files = [
        "feature_config.resolved.yaml",
        "feature_catalog.json",
        "feature_matrix.jsonl",
        "feature_matrix.csv",
        "feature_extraction_audit.jsonl",
    ]

    def input_files(self, state: dict[str, Any]) -> list[str | Path]:
        return [state["output_dir"] / "labeled_feedback.jsonl", state["feature_config_path"]]

    def _run(self, state: dict[str, Any]) -> None:
        feature_matrix, feature_catalog = extract_features(
            state["labeled"],
            api=state["api"],
            feature_config_path=state["feature_config_path"],
            output_dir=state["output_dir"],
            llm_execution_config=state["pipeline_config"].get("llm_execution", {}),
        )
        state.update(
            {
                "feature_matrix": feature_matrix,
                "feature_catalog": feature_catalog,
            }
        )

    def input_snapshot(self, state: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_config": task_config,
            "labeled_feedback": state.get("labeled", []),
            "feature_config_path": state["feature_config_path"],
        }

    def metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_role": "response feature extraction",
            "controlled_tool": self.name,
            "records_in": len(state.get("labeled", [])),
            "feature_rows_out": len(state.get("feature_matrix", [])),
        }


class ModelingTool(ControlledPipelineTool):
    name = "modeling_tool"
    agent_name = "modeling_code_agent"
    output_files = [
        "training_plan.json",
        "training_dataset.csv",
        "design_matrix.csv",
        "model_report.json",
        "coefficients.csv",
        "segment_insights.json",
        "model_insights.json",
        "model_insights.md",
        "model_card.md",
        "model.joblib",
    ]

    def input_files(self, state: dict[str, Any]) -> list[str | Path]:
        return [state["output_dir"] / "feature_matrix.jsonl", state["output_dir"] / "feature_catalog.json", state["modeling_config_path"]]

    def _run(self, state: dict[str, Any]) -> None:
        model_report, segment_insights = run_modeling(
            state["feature_matrix"],
            state["feature_catalog"],
            modeling_config_path=state["modeling_config_path"],
            output_dir=state["output_dir"],
        )
        state.update({"model_report": model_report, "segment_insights": segment_insights})

    def input_snapshot(self, state: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_config": task_config,
            "feature_catalog": state.get("feature_catalog", {}),
            "feature_matrix": state.get("feature_matrix", []),
            "modeling_config_path": state["modeling_config_path"],
        }

    def metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_role": "controlled sklearn logistic regression sandbox",
            "controlled_tool": self.name,
            "model_status": state.get("model_report", {}).get("status"),
            "training_rows": state.get("model_report", {}).get("n_samples"),
            "raw_is_positive_feedback_used": False,
        }


class PromptPatchGenerationTool(ControlledPipelineTool):
    name = "prompt_patch_generation_tool"
    agent_name = "prompt_patch_generator_agent"
    output_files = [
        "prompt_patches.jsonl",
        "prompt_patches.md",
        "prompt_patch_registry.json",
        "runtime_prompt_example.md",
    ]

    def input_files(self, state: dict[str, Any]) -> list[str | Path]:
        return [
            state["output_dir"] / "segment_insights.json",
            state["output_dir"] / "model_report.json",
            state["output_dir"] / "feature_catalog.json",
            state["modeling_config_path"],
        ]

    def _run(self, state: dict[str, Any]) -> None:
        patches = generate_prompt_patches(
            segment_insights=state["segment_insights"],
            model_report=state["model_report"],
            feature_catalog=state["feature_catalog"],
            api=state["api"],
            output_dir=state["output_dir"],
            modeling_config_path=state["modeling_config_path"],
            llm_execution_config=state["pipeline_config"].get("llm_execution", {}),
        )
        state["patches"] = patches

    def input_snapshot(self, state: dict[str, Any], task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_config": task_config,
            "segment_insights": state.get("segment_insights", {}),
            "model_report": state.get("model_report", {}),
            "feature_catalog": state.get("feature_catalog", {}),
        }

    def metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_role": "profile-conditioned prompt patch generation",
            "controlled_tool": self.name,
            "patch_count": len(state.get("patches", [])),
            "default_patch_status": "candidate_or_low_confidence_candidate",
        }


class RunSummaryTool:
    name = "run_summary_tool"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = _write_run_summary(
            state["output_dir"],
            state.get("labeled", []),
            state.get("review_queue", []),
            state.get("model_report", {}),
            state.get("segment_insights", {}),
            state.get("patches", []),
            state.get("normalization_warnings", []),
        )
        artifact_dirs = _organize_run_artifacts(state["output_dir"])
        summary["artifact_dirs"] = {name: str(path) for name, path in artifact_dirs.items()}
        state["summary"] = summary
        print(f"Flywheel completed: {state['output_dir']}")
        print(f"Summary: {artifact_dirs['core'] / 'run_summary.md'}")
        print(f"Agent debug outputs: {artifact_dirs['debug'] / 'output'}")
        print(f"Prompt patches: {artifact_dirs['core'] / 'prompt_patches.jsonl'}")
        return summary


def build_initial_state(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    pipeline_config_path: str | Path,
    label_config_path: str | Path,
    feature_config_path: str | Path,
    modeling_config_path: str | Path,
    api: UnifiedAgentAPI,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    return {
        "input_path": Path(input_path),
        "output_dir": output,
        "pipeline_config_path": Path(pipeline_config_path),
        "label_config_path": Path(label_config_path),
        "feature_config_path": Path(feature_config_path),
        "modeling_config_path": Path(modeling_config_path),
        "pipeline_config": read_yaml(pipeline_config_path),
        "api": api,
    }


def _agent_handoff_review(
    *,
    state: dict[str, Any],
    agent_name: str,
    task_config: dict[str, Any],
    tool_name: str,
    output_files: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    existing_outputs = [
        str(state["output_dir"] / file_path)
        for file_path in output_files
        if (state["output_dir"] / file_path).exists()
    ]
    missing_outputs = [
        str(state["output_dir"] / file_path)
        for file_path in output_files
        if not (state["output_dir"] / file_path).exists()
    ]
    payload = {
        "task": "agent_handoff_review",
        "agent_name": agent_name,
        "agent_config": task_config.get("agent_config", {}),
        "task_config": {
            "name": task_config.get("name"),
            "description": task_config.get("description"),
            "expected_output": task_config.get("expected_output"),
            "post_tool_review": task_config.get("post_tool_review", []),
            "handoff_contract": task_config.get("handoff_contract", {}),
        },
        "controlled_tool": tool_name,
        "tool_metadata": metadata,
        "existing_output_files": existing_outputs,
        "missing_output_files": missing_outputs,
        "previous_agent_handoff": state.get("last_handoff"),
    }
    response_text = state["api"].generate(
        [
            {"role": "system", "content": HANDOFF_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        response_format="json",
        metadata={"operation": "agent_handoff_review", "agent_name": agent_name},
    )
    try:
        handoff = json.loads(response_text)
    except json.JSONDecodeError:
        handoff = {
            "agent_name": agent_name,
            "status": "needs_review",
            "summary": response_text,
            "checks": [],
            "warnings": ["Agent handoff response was not valid JSON."],
            "next_agent_guidance": "Review the raw handoff summary before continuing.",
        }
    handoff.setdefault("agent_name", agent_name)
    handoff.setdefault("status", "ready_for_next_stage" if not missing_outputs else "needs_review")
    handoff.setdefault("summary", "")
    handoff.setdefault("checks", [])
    handoff.setdefault("warnings", [])
    handoff.setdefault("next_agent_guidance", "")
    handoff["controlled_tool"] = tool_name
    handoff["existing_output_files"] = existing_outputs
    handoff["missing_output_files"] = missing_outputs
    return handoff


def _handoff_file_names(agent_name: str) -> list[str]:
    return [f"{agent_name}_handoff.json", f"{agent_name}_handoff.md"]


def _write_handoff_files(output_dir: Path, agent_name: str, handoff: dict[str, Any]) -> None:
    json_path = output_dir / f"{agent_name}_handoff.json"
    md_path = output_dir / f"{agent_name}_handoff.md"
    json_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = "\n".join(f"- {item}" for item in handoff.get("checks", [])) or "- 无"
    warnings = "\n".join(f"- {item}" for item in handoff.get("warnings", [])) or "- 无"
    md_path.write_text(
        "\n".join(
            [
                f"# {agent_name} Handoff",
                "",
                f"- status: {handoff.get('status')}",
                f"- controlled_tool: {handoff.get('controlled_tool')}",
                "",
                "## Summary",
                str(handoff.get("summary", "")),
                "",
                "## Checks",
                checks,
                "",
                "## Warnings",
                warnings,
                "",
                "## Next Agent Guidance",
                str(handoff.get("next_agent_guidance", "")),
                "",
            ]
        ),
        encoding="utf-8",
    )


CORE_RESULT_FILES = {
    "run_summary.md",
    "model_insights.md",
    "model_insights.json",
    "model_report.json",
    "model_card.md",
    "prompt_patches.md",
    "prompt_patches.jsonl",
    "prompt_patch_registry.json",
    "runtime_prompt_example.md",
}

STAGE_ARTIFACT_FILES = {
    "normalized_interactions.jsonl",
    "normalized_interactions.csv",
    "labeled_feedback.jsonl",
    "human_review_queue.jsonl",
    "feedback_labeling_audit.jsonl",
    "feedback_labeling_rules.snapshot.yaml",
    "user_feedback_stats.csv",
    "feature_config.resolved.yaml",
    "feature_catalog.json",
    "feature_matrix.jsonl",
    "feature_matrix.csv",
    "feature_extraction_audit.jsonl",
    "training_plan.json",
    "training_dataset.csv",
    "design_matrix.csv",
    "coefficients.csv",
    "segment_insights.json",
    "model.joblib",
}


def _organize_run_artifacts(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "core": output_dir / "01_core_results",
        "stage": output_dir / "02_stage_artifacts",
        "handoff": output_dir / "03_agent_handoffs",
        "debug": output_dir / "04_debug_bundles",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for file_name in CORE_RESULT_FILES:
        _move_if_exists(output_dir / file_name, dirs["core"] / file_name)
    for file_name in STAGE_ARTIFACT_FILES:
        _move_if_exists(output_dir / file_name, dirs["stage"] / file_name)
    for path in output_dir.glob("*_handoff.*"):
        if path.is_file():
            _move_if_exists(path, dirs["handoff"] / path.name)

    debug_source = output_dir / "output"
    debug_destination = dirs["debug"] / "output"
    if debug_source.exists():
        if debug_destination.exists():
            shutil.rmtree(debug_destination)
        shutil.move(str(debug_source), str(debug_destination))
        _refresh_debug_manifests(debug_destination)
    return dirs


def _move_if_exists(source: Path, destination: Path) -> None:
    if source.exists() and source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        source.replace(destination)


def _refresh_debug_manifests(debug_output_dir: Path) -> None:
    for manifest_path in debug_output_dir.glob("*/output_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["debug_directory"] = str(manifest_path.parent)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_agent_start(
    *,
    index: int,
    total: int,
    agent_name: str,
    task_config: dict[str, Any],
    input_files: list[str | Path],
) -> None:
    print(f"\n[{index}/{total}] Starting {agent_name}", flush=True)
    task_name = task_config.get("name")
    if task_name:
        print(f"Task: {task_name}", flush=True)
    print("Input files:", flush=True)
    for path in input_files:
        print(f"  - {Path(path)}", flush=True)


def _print_agent_done(agent_name: str, run_dir: Path, output_files: list[str | Path], debug_dir: Path) -> None:
    print(f"[done] {agent_name}", flush=True)
    print("Output files:", flush=True)
    for file_path in output_files:
        path = Path(file_path)
        source = path if path.is_absolute() else run_dir / path
        status = "exists" if source.exists() else "missing"
        print(f"  - {source} ({status})", flush=True)
    print(f"Debug bundle: {debug_dir}", flush=True)


def _write_run_summary(
    output_dir: Path,
    labeled: list[dict[str, Any]],
    review_queue: list[dict[str, Any]],
    model_report: dict[str, Any],
    segment_insights: dict[str, Any],
    patches: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    counts = Counter(row["semantic_feedback_label"] for row in labeled)
    training_rows = int(model_report.get("n_samples") or 0)
    positive = ", ".join(driver["feature"] for driver in segment_insights.get("global_insights", {}).get("positive_drivers", [])[:3]) or "无"
    negative = ", ".join(driver["feature"] for driver in segment_insights.get("global_insights", {}).get("negative_drivers", [])[:3]) or "无"
    skipped = segment_insights.get("skipped_segments", [])
    summary = {
        "total_rows": len(labeled),
        "raw_is_positive_feedback_ignored": True,
        "training_rows": training_rows,
        "model_status": model_report.get("status"),
        "prompt_patch_count": len(patches),
        "human_review_queue_count": len(review_queue),
        "agent_debug_output_dir": str(output_dir / "04_debug_bundles" / "output"),
        "model_insights_path": str(output_dir / "01_core_results" / "model_insights.md"),
    }
    summary.update({label: count for label, count in sorted(counts.items())})
    label_count_lines = "\n".join(f"- {label}: {count}" for label, count in sorted(counts.items())) or "- 无标签"
    excluded_counts = model_report.get("excluded_label_counts", {})
    excluded_lines = "\n".join(f"- {label}: {count}" for label, count in sorted(excluded_counts.items())) or "- 无排除标签"
    exclusion_reasons = model_report.get("training_exclusion_reasons", {})
    reason_lines = "\n".join(f"- {reason}: {count}" for reason, count in sorted(exclusion_reasons.items())) or "- 无排除原因"
    text = f"""# Persona Prompt Flywheel Run Summary

- 读入样本数: {len(labeled)}
- raw_is_positive_feedback 是否被忽略: 是
- 训练使用样本数: {training_rows}
- 模型状态: {model_report.get('status')}
- 模型洞察报告: {output_dir / '01_core_results' / 'model_insights.md'}
- 主要正向驱动因素: {positive}
- 主要负向驱动因素: {negative}
- 生成 candidate prompt patch 数: {len(patches)}
- 样本不足跳过的 segment: {', '.join(item['segment'] for item in skipped) or '无'}
- 人工审核队列条数: {len(review_queue)}
- Agent debug 输出目录: {output_dir / '04_debug_bundles' / 'output'}
- normalization warnings: {', '.join(warnings) or '无'}

## 产物分类目录
- 核心结果: {output_dir / '01_core_results'}
- 阶段产物: {output_dir / '02_stage_artifacts'}
- Agent handoff: {output_dir / '03_agent_handoffs'}
- Debug bundles: {output_dir / '04_debug_bundles'}

## 标签分布
{label_count_lines}

## 未进入训练的标签分布
{excluded_lines}

## 未进入训练的原因
{reason_lines}
"""
    (output_dir / "run_summary.md").write_text(text, encoding="utf-8")
    return summary
