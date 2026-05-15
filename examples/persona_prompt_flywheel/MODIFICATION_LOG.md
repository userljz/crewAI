# Persona Prompt Flywheel 修改记录

## Repo Reading Summary

- 仓库是 CrewAI monorepo，根目录 `pyproject.toml` 使用 uv workspace，核心源码位于 `lib/crewai/src/crewai/`，测试位于 `lib/crewai/tests/`。
- CrewAI 推荐项目结构可从 `lib/cli/src/crewai_cli/templates/crew/` 看到：`crew.py` + `config/agents.yaml` + `config/tasks.yaml`，并使用 `@CrewBase`、`@agent`、`@task`、`@crew` 和 `Process.sequential`。
- 本需求最合适的扩展点是新增独立 example，而不是修改 CrewAI core；因此新增 `examples/persona_prompt_flywheel/`，保留 4 个 Agent 的配置和 `crew.py` 语义包装。
- 运行入口采用清晰 Python CLI：`python examples/persona_prompt_flywheel/run_flywheel.py --input ... --output ... --mock-api`。

## 修改过程

1. 新增 `examples/persona_prompt_flywheel/` 示例目录，避免改动 CrewAI core。
2. 新增 `run_flywheel.py`，串起 schema normalization、profile building、feedback labeling、feature extraction、controlled modeling、prompt patch generation。
3. 新增 `src/persona_prompt_flywheel/api_client.py`，提供 `UnifiedAgentAPI`、`MockAgentAPI` 和真实 API 占位 adapter。
4. 新增 `schema.py`，支持 CSV/TSV 解析、字段重命名、兼容 `aintention_code`、输出 normalized JSONL/CSV。
5. 新增 `profile_builder.py`，支持真实画像字段优先，否则构造 pseudo profile。
6. 新增 `feedback_labeling.py` 与 `human_review.py`，实现语义重标注、`uncertain_needs_human_review`、人工审核队列、规则合并。
7. 新增 `feature_extraction.py` / `feature_discovery.py`，输出 feature catalog、feature matrix、feature evidence/confidence/source 和 candidate features。
8. 新增 `modeling_tools.py` / `modeling.py`，通过固定 sklearn LogisticRegression sandbox 训练、评估并输出可解释系数和 segment insights。
9. 新增 `prompt_patch_generation.py` 与 `registry.py`，生成条件化 Prompt Patch，并提供线上注入 stub。
10. 新增 `config/agents.yaml`、`config/tasks.yaml`、`src/persona_prompt_flywheel/crew.py`，复用 CrewAI 的 4-Agent/Task/Crew 配置模式，并让每个 agent 在受控工具执行后生成 LLM handoff 摘要。
11. 新增 `config/*.yaml` 和 `data/sample_interactions.tsv`，集中提供默认规则、pipeline 配置、特征配置、建模配置与 smoke 数据。
12. 新增 `README.md`，说明安装、运行、输入输出、mock/real API 切换、画像替换、人工审核规则合并。
13. 新增 `src/persona_prompt_flywheel/debug_outputs.py`，并在 `run_flywheel.py` 中为 4 个 Agent 写入 `output/<agent_name>/` debug bundle。
14. 更新 smoke test，校验每个 Agent 都生成 `output_manifest.json`。

## 后续交接注意

- `raw_is_positive_feedback` 仅保留为审计字段，默认不参与打标或训练。
- mock 模式是 deterministic 的，适合 smoke test；真实 API 只需要替换 `ConfigurableAgentAPI` 对接用户提供的统一 API。
- 样例数据很小，模型报告通常是 `trained_low_confidence`；这是预期行为，不代表生产可用。

## 验证记录

- Demo 命令已运行成功：

```bash
python examples/persona_prompt_flywheel/run_flywheel.py \
  --input examples/persona_prompt_flywheel/data/sample_interactions.tsv \
  --output runs/persona_prompt_flywheel_demo \
  --mock-api
```

- 关键输出目录：`runs/persona_prompt_flywheel_demo/`，运行结束后按 `01_core_results/`、`02_stage_artifacts/`、`03_agent_handoffs/`、`04_debug_bundles/` 四类归档。
- `01_core_results/run_summary.md` 显示：读入 5 条样本；positive/negative/neutral/uncertain/no_signal 各 1/1/1/1/1；训练样本 3 条；模型状态 `trained_low_confidence`；生成 low-confidence candidate prompt patch；人工审核队列 1 条。
- 测试命令已运行成功：

```bash
uv run --no-project --with pytest --with scikit-learn --with joblib --with pyyaml \
  pytest --confcutdir=. -c /dev/null -o addopts='' tests
```

- 测试结果：`6 passed, 2 warnings`。两个 warning 分别来自 sklearn 未来版本参数提示和 `/dev/null` pytest 配置导致的 cache 权限提示，不影响示例功能。

## 2026-05-13 Debug 输出目录补充

- 新增每个 Agent 的独立 debug 输出目录：

```text
runs/persona_prompt_flywheel_demo/04_debug_bundles/output/
  feedback_labeling_agent/
  feature_discovery_agent/
  modeling_code_agent/
  prompt_patch_generator_agent/
```

- 每个目录包含：
  - `input_snapshot.json` 或 `input_snapshot.jsonl`：该 Agent 的输入快照。
  - `output_manifest.json`：该 Agent 复制了哪些输出文件、metadata、缺失文件列表。
  - 对应 Agent 主要输出文件的副本，例如 `labeled_feedback.jsonl`、`feature_matrix.jsonl`、`model_report.json`、`prompt_patches.jsonl`。

