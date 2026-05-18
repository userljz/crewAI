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
7. 新增 `feature_extraction.py`，输出 feature catalog、feature matrix、feature evidence/confidence/source。
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

## 2026-05-17 本地 vLLM 真实数据调试

- 使用本地 OpenAI-compatible vLLM 服务 `http://127.0.0.1:8000/v1/chat/completions` 和模型 `qwen3-30b-a3b` 跑通 `data/converted_feedback_data.csv`，输出目录为 `runs/persona_prompt_flywheel_qwen3_real_debug`。
- 发现第三条样本的 `response_content` 是系统协议残片/模板字段/转义结构。LLM 特征抽取已将语义特征置为 `-1`，但建模阶段仍将其作为正样本训练，导致缺失哨兵值进入逻辑回归。
- 修复 `feature_extraction.py`：当所有 LLM 语义/主观特征均为 `-1` 时，将该行标记为 `response_content_quality=unreadable_or_system_noise`，并设置 `usable_for_training=false`、`training_exclusion_reason=unreadable_response_content`。
- 修复 `modeling_tools.py`：交互特征现在遵守 `min_category_count_for_interaction`，避免 4 条样本生成数百个稀疏交互项；没有满足 holdout 条件时不再报告训练集上的 accuracy/precision/recall，避免把过拟合结果显示成泛化指标。
- 修复 `prompt_patch_generation.py`：当所有 segment 都因为样本不足被跳过时，只生成一个全局低置信 fallback patch，不再为每个单样本 segment 生成看似个性化的 patch。
- 修复 `controlled_tools.py`：run summary 的训练样本数改为读取 `model_report.n_samples`，并输出未进入训练的标签分布和排除原因。
- 扩展 `api_client.py` 与 README：显式支持 `LLM_PROVIDER=vllm` / `openai_compatible`，可通过 `OPENAI_BASE_URL=http://127.0.0.1:8000/v1` 接入本地 vLLM。
- 进一步修复 `feature_config.yaml` / `feature_extraction.py`：将段落数、问题数、列表数、加粗数、链接数、代码块数、表格判断等机械结构特征改为 Python 确定性抽取，避免 LLM 计数和 evidence 不一致。
- 收紧 `contains_medical_safety_note` 语义，并增加非医疗保险/退保流程的后处理规则，避免把 `长护险`、`95519`、退保步骤误判成临床安全提示。
- 最终验证输出目录：`runs/persona_prompt_flywheel_qwen3_real_final`。最终结果读入 4 条、训练使用 3 条、排除 1 条 `unreadable_response_content`，未报告 holdout 指标，仅生成 1 个全局低置信 fallback patch。

## 2026-05-18 5000 条合成数据全流程压测

- 新增 `data/synthetic_medical_feedback_5000.csv`，字段格式对齐 `converted_feedback_data.csv`，用于大数据量流程压测。
- 使用本地 vLLM 跑 5000 条时发现默认 `llm_execution.max_workers=4` 过慢；调整为 `24`，并在 `feedback_labeling.py` / `feature_extraction.py` 增加每 250 条进度输出，便于监督长流程。
- 发现特征抽取阶段仍按每条样本多次 LLM 调用，5000 条会拖到不可接受；新增 `feature_config.yaml.large_dataset_fast_path`，当样本数达到阈值时启用确定性语义启发式，完整产出 feature matrix，但避免逐条 LLM 评分。
- 发现 prompt patch 阶段可能收到非字符串 `patch_text` 并在写 Markdown 时崩溃；修复 `prompt_patch_generation.py`，将 dict/list/裸文本响应统一收敛为字符串，并新增测试。
- 发现模型虽为 `trained`，但正样本比例低且 holdout precision 约 0.19-0.22；新增 prompt patch 质量门槛：precision 低于 0.30 时只生成全局 `low_confidence_candidate` fallback patch，避免为低质量指标生成大量 segment candidate patch。
- 收紧 `modeling_config.yaml` 的 trained 文案，避免在指标质量不足时直接声称可灰度验证。
- 最终验证输出目录：`runs/persona_prompt_flywheel_synthetic_5000_final_reviewed`。最终读入 5000 条、训练使用 4897 条、模型状态 `trained`，但 patch 因 precision 低于质量门槛降级为 1 个全局低置信候选。
- 根据复核意见移除生产路径中的语义硬编码：删除 `contains_medical_safety_note` 的 Python 后处理，并清理已关闭的 fast path 语义启发式实现；保险/退保/长护险等边界改由 `feature_config.yaml` 中的 LLM rubric 描述处理。
- 根据复核意见移除当前 pipeline 中的候选新特征发现功能：删除 `candidate_discovery` 配置、`discover_candidate_features` 代码、`feature_discovery.py` shim、`candidate_feature_discoveries.jsonl` 产物和 prompt patch 对 candidate features 的依赖。后续如果需要 LLM-driven feature discovery，应作为独立 agent/流程重新设计。

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

