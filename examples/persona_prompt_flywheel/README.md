# Persona Prompt 数据飞轮

这个示例是一个离线数据飞轮，用用户后续交互来迭代 persona prompt。它采用“LLM-backed agents + 受控工具执行”的模式：CrewAI 风格的 crew runner 负责按 agent/task 顺序组织流程，每个 agent 调用预定义工具完成稳定执行，然后再用 LLM 检查本阶段产物并生成 handoff 交接摘要。系统不会直接使用原始 `is_positive_feedback` 做训练标签，而是先用 LLM 重新理解反馈语义，再抽取回答特征、训练受控逻辑回归模型，最后生成可人工复核的 prompt patch 候选。

## 流程总览

1. `feedback_labeling_agent` 通过 `feedback_labeling_tool` 读取 CSV/TSV，归一化字段、构建 profile，并用 LLM 给每条交互打语义反馈标签；随后 agent 检查标签分布和人工复核队列，输出 handoff。
2. `feature_discovery_agent` 通过 `feature_extraction_tool` 抽取 Python 确定性客观统计、LLM 语义客观特征和 rubric 主观特征，并发现候选新特征；随后 agent 检查特征矩阵完整性，输出 handoff。
3. `modeling_code_agent` 通过 `modeling_tool` 运行固定 sklearn 逻辑回归，分析哪些特征更容易带来正反馈；随后 agent 检查模型状态、样本量和 driver 可靠性，输出 handoff。
4. `prompt_patch_generator_agent` 通过 `prompt_patch_generation_tool` 按 profile/intent 生成 prompt patch 候选；随后 agent 检查 patch 状态和触发条件，输出最终 handoff。
5. `run_summary_tool` 汇总模型洞察、人工复核队列、patch registry 和调试文件。

## 快速运行

从 `test/crewAI` 目录运行：

```bash
python examples/persona_prompt_flywheel/run_flywheel.py \
  --input examples/persona_prompt_flywheel/data/sample_interactions.tsv \
  --output runs/persona_prompt_flywheel_demo \
  --mock-api
```

如果当前环境没有依赖，可以用 `uv`：

```bash
uv run --no-project --with pyyaml --with numpy --with scikit-learn --with joblib \
  python examples/persona_prompt_flywheel/run_flywheel.py \
    --input examples/persona_prompt_flywheel/data/sample_interactions.tsv \
    --output runs/persona_prompt_flywheel_demo \
    --mock-api
```

`--mock-api` 使用本地确定性 mock，不需要真实 LLM key，适合 demo 和测试。

运行过程中会打印当前执行到第几个 agent、该 agent 正在读取的输入文件、完成后写出的主输出文件路径，以及对应的 debug bundle 目录，方便实时掌握 pipeline 进度。

## 使用真实 LLM

所有 LLM 调用都通过 `UnifiedAgentAPI`。如果使用 OpenRouter，推荐这样配置：

```bash
export LLM_PROVIDER=openrouter
export LLM_MODEL=deepseek/deepseek-chat
export OPENROUTER_API_KEY="..."

python examples/persona_prompt_flywheel/run_flywheel.py \
  --input examples/persona_prompt_flywheel/data/sample_interactions.tsv \
  --output runs/persona_prompt_flywheel_real
```

`LLM_MODEL` 使用 OpenRouter 的模型 ID，例如 `deepseek/deepseek-chat`。如果你习惯写 `openrouter/deepseek/deepseek-v3.2`，代码会自动去掉开头的 `openrouter/` 后再发给 OpenRouter。API key 可以放在 `OPENROUTER_API_KEY`；如果你已有统一变量名，也可以用 `LLM_API_KEY`。

可选配置：

```bash
export OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions"
export OPENROUTER_HTTP_REFERER="https://your-app.example"
export OPENROUTER_APP_TITLE="Persona Prompt Flywheel"
```

如果使用本地 vLLM 或其他 OpenAI-compatible 服务：

```bash
export LLM_PROVIDER=vllm
export LLM_MODEL=qwen3-30b-a3b
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

python examples/persona_prompt_flywheel/run_flywheel.py \
  --input examples/persona_prompt_flywheel/data/converted_feedback_data.csv \
  --output runs/persona_prompt_flywheel_real
```

如果不用 OpenRouter，也可以继续接入自定义统一 HTTP endpoint：

```bash
export UNIFIED_AGENT_API_URL="https://your-unified-agent-api.example/generate"
export UNIFIED_AGENT_API_KEY="..."

python examples/persona_prompt_flywheel/run_flywheel.py \
  --input examples/persona_prompt_flywheel/data/sample_interactions.tsv \
  --output runs/persona_prompt_flywheel_real
```

endpoint 返回内容需要放在 `text`、`content` 或 `result` 字段中。

## 输入数据

输入可以是 CSV 或 TSV。字段要求和别名在 `config/pipeline_config.yaml` 的 `input_schema` 中配置。

默认必需字段包括：`trace_id`、`chat_id`、`app_user_id`、`user_id`、`intention_code`、`total_session_id`、`user_query`、`query_from`、`response_content`、`rn_in_session`、`subsequent_user_query`、`next_query_from`、`next_query_cate`、`dt`、`interaction_cnt`、`raw_is_positive_feedback`。

常见字段别名也在配置中，例如：

```yaml
input_schema:
  column_aliases:
    query: user_query
    content*: response_content
    next_query: subsequent_user_query
    is_positive_feedback: raw_is_positive_feedback
```

`raw_is_positive_feedback` 只保留做审计，默认不会参与 LLM 打标，也不会直接作为训练标签。

## 配置文件

这个飞轮的原则是：可调行为尽量放在 YAML，Python 只保留流程编排和固定算法实现。
所有可调 YAML 都集中在 `config/` 目录下，方便统一查看和修改。

配置文件总览：

- `config/agents.yaml`：CrewAI agent 的角色、目标、背景，以及阶段 handoff 风格。
- `config/tasks.yaml`：CrewAI task 描述、expected output、绑定的受控 tool、tool 后检查项和 handoff contract。
- `config/pipeline_config.yaml`：输入字段、profile、session 分桶、全局 pipeline 配置。
- `config/feedback_labeling_rules.yaml`：反馈标签和 LLM 打标 prompt。
- `config/feature_config.yaml`：确定性特征、LLM 特征抽取和候选特征发现定义。
- `config/modeling_config.yaml`：逻辑回归、洞察、prompt patch 生成配置。

### `config/agents.yaml` 和 `config/tasks.yaml`

负责 CrewAI 编排层的 agent 角色、目标、背景、handoff 风格，以及 task 描述、expected output、`controlled_tool` 绑定、tool 后检查项和 handoff contract。真实执行由 `PersonaPromptFlywheelControlledCrew` 按 `tasks.yaml` 的 task 顺序调度；每个 task 先调用对应受控工具，再由当前 agent 使用 LLM 检查输出并生成给下一阶段的 handoff 摘要。

### `config/pipeline_config.yaml`

负责输入和 profile 配置。

- `input_schema`：输入字段、字段别名、分隔符策略。
- `profile_columns`：如果输入中存在这些真实画像字段，会优先用它们组成 profile。
- `profile`：profile 输出字段、来源列、session stage 分桶、interaction band 分桶、fallback composite 规则。
- `labeling.training_label_policy`：训练标签策略名称，用于输出审计。
- `llm_execution.max_workers`：全局 LLM 并发数，当前用于反馈打标、特征抽取和 prompt patch 生成；每条样本仍单独请求，不做大 batch。

修改 session 分桶示例：

```yaml
profile:
  session_stage_bins:
    - label: early
      max_inclusive: 3
    - label: mid
      max_inclusive: 8
    - label: late
```

调整全局 LLM 并发数：

```yaml
llm_execution:
  max_workers: 4
```

这个配置控制样本之间或 segment 之间的并发；每条样本仍然单独调用 LLM，避免 batch 内互相污染证据或标签。

### `config/feedback_labeling_rules.yaml`

负责 LLM 反馈打标。

- `decision_policy`：是否忽略原始反馈字段、自动标签置信度阈值、不确定标签名。
- `prompt`：给 LLM 的 system prompt、task、低置信处理指令、输出 JSON contract。
- `labels`：所有可选标签、标签描述、训练 target、是否需要人工复核。

示例：

```yaml
labels:
  positive_feedback:
    description: 用户明确表达上一轮回答有帮助、解决问题、感谢且语义上指向满意。
    training_target_positive: 1
  negative_feedback:
    description: 用户指出上一轮回答错误、没用、误解意图、要求重来或表达投诉/不满。
    training_target_positive: 0
  uncertain_needs_human_review:
    description: 语义含糊、上下文不足或多种标签都合理，需要人工复核。
    training_target_positive:
    needs_human_review: true
```

新增或修改标签时，只改这里。`training_target_positive: 1` 表示正样本，`0` 表示非正样本，空值表示不进入训练。

### `config/feature_config.yaml`

负责特征抽取。

- `deterministic_objective_features`：由 Python 直接计算的客观统计特征，例如字符数和 token 数。
- `objective_features`：由 LLM 判断的语义客观特征。
- `objective_extraction_prompt`：语义客观特征抽取 prompt，要求 LLM 返回每个特征的数量或 0/1 判断。
- `subjective_features`：由 LLM rubric 打分的主观特征。
- `subjective_scoring_prompt`：主观特征评分 prompt。
- `feature_matrix_record_fields` / `feature_matrix_csv_fields`：输出字段。

例如 `action_item_count`：

```yaml
objective_features:
  action_item_count:
    type: numeric
    description: 回答中有多少个用户可以执行的具体行动项。
    extraction_rubric: 统计明确可执行的下一步、检查项、联系对象、操作命令或建议动作数量。
```

机械统计量会由 Python 直接计算；语义客观特征会把 `description` 和 `extraction_rubric` 发给 LLM，由 LLM 根据整段回答语义返回结构化数量、置信度和证据。

### `config/modeling_config.yaml`

负责建模、洞察和 prompt patch 生成。

- `training_plan`：数值特征、分类特征、交互项、排除标签、模型名称。
- `training_dataset`：训练集保留哪些元数据字段。
- `logistic_regression`：逻辑回归超参、样本量阈值、状态标签、warning 文案。
- `insights`：top driver 数量、segment 最小样本数、解释文案、复核建议模板。
- `thresholds`：全局训练和 segment 阈值。
- `prompt_patch_generation`：patch 状态、fallback patch、do/don’t rules、置信度映射、runtime prompt 示例。

修改逻辑回归样本阈值：

```yaml
logistic_regression:
  min_total_samples_for_trained: 100
  min_class_samples_for_trained: 50
```

修改 prompt patch 默认文案：

```yaml
prompt_patch_generation:
  patch_text:
    base: 回答时先给出简短结论，再补充关键依据。
```

## 输出文件

每次运行会在 `--output` 目录下创建 4 个分类目录，并在终端逐个打印各 agent 的输入文件和输出文件路径。

```text
runs/persona_prompt_flywheel_demo/
  01_core_results/
  02_stage_artifacts/
  03_agent_handoffs/
  04_debug_bundles/
```

### `01_core_results/`

放最常看的核心结果和最终交付产物：

- `run_summary.md`：运行摘要。
- `model_report.json`：模型状态、指标、样本统计。
- `model_insights.json`：结构化模型洞察，供程序或后续 agent 使用。
- `model_insights.md`：人工可读的模型洞察报告。
- `model_card.md`：模型卡片。
- `prompt_patches.jsonl`：prompt patch 候选。
- `prompt_patches.md`：人工可读的 prompt patch 候选说明。
- `prompt_patch_registry.json`：patch registry。
- `runtime_prompt_example.md`：应用 patch 后的运行时 prompt 示例。

### `02_stage_artifacts/`

放每个阶段产生的中间数据、审计文件和可复现实验材料：

- `normalized_interactions.jsonl`：字段归一化后的输入数据。
- `normalized_interactions.csv`：字段归一化后的 CSV 版本，方便人工查看或用表格工具打开。
- `labeled_feedback.jsonl`：LLM 打标后的数据。
- `human_review_queue.jsonl`：需要人工复核的样本。
- `feedback_labeling_audit.jsonl`：反馈打标审计记录，记录每条样本的标签来源、原始反馈字段是否被使用、证据摘要。
- `feedback_labeling_rules.snapshot.yaml`：本次运行使用的标签配置快照。
- `user_feedback_stats.csv`：按用户聚合的反馈标签统计、有效交互率、正反馈率和人工复核率。
- `feature_matrix.jsonl` / `feature_matrix.csv`：特征矩阵。
- `feature_catalog.json`：正式特征目录。
- `feature_config.resolved.yaml`：本次运行实际使用的特征配置快照。
- `feature_extraction_audit.jsonl`：特征抽取审计记录，记录每条样本抽取了多少客观/主观特征以及使用的 LLM adapter。
- `training_plan.json`：根据特征目录和建模配置生成的训练计划，包括数值特征、分类特征和交互项设置。
- `training_dataset.csv`：进入逻辑回归训练的数据。
- `design_matrix.csv`：模型实际使用的设计矩阵。
- `coefficients.csv`：逻辑回归系数。
- `segment_insights.json`：分 segment 洞察。
- `model.joblib`：训练出的 sklearn 逻辑回归模型文件；这是运行产物，不建议提交到 Git。

### `03_agent_handoffs/`

放每个 LLM-backed agent 在调用受控工具后生成的阶段检查和交接摘要：

- `feedback_labeling_agent_handoff.json` / `.md`
- `feature_discovery_agent_handoff.json` / `.md`
- `modeling_code_agent_handoff.json` / `.md`
- `prompt_patch_generator_agent_handoff.json` / `.md`

### `04_debug_bundles/`

放按 agent 分组的完整 debug bundle：

```text
runs/persona_prompt_flywheel_demo/04_debug_bundles/output/
  feedback_labeling_agent/
  feature_discovery_agent/
  modeling_code_agent/
  prompt_patch_generator_agent/
```

每个目录里会有该阶段输入快照、输出清单和相关产物副本：

- `input_snapshot.json` / `input_snapshot.jsonl`：该 agent 执行时看到的输入快照，方便复盘当时使用了哪些数据和配置。
- `output_manifest.json`：该 agent 的输出清单，包含复制了哪些输出文件、缺失哪些文件、以及该阶段 metadata。
- `<agent_name>_handoff.json` / `<agent_name>_handoff.md`：该阶段 agent 的 LLM 检查结果和给下一阶段的交接说明副本。
- 其他同名产物副本：例如 `04_debug_bundles/output/modeling_code_agent/model_report.json` 是建模阶段的产物副本，便于只查看建模阶段的完整 debug bundle。

这 4 个目录分别对应：核心结果、阶段产物、agent 交接、debug 证据。这样既保留可审计性，又避免所有文件平铺在 run 根目录。

## 人工复核闭环

不确定样本会进入：

```text
runs/persona_prompt_flywheel_demo/human_review_queue.jsonl
```

人工复核后，可以创建 JSONL：

```json
{"trace_id":"t-uncertain-001","resolved_label":"neutral_continuation_no_feedback","label_update":{"description":"用户说还行或说不上来但未否定上一轮回答时，标为中性继续交互。","training_target_positive":0}}
```

合并回标签配置：

```bash
python examples/persona_prompt_flywheel/run_flywheel.py \
  --merge-human-labels runs/persona_prompt_flywheel_demo/human_resolved_labels.jsonl \
  --rules examples/persona_prompt_flywheel/config/feedback_labeling_rules.yaml
```

这会更新 `feedback_labeling_rules.yaml` 中的 `labels`，并递增 `version`。

## 常见修改

新增反馈标签：

```yaml
labels:
  partially_helpful_feedback:
    description: 用户认为上一轮回答部分有帮助，但仍指出遗漏或需要补充。
    training_target_positive:
```

新增主观特征：

```yaml
subjective_features:
  warmth_score:
    type: ordinal_1_5
    rubric: 回答是否语气温和、让用户感到被认真对待。
```

调整客观特征定义：

```yaml
objective_features:
  action_item_count:
    type: numeric
    description: 回答中有多少个用户可以执行的具体行动项。
    extraction_rubric: 统计明确可执行的下一步、检查项、联系对象、操作命令、预约或复诊建议数量。
```

调整 prompt patch 规则：

```yaml
prompt_patch_generation:
  patch_text:
    base: 回答时先用 1-2 句话给出方向或下一步判断框架。
    llm_generation_guidance:
      - 根据 positive_drivers 和 negative_drivers 的语义生成 patch，不要用字符串包含关系匹配规则。
      - 如果洞察显示阅读负担高，降低信息密度，优先给出最关键的 2-3 点。
```

## 测试

在 `examples/persona_prompt_flywheel` 目录运行：

```bash
uv run --no-project --python /usr/bin/python3 \
  --with pytest --with pyyaml --with numpy --with scikit-learn --with joblib \
  python -m pytest -c /dev/null --confcutdir=. "tests"
```

当前测试覆盖输入归一化、LLM feedback labeling mock 路径、特征抽取、全链路 smoke pipeline 和 prompt patch registry。

## 设计边界

标签、prompt、阈值、特征定义、分桶、模型超参、patch 文案等可调策略都放在 YAML 中。

Python 中仍保留实现逻辑，例如 CSV 读取、确定性文本统计、LLM 调用编排、特征矩阵构建、逻辑回归训练、文件写出等。机械统计类特征放在 `deterministic_objective_features`；语义判断类客观特征放在 `objective_features`，通过 `description` 和 `extraction_rubric` 交给 LLM 判断数量或二值结果。

## 为什么使用 LLM-backed agents + 受控工具

相比纯 Python 写死 workflow，当前模式的优势是：

- `agents.yaml` 和 `tasks.yaml` 明确描述每个阶段的职责、目标、输出、允许调用的工具、检查项和 handoff contract，pipeline 结构更容易被人审阅。
- 每个阶段都有独立 agent 名称、task 配置、输入快照、输出清单、debug bundle 和 LLM 生成的 handoff 摘要，出问题时更容易定位是哪一段失败。
- 执行能力仍然被限制在受控工具里，agent 不能随意写训练代码、乱改标签体系或绕过固定输出格式，因此可复现性和审计性比完全 agentic 的流程更强。
- agent 不是只作为目录名存在；它会在工具执行后用 LLM 检查产物、总结风险、给下一个 agent 提供上下文。
- 后续如果要增加人工复核 agent、实验设计 agent、报告审查 agent，可以在 CrewAI task/tool 层扩展，而不需要把所有流程都继续塞进一个 Python 主函数。
- 业务策略仍然集中在 YAML 中，工程实现集中在 tool 中，配置、编排和执行边界更清晰。
