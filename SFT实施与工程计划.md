# SFT 实施与工程计划

## 1. 本版结论

- 保留“先 SFT、后 DPO”的总路线。
- `answerable` 样本采用“gold 主干 + teacher 受约束增广”的方案。
- `unanswerable` 样本改为独立管线：`paired + external + NLI prefilter + teacher refusal rationale + dedicated validate`。
- `confidence` 字段在 `answerable` 与 `unanswerable` 上分开处理：
  - `answerable`: 训练前重写为 `derived_confidence`
  - `unanswerable`: 暂保留 teacher 原值
- 结构化输出的 JSON 字段顺序暂固定为：
  - `answerability`
  - `evidence`
  - `rationale`
  - `answer`
  - `confidence`
- 拒答模板使用小集合，训练集构建时默认收敛为 canonical 模板，检查时允许模板集合。

## 2. 数据与监督策略

### 2.1 answerable

- 数据源仍为 HotpotQA。
- 原始 gold `reference_answer` 不强行改写为固定句式。
- teacher 的职责不是替代 gold，而是在约束下输出更适合结构化训练的样本：
  - `evidence` 必须从 supporting facts 或 support window 中抽取。
  - `rationale` 必须给出从证据到答案的简短推导，且不能与 `answer` 完全相同。
  - `answer` 必须与 baseline 语义一致。
  - 在正确度保持的前提下，尽量保留原始答案分布和形式。
- 自动选择 teacher 候选时，优先满足：
  - schema 合法；
  - quote 合法；
  - 与 reference answer 一致；
  - 语义支持更强。
- 若 teacher 候选未通过约束，则回退到 baseline target。

### 2.2 unanswerable

- 工作流与 `answerable` 解耦，但复用相同的协议、teacher API 和底层 I/O 工具。
- raw 构造来源分为两类：
  - `paired_answerable`: 从已选 `answerable` 样本中构造近似一半
  - `external_raw`: 从同 split 的剩余 Hotpot raw 池中补齐剩余部分
- 构造规则：
  - 先按 `answerable` 相同参数构造 supporting scaffold
  - 再将 1 条或多条 supporting fact 替换为邻句、同段落非-support 句，或临近段落句
  - 目标是保持 knowledge 长度与表面分布接近，而不是简单变短
- teacher 生成前必须经过 `full_binary` NLI 先验判负：
  - hypothesis 固定为 `The answer is {reference_answer}.`
  - 仅保留 `full_binary = no` 的样本
- teacher 生成时不提供 `reference_answer`
- teacher 的 `unanswerable` completion 仍需满足协议，但 `rationale` 必须指出知识中的缺失、歧义或矛盾
- `unanswerable` validate 使用独立流程：
  - 只做 canonicalization、parse、protocol、answerability 一致性、completion token budget
  - 不做 evidence / correctness / semantic 检查
  - 暂不做软排序，直接选择首个 hard-pass 候选

### 2.3 confidence

- schema 中保留 `high | medium | low`。
- `answerable`：
  - 训练前将 teacher 给出的 `confidence` 覆盖为 `derived_confidence`
  - `derived_confidence` 由 answerable validator 的 semantic margin 分位数生成
- `unanswerable`：
  - v1 暂保留 teacher 原始 `confidence`
  - 不要求 `derived_confidence`

## 3. 包结构

```text
src/
  grounded_qa/
    cli.py
    config.py
    workflow.py

  qa_protocol/
    __init__.py
    answer_style.py
    normalize.py
    parsing.py
    prompting.py
    refusal.py
    schema.py
    spec.py

  qa_data/
    __init__.py
    construct.py
    export.py
    hotpot.py
    hotpot_raw.py
    manifest.py
    negatives.py
    records.py
    split.py
    unanswerable.py

  qa_checks/
    __init__.py
    correctness.py
    evidence.py
    protocol.py
    report.py
    selection.py
    semantics.py
    unanswerable_prefilter.py
    unanswerable_selection.py

  qa_judge/
    structured.py

  llm_textgen/
    api_client.py

  sft_trainer/
    __init__.py
    formatting.py
    prepare_sft_dataset.py
    train_sft.py

  model_eval/
    eval_grounded_qa.py
    generate_structured_answers.py
```

## 4. 目录职责

### 4.1 `grounded_qa`

- 只负责任务级 CLI 和工作流编排。
- 保持薄封装，不承载协议和检查细节。

### 4.2 `qa_protocol`

- 结构化输出 schema。
- prompt builder。
- JSON parse / canonical serialization。
- refusal 模板与答案形式策略。

### 4.3 `qa_data`

- HotpotQA 读取。
- split-assigned raw Hotpot 导出。
- answerable 构造。
- legacy simple unanswerable 派生。
- v1 unanswerable 构造。
- split 分配。
- dataclass 与导出转换。

### 4.4 `qa_checks`

- protocol 检查。
- evidence quote 检查。
- unanswerable NLI 先验判负。
- correctness 检查。
- 结构化语义检查适配。
- teacher 候选选择与 `derived_confidence` 打分。
- unanswerable 首候选选择。

### 4.5 `qa_judge.structured`

- 将现有 `qa_judge` 的 plain QA verifier 适配到结构化输出。
- answerable 样本优先用 evidence quote 拼接后的支持文本做验证。
- 不再回退到 support window knowledge。

### 4.6 `sft_trainer`

- 将验证通过的结构化记录打包为 SFT `prompt/completion` 数据集。
- 提供 LoRA SFT 训练入口。

### 4.7 `model_eval`

- 继续承接模型生成与 checkpoint 对比。
- 新增结构化输出生成与 grounded QA 指标评估脚本。

## 5. 中间产物合同

### 5.1 `raw_examples.jsonl`

- `id`
- `source_id`
- `variant_id`
- `split`
- `question`
- `knowledge`
- `reference_answer`
- `answerability_label`
- `difficulty`
- `supporting_sentences`
- `context_documents`
- `metadata`

### 5.2 `hotpot_rows_split.jsonl`

- `_id`
- `question`
- `answer`
- `type`
- `level`
- `context`
- `supporting_facts`
- `_source_index`
- `split`

### 5.3 `unanswerable_prefiltered.jsonl`

- 保留 `raw_examples` 字段；
- 在 `metadata` 中新增：
  - `origin_track`
  - `negative_strategy`
  - `nli_prefilter`

### 5.4 `teacher_candidates.jsonl`

- `id`
- `source_id`
- `candidate_id`
- `teacher_model`
- `prompt_style`
- `prompt`
- `raw_output`

### 5.5 `validated_candidates.jsonl`

- 保留 `teacher_candidates` 字段；
- 新增：
  - `parse_ok`
  - `parsed_output`
  - `canonical_output`
  - `validation_report`
  - `selection_score`
  - `derived_confidence`

### 5.6 `validated_unanswerable_candidates.jsonl`

- 保留 `teacher_candidates` 字段；
- 新增：
  - `parse_ok`
  - `parsed_output`
  - `canonical_output`
  - `validation_report`
- 其中：
  - `correctness = null`
  - `semantics = null`
  - `derived_confidence = null`

### 5.7 `sft_records.jsonl`

- `id`
- `source_id`
- `split`
- `prompt_style`
- `prompt`
- `completion`
- `target_structured`
- `metadata`

## 6. CLI 计划

```bash
python -m src.grounded_qa.cli source hotpot
python -m src.grounded_qa.cli source hotpot-raw
python -m src.grounded_qa.cli source split
python -m src.grounded_qa.cli source unanswerable
python -m src.grounded_qa.cli source unanswerable-prefilter
python -m src.grounded_qa.cli teacher generate
python -m src.grounded_qa.cli teacher validate
python -m src.grounded_qa.cli teacher validate-unanswerable
python -m src.grounded_qa.cli build sft-records

python -m src.sft_trainer.prepare_sft_dataset
accelerate launch -m src.sft_trainer.train_sft

python -m src.model_eval.generate_structured_answers
python -m src.model_eval.eval_grounded_qa
```

## 7. 实施顺序

1. 落协议层与数据 dataclass。
2. 打通 Hotpot source、split-assigned raw Hotpot、answerable split。
3. 打通 v1 unanswerable raw 构造与 NLI prefilter。
4. 打通 teacher prompt、生成、answerable validate、unanswerable validate。
5. 打通 SFT records 与 DatasetDict 导出。
6. 接入 LoRA SFT 训练入口。
7. 接入结构化生成与评估脚本。
8. 在该骨架稳定后，再对接结构化 DPO。

## 8. 本次骨架实现范围

- 新增协议层、数据层、检查层与任务入口骨架。
- 落可运行的 SFT 数据打包与训练脚手架。
- 落结构化生成与评估脚手架。
- 不在本次内完成：
  - 复杂负例难度调优；
  - unanswerable teacher 候选的高级 rerank；
  - DPO 结构化 pair builder；
  - calibration 桶阈值的实验调优。
