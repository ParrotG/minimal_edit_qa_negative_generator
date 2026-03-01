# SFT 实施与工程计划

## 1. 本版结论

- 保留“先 SFT、后 DPO”的总路线。
- `answerable` 样本采用“gold 主干 + teacher 受约束增广”的方案。
- `unanswerable` 样本首版采用简单派生，不追求强误导性，先保证可验证与可过滤。
- `confidence` 字段暂保留在 schema 中，但训练前使用检查器结果重写，不直接使用 teacher 自报值。
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

- 从 answerable 样本派生。
- 首版采用简单负例策略：
  - 删除至少一个 supporting fact 及其邻句窗口；
  - 保留段落剩余句子；
  - 必要时加入相邻段落的少量句子。
- 自动过滤阶段不追求“最难负例”，先保证：
  - 剩余 knowledge 确实不支持 reference answer；
  - 输出以规则化拒答形式呈现。

### 2.3 confidence

- schema 中保留 `high | medium | low`。
- 训练前将 teacher 给出的 `confidence` 覆盖为 `derived_confidence`。
- 首版的分桶策略：
  - `high`: 强约束全部通过，且 correctness / semantics 信号最强。
  - `medium`: 通过主约束，但匹配和支持信号弱于 `high`。
  - `low`: 通过最小训练约束，但只能判为低信心。
- 三档都保留进入训练，避免模型只学会输出高信心。

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
    manifest.py
    negatives.py
    records.py
    split.py

  qa_checks/
    __init__.py
    correctness.py
    evidence.py
    protocol.py
    report.py
    selection.py
    semantics.py

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
- answerable 构造。
- 简单 unanswerable 派生。
- split 分配。
- dataclass 与导出转换。

### 4.4 `qa_checks`

- protocol 检查。
- evidence quote 检查。
- rationale 检查。
- correctness 检查。
- 结构化语义检查适配。
- teacher 候选选择与 `derived_confidence` 打分。

### 4.5 `qa_judge.structured`

- 将现有 `qa_judge` 的 plain QA verifier 适配到结构化输出。
- answerable 样本优先用 evidence quote 拼接后的支持文本做验证。
- 必要时回退到 support window knowledge。

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

### 5.2 `teacher_candidates.jsonl`

- `id`
- `source_id`
- `candidate_id`
- `teacher_model`
- `prompt_style`
- `prompt`
- `raw_output`

### 5.3 `validated_candidates.jsonl`

- 保留 `teacher_candidates` 字段；
- 新增：
  - `parse_ok`
  - `parsed_output`
  - `canonical_output`
  - `validation_report`
  - `selection_score`
  - `derived_confidence`

### 5.4 `sft_records.jsonl`

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
python -m src.grounded_qa.cli source negatives
python -m src.grounded_qa.cli source split
python -m src.grounded_qa.cli teacher generate
python -m src.grounded_qa.cli teacher validate
python -m src.grounded_qa.cli build sft-records

python -m src.sft_trainer.prepare_sft_dataset
accelerate launch -m src.sft_trainer.train_sft

python -m src.model_eval.generate_structured_answers
python -m src.model_eval.eval_grounded_qa
```

## 7. 实施顺序

1. 落协议层与数据 dataclass。
2. 打通 Hotpot source、简单 negative、split。
3. 打通 teacher prompt、生成、验证、选优。
4. 打通 SFT records 与 DatasetDict 导出。
5. 接入 LoRA SFT 训练入口。
6. 接入结构化生成与评估脚本。
7. 在该骨架稳定后，再对接结构化 DPO。

## 8. 本次骨架实现范围

- 新增协议层、数据层、检查层与任务入口骨架。
- 落可运行的 SFT 数据打包与训练脚手架。
- 落结构化生成与评估脚手架。
- 不在本次内完成：
  - 复杂负例难度调优；
  - teacher 候选的高级 rerank；
  - DPO 结构化 pair builder；
  - calibration 桶阈值的实验调优。
