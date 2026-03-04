# SFT 实施与工程计划

## 1. 当前设计结论

- 工作流按 `source -> teacher -> build` 三大阶段组织。
- `data_split` 与 `answerability_split` 已解耦：
  - `data_split`: `validation / test / train_sft_raw / train_dpo_raw`
  - `answerability_split`: `answerable / unanswerable / both`
- `source` 阶段固定为四步：
  - `tag`
  - `build`
  - `prefilter`
  - `partition`
- `teacher` 阶段固定为两步：
  - `generate`
  - `validate`
- `teacher validate` 接受混合样本，内部按 `answerability_label` 分流：
  - `answerable` 走完整检查链
  - `unanswerable` 走简化拒答检查链

## 2. 双标签数据组织

### 2.1 原始 Hotpot 行

在任何知识构造之前，先对原始 Hotpot 样本打两个独立标签：

- `data_split`
- `answerability_split`

两套标签使用不同盐值的稳定哈希函数分配，保证统计上独立。

### 2.2 默认比例

- `data_split`
  - `validation_ratio = 0.05`
  - `test_ratio = 0.05`
  - `train_sft_ratio = 0.45`
  - `train_dpo_ratio = 0.45`
- `answerability_split`
  - `answerable_ratio = 0.8`
  - `unanswerable_ratio = 0.1`
  - `both_ratio = 0.1`

`both` 表示同一 raw source 同时生成一条 `answerable` concrete 样本和一条 `unanswerable` concrete 样本。

## 3. Concrete 样本结构

统一使用 `QaExample`：

- `id`
- `source_id`
- `variant_id`
- `data_split`
- `answerability_split`
- `question`
- `knowledge`
- `reference_answer`
- `answerability_label`
- `difficulty`
- `supporting_sentences`
- `context_documents`
- `metadata`

说明：

- `answerability_split` 是原始样本的构造轨道标签。
- `answerability_label` 是当前 concrete 样本自身的标签。
- 旧顶层字段 `split` 已删除，不再保留兼容别名。

## 4. Source 阶段

### 4.1 `source tag`

输入：Hotpot 原始样本。  
输出：`tagged_hotpot_rows.jsonl`

新增字段：

- `data_split`
- `answerability_split`

### 4.2 `source build`

输入：`tagged_hotpot_rows.jsonl`  
输出：`prepared_examples.jsonl`

构造规则：

- `answerability_split = answerable`
  - 只产一条 `answerable`
- `answerability_split = unanswerable`
  - 只产一条 `unanswerable`
- `answerability_split = both`
  - 同时产一条 `answerable` 与一条 `unanswerable`

### 4.3 `source prefilter`

输入：`prepared_examples.jsonl`  
输出：`prefiltered_examples.jsonl`

过滤顺序：

1. 所有样本统一做 `infer_prompt(schema2)` token 长度过滤
2. 仅对 `unanswerable` 样本做 `full_binary=no` NLI 判负过滤

这一步是唯一的 source 级 prompt 预算过滤入口。`teacher generate` 不再承担该职责。

### 4.4 `source partition`

输入：`prefiltered_examples.jsonl`  
输出目录：

- `validation.jsonl`
- `test.jsonl`
- `train_sft_raw.jsonl`
- `train_dpo_raw.jsonl`

每个文件内部仍然是混合的 `answerable / unanswerable` 样本。

## 5. Teacher 阶段

### 5.1 `teacher generate`

输入：混合 concrete 样本。  
输出：`teacher_candidates.jsonl`

规则：

- `answerable` 样本向 teacher prompt 提供 `reference_answer`
- `unanswerable` 样本不提供 `reference_answer`
- 不再进行 source 级 token 预算过滤

### 5.2 `teacher validate`

输入：混合候选。  
输出：

- `validated_candidates.jsonl`
- `selected_candidates.jsonl`

内部逻辑：

1. 读取混合候选
2. 按 `answerability_label` 分流
3. `answerable` 走完整 validate
4. `unanswerable` 走简化 validate
5. 合并 validated 与 selected 输出

#### `answerable` validate

- canonicalization
- parse
- protocol
- evidence
- correctness
- optional semantic
- soft ranking
- `derived_confidence`

#### `unanswerable` validate

- canonicalization
- parse
- protocol
- answerability 一致性
- completion token budget
- 不做 evidence / correctness / semantic
- 选 `candidate_id` 最小的首个 hard-pass 候选

## 6. SFT 打包

`build sft-records` 接受混合 selected 样本：

- `answerable`
  - 必须存在 `derived_confidence`
  - completion 中的 `confidence` 用 `derived_confidence` 覆盖
- `unanswerable`
  - 保留 teacher 原始 `confidence`
  - 不要求 `derived_confidence`

SFT record 顶层使用：

- `data_split`

不再使用旧字段 `split`。

## 7. 代码结构

```text
src/
  grounded_qa/
    cli.py
    config.py
    workflow.py

  qa_protocol/
    answer_style.py
    normalize.py
    parsing.py
    prompting.py
    refusal.py
    schema.py
    spec.py
    token_budget.py

  qa_data/
    construct.py
    export.py
    hotpot.py
    partition.py
    records.py
    tagging.py
    unanswerable.py

  qa_checks/
    correctness.py
    evidence.py
    protocol.py
    report.py
    selection.py
    semantics.py
    source_prefilter.py
    unanswerable_prefilter.py
    unanswerable_selection.py

  qa_judge/
    structured.py

  sft_trainer/
    formatting.py
    prepare_sft_dataset.py
    train_sft.py

  model_eval/
    generate_structured_answers.py
    eval_grounded_qa.py
```

## 8. 当前主命令集

```bash
python -m grounded_qa.cli source tag
python -m grounded_qa.cli source build
python -m grounded_qa.cli source prefilter
python -m grounded_qa.cli source partition

python -m grounded_qa.cli teacher generate
python -m grounded_qa.cli teacher validate

python -m grounded_qa.cli build sft-records
```

## 9. 当前全流程工作命令

运行前建议：

```bash
export PYTHONPATH=src
```

```bash
python -m grounded_qa.cli source tag \
  --out data/grounded_qa/tagged_hotpot_rows.jsonl \
  --split train \
  --dataset-name hotpotqa/hotpot_qa \
  --max-samples -1 \
  --seed 42 \
  --validation-ratio 0.05 \
  --test-ratio 0.05 \
  --train-sft-ratio 0.45 \
  --train-dpo-ratio 0.45 \
  --answerable-ratio 0.8 \
  --unanswerable-ratio 0.1 \
  --both-ratio 0.1 \
  --metrics-out outputs/grounded_qa/tag_metrics.json

python -m grounded_qa.cli source build \
  --in-path data/grounded_qa/tagged_hotpot_rows.jsonl \
  --out data/grounded_qa/prepared_examples.jsonl \
  --metrics-out outputs/grounded_qa/build_metrics.json

python -m grounded_qa.cli source prefilter \
  --in-path data/grounded_qa/prepared_examples.jsonl \
  --out data/grounded_qa/prefiltered_examples.jsonl \
  --metrics-out outputs/grounded_qa/prefilter_metrics.json

python -m grounded_qa.cli source partition \
  --in-path data/grounded_qa/prefiltered_examples.jsonl \
  --out-dir data/grounded_qa/partitioned \
  --metrics-out outputs/grounded_qa/partition_metrics.json

python -m grounded_qa.cli teacher generate \
  --in-path data/grounded_qa/partitioned/train_sft_raw.jsonl \
  --out data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --metrics-out outputs/grounded_qa/teacher-generate-train-sft-metrics.json

python -m grounded_qa.cli teacher validate \
  --in-path data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --out data/grounded_qa/validated_train_sft.jsonl \
  --selected-out data/grounded_qa/selected_train_sft.jsonl \
  --metrics-out outputs/grounded_qa/teacher-validate-train-sft-metrics.json

python -m grounded_qa.cli build sft-records \
  --in-path data/grounded_qa/selected_train_sft.jsonl \
  --out data/grounded_qa/sft_records_train_sft.jsonl \
  --prompt-style infer_v1 \
  --keep-only-overall-ok \
  --metrics-out outputs/grounded_qa/sft-records-train-sft-metrics.json
```
