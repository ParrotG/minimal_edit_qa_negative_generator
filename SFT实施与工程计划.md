# SFT实施与工程计划

## 1. 工程范围

本工程现仅保留 SFT 阶段相关能力：

- grounded QA source 构造
- teacher structured completion 生成与验证
- SFT 数据集准备与训练
- base / ckpt 的 validation / test 评测与报告

以下内容已不再属于当前工程范围：

- DPO 数据构造
- DPO 训练与 pairwise preference 评测
- `meqng`
- `ssqpg`

## 2. 当前数据模型

### 2.1 Raw tagging

原始 Hotpot 行在 `source tag` 后同时带两类标签：

- `data_split`
  - `train_sft_raw`
  - `validation`
  - `test`
- `answerability_split`
  - `answerable`
  - `unanswerable`
  - `both`

另保留：

- `hotpot_source_split`
  - `train`
  - `validation`

规则固定为：

- Hotpot `train` -> `train_sft_raw`
- Hotpot `validation` -> 再切成 `validation/test`

### 2.2 Concrete example

`QaExample` 顶层使用：

- `data_split`
- `answerability_split`
- `answerability_label`

其中：

- `answerability_split` 表示原始样本走哪条构造轨
- `answerability_label` 表示当前 concrete 样本本身是 `answerable` 还是 `unanswerable`

## 3. 当前主流程

### 3.1 Source

`source tag -> source build -> source prefilter -> source partition`

- `source tag` 同时读取 Hotpot distractor 的 `train` 和 `validation`
- `source build` 输出 mixed answerable / unanswerable concrete 样本
- `source prefilter` 统一做 infer prompt token 过滤，并对 unanswerable 做 NLI 判负过滤
- `source partition` 只按 `data_split` 切出：
  - `train_sft_raw.jsonl`
  - `validation.jsonl`
  - `test.jsonl`

### 3.2 Teacher

`teacher generate -> teacher validate -> build sft-records`

- `teacher generate` 接受 mixed source
- `teacher validate` 内部按 `answerability_label` 分流 answerable / unanswerable 检查
- answerable 使用 derived confidence
- unanswerable 保留其自身置信度路径

### 3.3 SFT 与评测

- `prepare_sft_dataset` 接收 train / validation / test 三个来源
- validation 需要 completion 以计算 loss
- test 不需要 completion，可直接来自 source partition 的产物
- `run_sft_eval_report` 在 validation 上做：
  - SFT loss
  - structured generation
  - grounded evaluation
  - 最佳 ckpt 选择
- 然后在 test 上比较：
  - best ckpt structured
  - base protocol
  - base task no-think
  - base task think

## 4. 包结构

当前保留的主要包：

- `grounded_qa`
- `sft_trainer`
- `model_eval`
- `qa_protocol`
- `qa_checks`
- `qa_judge`
- `qa_data`
- `llm_textgen`
- `dataio`

删除的包：

- `dpo_trainer`
- `meqng`
- `ssqpg`

## 5. 当前评测约定

### 5.1 Structured grounded eval

`eval_grounded_qa.py` 当前输出至少包含：

- `parse_ok_rate`
- `protocol_ok_rate_given_parse_ok`
- `answerability_accuracy`
- `evidence_substring_ok_rate`
- `correctness_strict_rate`
- `correctness_reviewed_rate`
- `semantic_yes_rate`
- `semantic_margin_mean`
- `confidence_reliability_auroc`

### 5.2 DeepEval

`eval_deepeval_hallucination.py` 支持两种输入模式：

- `flat`
- `structured`

`structured` 模式会从结构输出中提取：

`rationale + "\nTherefore the answer is " + answer`

### 5.3 合并曲线

`merge_eval_curves.py` 纵向合并以下 validation 曲线：

- `sft_val_structured_curve.csv`
- `base_protocol_curve.csv`
- `base_task_think_curve.csv`
- `base_task_nothink_curve.csv`

并统一保留：

- `model_tag`
- `model_step`
- `model_path`
- `eval_track`
- `eval_variant`

## 6. 当前全流程工作命令

```bash
python -m grounded_qa.cli source tag \
  --out data/grounded_qa/tagged_hotpot_rows.jsonl \
  --dataset-name hotpotqa/hotpot_qa \
  --train-split train \
  --validation-split validation \
  --max-train-samples -1 \
  --max-validation-samples -1 \
  --seed 42 \
  --validation-ratio 0.5 \
  --test-ratio 0.5 \
  --answerable-ratio 0.8 \
  --unanswerable-ratio 0.1 \
  --both-ratio 0.1 \
  --metrics-out outputs/grounded_qa/tag_metrics.json

python -m grounded_qa.cli source build \
  --in-path data/grounded_qa/tagged_hotpot_rows.jsonl \
  --out data/grounded_qa/prepared_examples.jsonl \
  --window-size 1 \
  --max-supporting-facts 4 \
  --replace-supporting-facts-min 1 \
  --replace-supporting-facts-max 1 \
  --same-doc-candidate-radius 1 \
  --allow-same-doc-non-adjacent \
  --adjacent-doc-sentence-limit 1 \
  --include-title-prefix \
  --seed 42 \
  --metrics-out outputs/grounded_qa/build_metrics.json

python -m grounded_qa.cli source prefilter \
  --in-path data/grounded_qa/prepared_examples.jsonl \
  --out data/grounded_qa/prefiltered_examples.jsonl \
  --tokenizer-name Qwen/Qwen3-0.6B \
  --max-prompt-tokens 512 \
  --enable-unanswerable-nli \
  --metrics-out outputs/grounded_qa/prefilter_metrics.json

python -m grounded_qa.cli source partition \
  --in-path data/grounded_qa/prefiltered_examples.jsonl \
  --out-dir data/grounded_qa/partitioned \
  --metrics-out outputs/grounded_qa/partition_metrics.json

python -m grounded_qa.cli teacher generate \
  --in-path data/grounded_qa/partitioned/train_sft_raw.jsonl \
  --out data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --answerable-num-candidates-per-example 3 \
  --unanswerable-num-candidates-per-example 1 \
  --api-model-name qwen3.5-plus \
  --api-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --api-key-env DASHSCOPE_API_KEY \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --top-p 0.95 \
  --metrics-out outputs/grounded_qa/teacher_generate_metrics.json

python -m grounded_qa.cli teacher validate \
  --in-path data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --out data/grounded_qa/validated_train_sft.jsonl \
  --selected-out data/grounded_qa/selected_train_sft.jsonl \
  --enable-semantics \
  --semantic-drop-by-nli \
  --semantic-decision-source full_binary \
  --tokenizer-name Qwen/Qwen3-0.6B \
  --max-completion-tokens 512 \
  --metrics-out outputs/grounded_qa/teacher_validate_metrics.json

python -m grounded_qa.cli build sft-records \
  --in-path data/grounded_qa/selected_train_sft.jsonl \
  --out data/grounded_qa/sft_records_train_sft.jsonl \
  --prompt-style infer_v1 \
  --keep-only-overall-ok \
  --metrics-out outputs/grounded_qa/sft_records_metrics.json

python -m sft_trainer.prepare_sft_dataset \
  --train-paths data/grounded_qa/sft_records_train_sft.jsonl \
  --validation-paths data/grounded_qa/sft_records_validation.jsonl \
  --test-paths data/grounded_qa/partitioned/test.jsonl \
  --output-dir data/sft_dataset \
  --overwrite-output \
  --max-prompt-tokens 512 \
  --max-completion-tokens 512 \
  --metrics-out outputs/sft/prepare_dataset_metrics.json

python -m sft_trainer.train_sft \
  --data-dir data/sft_dataset \
  --base-model Qwen/Qwen3-0.6B \
  --output-dir ckpt/sft_lora_qwen3_06b_groundedqa

python -m model_eval.run_sft_eval_report \
  --validation-data-path data/sft_dataset \
  --test-data-path data/sft_dataset \
  --validation-split validation \
  --test-split test \
  --base-model Qwen/Qwen3-0.6B \
  --lora-ckpt-list-path ckpt/sft_lora_qwen3_06b_groundedqa \
  --out-dir outputs/model_eval/final_report \
  --validation-max-samples 1000 \
  --test-max-samples 1000 \
  --structured-max-new-tokens 512 \
  --base-protocol-max-new-tokens 512 \
  --base-task-max-new-tokens 512 \
  --base-task-think-max-new-tokens 1024
```
