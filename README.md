# Grounded-QA SFT Pipeline

本仓库现已收敛为 **SFT-only** 工程，只负责：

1. Grounded-QA 训练数据构造
2. SFT 数据集准备与训练
3. base / ckpt 的结构化评测、DeepEval 评测与报告汇总

此前的 DPO、`meqng`、`ssqpg` 及其配套路线已经删除，不再作为当前研究路径的一部分。

## 目录说明

- `src/grounded_qa`: source 构造、teacher generate/validate、SFT record 打包
- `src/sft_trainer`: SFT 数据集准备与训练脚本
- `src/model_eval`: validation/test 生成、结构评测、task baseline、DeepEval、报告编排
- `src/qa_protocol`, `src/qa_checks`, `src/qa_judge`, `src/qa_data`: 共享协议、检查器、判定器和数据构造工具
- `src/llm_textgen`: 本地模型与 API 生成封装

## 当前数据工作流

### 1. Source

`source tag -> source build -> source prefilter -> source partition`

- `source tag` 同时读取 HotpotQA distractor 的 `train` 与 `validation`
- Hotpot `train` 全部进入 `train_sft_raw`
- Hotpot `validation` 再按稳定哈希切成 `validation` / `test`
- `answerability_split` 与 `data_split` 保持独立

### 2. Teacher

`teacher generate -> teacher validate -> build sft-records`

- 输入是 mixed answerable / unanswerable 样本
- `teacher validate` 内部分流 answerable / unanswerable 检查链
- answerable 使用 derived confidence
- unanswerable 保留 teacher/self-derived 置信度路径

### 3. SFT 与评测

- `prepare_sft_dataset.py` 从 train / validation / test 三个来源准备 `DatasetDict`
- `train_sft.py` 训练 base -> LoRA ckpt
- `run_sft_eval_report.py` 在 validation 上选择最佳 ckpt，并在 test 上对比：
  - best ckpt structured
  - base protocol
  - base task no-think
  - base task think

## 当前主命令

### Source 与 teacher

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
```

### SFT 数据集与训练

```bash
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
```

### 评测与报告

```bash
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

## 说明

- 结构化评测口径见 `src/model_eval/eval_grounded_qa.py`
- task baseline 使用 LLM 做答案抽取与拒答识别，再做 strict/matcher correctness 与语义判定
- DeepEval 已支持 structured 模式，直接对 `rationale + answer` 做 hallucination 评测
