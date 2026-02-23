# minimal_edit_qa_negative_generator

Primary pipeline: `ssqpg`  
Legacy route kept for compatibility: `meqng` (not the active research path).

## Environment Setup (uv)

### 1) Create and sync environment

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
```

### 2) Required runtime assets

```bash
python -m spacy download en_core_web_trf
```

### 3) Optional extras

```bash
# LoRA adapter loading in llm_textgen
uv sync --extra lora

# DeepEval-based evaluation scripts in model_eval
uv sync --extra eval

# Legacy optional components (old route)
uv sync --extra grammar --extra attack

# DPO training (TRL)
uv pip install trl
```

## Primary Pipeline (`ssqpg`)

The project is stage-driven. Run each stage explicitly and keep intermediate JSONL/JSON metrics.

### Stage 1: Source sampling

```bash
python -m src.ssqpg.cli source sample \
  --out data/ssqpg/source.jsonl \
  --source halueval \
  --max-samples 5000 \
  --include-reference-answer true
```

### Stage 2: Self-sampled answer generation

```bash
python -m src.ssqpg.cli generate answers \
  --in-path data/ssqpg/source.jsonl \
  --out data/ssqpg/generated.jsonl \
  --backend local \
  --model-name Qwen/Qwen3-0.6B \
  --num-samples-per-question 6
```

### Stage 3: Calibrated answer judgment (Method B + Method D)

```bash
python -m src.ssqpg.cli judge answers \
  --in-path data/ssqpg/generated.jsonl \
  --out data/ssqpg/judged.jsonl
```

### Stage 4: Answer filter (drop abstain by default)

```bash
python -m src.ssqpg.cli pair answer-filter \
  --in-path data/ssqpg/judged.jsonl \
  --out data/ssqpg/judged_filtered.jsonl
```

### Stage 5: Group summary (inspection step)

```bash
python -m src.ssqpg.cli pair group \
  --in-path data/ssqpg/judged_filtered.jsonl \
  --out data/ssqpg/grouped.jsonl
```

### Stage 6: Hard augmentation (rejudge + re-filter)

```bash
python -m src.ssqpg.cli pair hard-augment \
  --in-path data/ssqpg/judged_filtered.jsonl \
  --out data/ssqpg/judged_augmented.jsonl \
  --extra-samples-per-hard 0 \
  --enable-api-repair true
```

### Stage 7: Pairing

```bash
python -m src.ssqpg.cli pair pairing \
  --in-path data/ssqpg/judged_augmented.jsonl \
  --out data/ssqpg/pairs.jsonl
```

### Stage 8: Final pair filtering (no NLI re-run)

```bash
python -m src.ssqpg.cli filter apply \
  --in-path data/ssqpg/pairs.jsonl \
  --out data/ssqpg/dpo_pairs.jsonl
```

### Stage 9: Surface-signal audit

```bash
python -m src.ssqpg.cli audit surface \
  --in-path data/ssqpg/dpo_pairs.jsonl \
  --out data/ssqpg/surface_audit.json
```

### Stage 10: Difficulty bucketing

```bash
python -m src.ssqpg.cli bucket difficulty \
  --in-path data/ssqpg/dpo_pairs.jsonl \
  --out data/ssqpg/dpo_pairs_bucketed.jsonl
```

## DPO Train + Eval Routes

### Route A: Train on `ssqpg` pairs and evaluate

```bash
python3 -m src.dpo_trainer.prepare_dpo_dataset \
  --in_jsonl data/ssqpg/dpo_pairs_bucketed.jsonl \
  --output_dir data/dpo/ssqpg_pairs \
  --build_halueval_compare \
  --compare_output_dir data/dpo/halueval_pairs

accelerate launch -m src.dpo_trainer.train_dpo \
  --data_dir data/dpo/ssqpg_pairs \
  --output_dir ckpt/dpo_lora_qwen3_06b_ssqpg \
  --precompute_ref_log_probs

python -m src.model_eval.generate_answers \
  --data_path data/dpo/ssqpg_pairs \
  --out_jsonl outputs/model_eval/ssqpg_dpo.jsonl \
  --lora_ckpt_list_path ckpt/dpo_lora_qwen3_06b_ssqpg \
  --include_base \
  --use_chat_template

python -m src.model_eval.eval_pairwise_preference \
  --data_dir data/dpo/ssqpg_pairs \
  --out_csv outputs/model_eval/ssqpg_dpo_pairwise_preference_curve.csv \
  --lora_ckpt_list_path ckpt/dpo_lora_qwen3_06b_ssqpg \
  --include_base

python -m src.model_eval.eval_nli_faithfulness \
  --generated_path outputs/model_eval/ssqpg_dpo.jsonl \
  --split test \
  --out_csv outputs/model_eval/ssqpg_dpo_nli_faithfulness_curve.csv \
  --out_jsonl outputs/model_eval/ssqpg_dpo_nli_faithfulness_samples.jsonl

python -m src.model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/ssqpg_dpo.jsonl \
  --metrics_out outputs/model_eval/ssqpg_dpo_deepeval_hallucination_curve.csv \
  --details_out outputs/model_eval/ssqpg_dpo_deepeval_hallucination_samples.jsonl
```

### Route B: Train on `halueval` compare pairs and evaluate

```bash
accelerate launch -m src.dpo_trainer.train_dpo \
  --data_dir data/dpo/halueval_pairs \
  --output_dir ckpt/dpo_lora_qwen3_06b_halueval \
  --precompute_ref_log_probs

python -m src.model_eval.generate_answers \
  --data_path data/dpo/halueval_pairs \
  --out_jsonl outputs/model_eval/halueval_dpo.jsonl \
  --lora_ckpt_list_path ckpt/dpo_lora_qwen3_06b_halueval \
  --include_base \
  --use_chat_template

python -m src.model_eval.eval_pairwise_preference \
  --data_dir data/dpo/halueval_pairs \
  --out_csv outputs/model_eval/halueval_dpo_pairwise_preference_curve.csv \
  --lora_ckpt_list_path ckpt/dpo_lora_qwen3_06b_halueval \
  --include_base

python -m src.model_eval.eval_nli_faithfulness \
  --generated_path outputs/model_eval/halueval_dpo.jsonl \
  --split test \
  --out_csv outputs/model_eval/halueval_dpo_nli_faithfulness_curve.csv \
  --out_jsonl outputs/model_eval/halueval_dpo_nli_faithfulness_samples.jsonl

python -m src.model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/halueval_dpo.jsonl \
  --metrics_out outputs/model_eval/halueval_dpo_deepeval_hallucination_curve.csv \
  --details_out outputs/model_eval/halueval_dpo_deepeval_hallucination_samples.jsonl
```

## Command Runner

Use `scripts/run_command_sequence.sh` to run a sequence of commands in order.

```bash
# from a command file (one command per line; empty lines and # comments are ignored)
bash scripts/run_command_sequence.sh --file path/to/commands.txt

# or pass commands directly
bash scripts/run_command_sequence.sh \
  -c "python -m src.ssqpg.cli version" \
  -c "python -m src.model_eval.generate_answers --help"
```

## Evaluation Utilities (`model_eval`)

Run them from source, for example:

```bash
python -m src.model_eval.generate_answers --help
python -m src.model_eval.eval_nli_faithfulness --help
python -m src.model_eval.eval_deepeval_hallucination --help
python -m src.model_eval.eval_pairwise_preference --help
```

## Notes on Naming

- Active judge package: `qa_judge`.
- Active data pipeline: `ssqpg`.
- Legacy `meqng` code is preserved but not the primary research route.
