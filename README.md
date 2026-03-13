# Grounded-QA SFT Pipeline

This repository provides an end-to-end pipeline for grounded question answering with supervised fine-tuning (SFT). It prepares grounded-QA training data from HotpotQA, generates structured supervision with a teacher model, trains an SFT model, and evaluates base models and checkpoints with protocol-aware groundedness metrics.

The project is designed for research on grounded factuality under explicit evidence constraints. The structured target format requires the model to predict answerability, extract evidence, produce a short rationale, return an answer, and self-report confidence. The surrounding tooling supports data construction, filtering, calibration, training, validation-time checkpoint selection, and final test-time reporting.

## Repository Layout

- `src/grounded_qa`: source construction, teacher generation/validation, and SFT-record packing
- `src/sft_trainer`: dataset preparation and SFT training
- `src/model_eval`: validation/test generation, grounded evaluation, DeepEval, and report orchestration
- `src/calibrate`: optional calibration utilities for NLI and qa-metrics agreement analysis
- `src/qa_protocol`, `src/qa_checks`, `src/qa_judge`, `src/qa_data`: shared protocol, checking, judging, and data helpers
- `src/llm_textgen`: local and API-based text generation
- `src/project_config`: centralized default settings for models, tokenizers, APIs, and calibration

## Requirements

- Python 3.11+
- A GPU environment for local model generation, SFT training, and NLI evaluation
- Required environment variables for API-backed steps:
  - `DASHSCOPE_API_KEY` for teacher generation and answer extraction
  - Any additional key required by your DeepEval judge model, if you enable that stage

Optional dependency groups:

- `pip install -e .[lora]` for LoRA training
- `pip install -e .[eval]` for DeepEval

## End-to-End Workflow

The standard execution order is:

1. Source tagging
2. Mixed source building
3. Source prefiltering
4. Source partitioning
5. Teacher generation
6. Teacher validation
7. SFT record building
8. SFT dataset preparation
9. SFT training
10. Validation-time checkpoint selection and final evaluation summaries

Validation is used only for SFT checkpoint selection. Baseline comparisons are generated and evaluated only on the test split.

## Minimal Usage

### 1. Source Tagging

```bash
python -m grounded_qa.cli source tag \
  --out data/grounded_qa/tagged_hotpot_rows.jsonl \
  --metrics-out outputs/grounded_qa/tag_metrics.json
```

### 2. Source Building

```bash
python -m grounded_qa.cli source build \
  --in-path data/grounded_qa/tagged_hotpot_rows.jsonl \
  --out data/grounded_qa/prepared_examples.jsonl \
  --metrics-out outputs/grounded_qa/build_metrics.json
```

### 3. Source Prefiltering

```bash
python -m grounded_qa.cli source prefilter \
  --in-path data/grounded_qa/prepared_examples.jsonl \
  --out data/grounded_qa/prefiltered_examples.jsonl \
  --metrics-out outputs/grounded_qa/prefilter_metrics.json
```

### 4. Source Partitioning

```bash
python -m grounded_qa.cli source partition \
  --in-path data/grounded_qa/prefiltered_examples.jsonl \
  --out-dir data/grounded_qa/partitioned \
  --metrics-out outputs/grounded_qa/partition_metrics.json
```

### 5. Teacher Generation

```bash
python -m grounded_qa.cli teacher generate \
  --in-path data/grounded_qa/partitioned/train_sft_raw.jsonl \
  --out data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --metrics-out outputs/grounded_qa/teacher_generate_metrics.json
```

### 6. Teacher Validation

```bash
python -m grounded_qa.cli teacher validate \
  --in-path data/grounded_qa/teacher_candidates_train_sft.jsonl \
  --out data/grounded_qa/validated_train_sft.jsonl \
  --selected-out data/grounded_qa/selected_train_sft.jsonl \
  --metrics-out outputs/grounded_qa/teacher_validate_metrics.json
```

### 7. Build SFT Records

```bash
python -m grounded_qa.cli build sft-records \
  --in-path data/grounded_qa/selected_train_sft.jsonl \
  --out data/grounded_qa/sft_records_train_sft.jsonl \
  --metrics-out outputs/grounded_qa/sft_records_metrics.json
```

### 8. Prepare the SFT Dataset

```bash
python -m sft_trainer.prepare_sft_dataset \
  --train-paths data/grounded_qa/sft_records_train_sft.jsonl \
  --validation-paths data/grounded_qa/sft_records_validation.jsonl \
  --test-paths data/grounded_qa/partitioned/test.jsonl \
  --output-dir data/sft_dataset \
  --overwrite-output \
  --metrics-out outputs/sft/prepare_dataset_metrics.json
```

### 9. Train the SFT Model

```bash
python -m sft_trainer.train_sft \
  --data_dir data/sft_dataset \
  --output_dir ckpt/sft_lora_groundedqa
```

### 10. Run Validation Selection and Final Evaluation

```bash
python -m model_eval.run_sft_eval_report \
  --validation_data_path data/sft_dataset \
  --test_data_path data/sft_dataset \
  --lora_ckpt_list_path ckpt/sft_lora_groundedqa \
  --out_dir outputs/model_eval/final_report
```

This workflow writes two final summary CSV files:

- `outputs/model_eval/final_report/report/validation_summary.csv` for checkpoint selection on the validation split
- `outputs/model_eval/final_report/report/test_summary.csv` for final test-time comparison across the best checkpoint and the three baselines

### 10.1 Step-by-Step Model Evaluation

The one-command report workflow is convenient for final runs. For iterative development, the same evaluation pipeline can be executed step by step.

Validation split, SFT checkpoints only:

```bash
python -m model_eval.eval_sft_loss_curve \
  --data_path data/sft_dataset \
  --split validation \
  --lora_ckpt_list_path ckpt/sft_lora_groundedqa \
  --out_csv outputs/model_eval/validation/sft_val_loss_curve.csv
```

```bash
python -m model_eval.generate_structured_answers \
  --data_path data/sft_dataset \
  --split validation \
  --lora_ckpt_list_path ckpt/sft_lora_groundedqa \
  --retry_on_protocol_fail \
  --max_attempts 2 \
  --temperature 0.2 \
  --out_jsonl outputs/model_eval/validation/sft_val_structured_generations.jsonl
```

```bash
python -m model_eval.eval_grounded_qa \
  --generated_path outputs/model_eval/validation/sft_val_structured_generations.jsonl \
  --metrics_out outputs/model_eval/validation/sft_val_structured_curve.csv \
  --details_out outputs/model_eval/validation/sft_val_structured_details.jsonl \
  --confidence_out outputs/model_eval/validation/sft_val_confidence_analysis.json
```

```bash
python -m model_eval.build_validation_summary \
  --eval_curve_path outputs/model_eval/validation/sft_val_structured_curve.csv \
  --loss_curve_path outputs/model_eval/validation/sft_val_loss_curve.csv \
  --out_csv outputs/model_eval/validation/validation_summary.csv \
  --selection_out outputs/model_eval/validation/selection.json
```

Test split, best checkpoint after manual selection:

```bash
python -m model_eval.generate_structured_answers \
  --data_path data/sft_dataset \
  --split test \
  --lora_ckpt_path ckpt/sft_lora_groundedqa/checkpoint-XXX \
  --retry_on_protocol_fail \
  --max_attempts 2 \
  --temperature 0.2 \
  --out_jsonl outputs/model_eval/test/best_ckpt_structured_generations.jsonl
```

```bash
python -m model_eval.eval_grounded_qa \
  --generated_path outputs/model_eval/test/best_ckpt_structured_generations.jsonl \
  --metrics_out outputs/model_eval/test/best_ckpt_structured_curve.csv \
  --details_out outputs/model_eval/test/best_ckpt_structured_details.jsonl \
  --confidence_out outputs/model_eval/test/best_ckpt_structured_confidence.json
```

```bash
python -m model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/test/best_ckpt_structured_generations.jsonl \
  --input_mode structured \
  --metrics_out outputs/model_eval/test/best_ckpt_structured_deepeval_curve.csv \
  --details_out outputs/model_eval/test/best_ckpt_structured_deepeval_details.jsonl
```

Test split, base protocol baseline:

```bash
python -m model_eval.generate_structured_answers \
  --data_path data/sft_dataset \
  --split test \
  --include_base \
  --prompt_mode teacher_fewshot \
  --retry_on_protocol_fail \
  --max_attempts 3 \
  --temperature 0.2 \
  --out_jsonl outputs/model_eval/test/base_protocol_generations.jsonl
```

```bash
python -m model_eval.eval_grounded_qa \
  --generated_path outputs/model_eval/test/base_protocol_generations.jsonl \
  --metrics_out outputs/model_eval/test/base_protocol_curve.csv \
  --details_out outputs/model_eval/test/base_protocol_details.jsonl \
  --confidence_out outputs/model_eval/test/base_protocol_confidence.json
```

```bash
python -m model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/test/base_protocol_generations.jsonl \
  --input_mode structured \
  --metrics_out outputs/model_eval/test/base_protocol_deepeval_curve.csv \
  --details_out outputs/model_eval/test/base_protocol_deepeval_details.jsonl
```

Test split, base task baselines:

```bash
python -m model_eval.generate_answers \
  --data_path data/sft_dataset \
  --split test \
  --include_base \
  --out_jsonl outputs/model_eval/test/base_task_no_think_generations.jsonl
```

```bash
python -m model_eval.eval_task_content_baseline \
  --generated_path outputs/model_eval/test/base_task_no_think_generations.jsonl \
  --metrics_out outputs/model_eval/test/base_task_no_think_curve.csv \
  --details_out outputs/model_eval/test/base_task_no_think_details.jsonl
```

```bash
python -m model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/test/base_task_no_think_generations.jsonl \
  --input_mode flat \
  --answer_source answer \
  --metrics_out outputs/model_eval/test/base_task_no_think_deepeval_curve.csv \
  --details_out outputs/model_eval/test/base_task_no_think_deepeval_details.jsonl
```

```bash
python -m model_eval.generate_answers \
  --data_path data/sft_dataset \
  --split test \
  --include_base \
  --enable_thinking \
  --max_new_tokens 1024 \
  --out_jsonl outputs/model_eval/test/base_task_think_generations.jsonl
```

```bash
python -m model_eval.eval_task_content_baseline \
  --generated_path outputs/model_eval/test/base_task_think_generations.jsonl \
  --metrics_out outputs/model_eval/test/base_task_think_curve.csv \
  --details_out outputs/model_eval/test/base_task_think_details.jsonl
```

```bash
python -m model_eval.eval_deepeval_hallucination \
  --generated_path outputs/model_eval/test/base_task_think_generations.jsonl \
  --input_mode flat \
  --answer_source answer \
  --metrics_out outputs/model_eval/test/base_task_think_deepeval_curve.csv \
  --details_out outputs/model_eval/test/base_task_think_deepeval_details.jsonl
```

```bash
python -m model_eval.build_test_summary \
  --curve_path outputs/model_eval/test/best_ckpt_structured_curve.csv \
  --curve_path outputs/model_eval/test/base_protocol_curve.csv \
  --curve_path outputs/model_eval/test/base_task_no_think_curve.csv \
  --curve_path outputs/model_eval/test/base_task_think_curve.csv \
  --details_path outputs/model_eval/test/best_ckpt_structured_details.jsonl \
  --details_path outputs/model_eval/test/base_protocol_details.jsonl \
  --details_path outputs/model_eval/test/base_task_no_think_details.jsonl \
  --details_path outputs/model_eval/test/base_task_think_details.jsonl \
  --deepeval_details_path outputs/model_eval/test/best_ckpt_structured_deepeval_details.jsonl \
  --deepeval_details_path outputs/model_eval/test/base_protocol_deepeval_details.jsonl \
  --deepeval_details_path outputs/model_eval/test/base_task_no_think_deepeval_details.jsonl \
  --deepeval_details_path outputs/model_eval/test/base_task_think_deepeval_details.jsonl \
  --out_csv outputs/model_eval/test/test_summary.csv
```

## Optional Calibration

Calibration is optional. The repository already ships default judge settings, but you can recalibrate the task-level NLI and answer-equivalence judges against human annotations when needed.

The formal calibration workflow is:

1. Build an annotation pack from evaluation detail files
2. Manually fill `human_label` and optional `human_notes`
3. Evaluate the labeled pack with the judge models
4. Inspect `best_params.json` and `summary.csv`

The package name is `calibrate`. The previous `cablibrate` spelling was a typo and is no longer the public entry point.

### Recommended Inputs

- `nli_structured`: use `eval_grounded_qa` details, for example `outputs/model_eval/final_report/validation/sft_val_structured_details.jsonl`
- `nli_flat` and `matcher`: use `eval_task_content_baseline` details, for example `outputs/model_eval/final_report/test/base_task_think_details.jsonl`
- If you want to calibrate multiple sources together, repeat `--data_path` and the builder will sample from the combined pool

### Build an Annotation Pack

```bash
python -m calibrate.build_annotation_pack \
  --data_path outputs/model_eval/validation/sft_val_structured_details.jsonl \
  --data_path outputs/model_eval/test/base_protocol_details.jsonl \
  --data_path outputs/model_eval/test/base_task_think_details.jsonl \
  --data_path outputs/model_eval/test/base_task_no_think_details.jsonl \
  --task_types nli_structured,nli_flat,matcher \
  --out_jsonl outputs/calibration/annotation_pack.jsonl \
  --metrics_out outputs/calibration/annotation_pack_metrics.json
```

```bash
python -m calibrate.build_annotation_pack \
  --data_path outputs/model_eval/final_report/test/base_task_think_details.jsonl \
  --data_path outputs/model_eval/final_report/test/base_task_no_think_details.jsonl \
  --task_types nli_flat,matcher \
  --out_jsonl outputs/calibration/annotation_pack_flat_matcher.jsonl \
  --metrics_out outputs/calibration/annotation_pack_flat_matcher_metrics.json
```

Each annotation row contains the model-facing fields plus two human fields:

- `human_label`: binary manual judgment
- `human_notes`: optional free-form notes

### Evaluate a Human-Annotated Pack

```bash
python -m calibrate.evaluate_annotation_pack \
  --data_path outputs/calibration/annotation_pack_structured_labeled.jsonl \
  --out_jsonl outputs/calibration/scored_rows_structured.jsonl \
  --summary_csv outputs/calibration/summary_structured.csv \
  --summary_json outputs/calibration/summary_structured.json \
  --best_out outputs/calibration/best_params_structured.json
```

```bash
python -m calibrate.evaluate_annotation_pack \
  --data_path outputs/calibration/annotation_pack_flat_matcher_labeled.jsonl \
  --out_jsonl outputs/calibration/scored_rows_flat_matcher.jsonl \
  --summary_csv outputs/calibration/summary_flat_matcher.csv \
  --summary_json outputs/calibration/summary_flat_matcher.json \
  --best_out outputs/calibration/best_params_flat_matcher.json
```

Default calibration behavior:

- calibration is aggregated by `task_type`, not by checkpoint
- the default search objective is `f1`
- `model_tag`, `model_step`, and `model_path` are retained only for traceability

Outputs:

- `scored_rows.jsonl`: per-row model scores, predictions, and human labels
- `summary.csv/json`: per-task search results, including `accuracy`, `f1`, `precision`, `recall`, `cohen_kappa`, `coverage`, `num_rows`, and `num_labeled`
- `best_params.json`: one best parameter row for each task type

## Notes

- The default target training model and tokenizer are centrally managed in `src/project_config/settings.py`.
- Token usage accounting is available in `llm_textgen` and can be enabled by generation callers when comparing test-time cost.
- The main validation/test report workflow writes per-stage details plus `validation_summary.csv` and `test_summary.csv` into one report directory.
