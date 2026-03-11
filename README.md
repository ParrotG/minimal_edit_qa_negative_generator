# Grounded-QA SFT Pipeline

This repository provides an end-to-end pipeline for grounded question answering with supervised fine-tuning (SFT). It prepares grounded-QA training data from HotpotQA, generates structured supervision with a teacher model, trains an SFT model, and evaluates base models and checkpoints with protocol-aware groundedness metrics.

The project is designed for research on grounded factuality under explicit evidence constraints. The structured target format requires the model to predict answerability, extract evidence, produce a short rationale, return an answer, and self-report confidence. The surrounding tooling supports data construction, filtering, calibration, training, validation-time checkpoint selection, and final test-time reporting.

## Repository Layout

- `src/grounded_qa`: source construction, teacher generation/validation, and SFT-record packing
- `src/sft_trainer`: dataset preparation and SFT training
- `src/model_eval`: validation/test generation, grounded evaluation, DeepEval, and report orchestration
- `src/cablibrate`: optional calibration utilities for NLI and qa-metrics agreement analysis
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
10. Validation-time checkpoint selection and final evaluation report

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

## Optional Calibration

Calibration is optional. The repository already ships default judge settings, but you can build a manual annotation pack and compare model judgments against human labels when recalibration is needed.

### Build a Manual Annotation Pack

```bash
python -m cablibrate.build_annotation_pack \
  --data_path outputs/model_eval/sft_val_structured_details.jsonl \
  --out_jsonl outputs/calibration/annotation_pack.jsonl \
  --metrics_out outputs/calibration/annotation_pack_metrics.json
```

### Evaluate Human-Annotated Calibration Data

```bash
python -m cablibrate.evaluate_annotation_pack \
  --data_path outputs/calibration/annotation_pack_labeled.jsonl \
  --out_jsonl outputs/calibration/scored_rows.jsonl \
  --summary_csv outputs/calibration/summary.csv \
  --best_out outputs/calibration/best_params.json
```

The annotation pack supports three task types:

- `nli_flat`: `knowledge + question + answer`
- `nli_structured`: `evidence + question + rationale + answer`
- `matcher`: human-readable `knowledge + question + reference_answer + answer`, with matcher scoring based on `question + reference_answer + answer`

## Notes

- The default target training model and tokenizer are centrally managed in `src/project_config/settings.py`.
- Token usage accounting is available in `llm_textgen` and can be enabled by generation callers when comparing test-time cost.
- The main validation/test report workflow writes structured evaluation curves, confidence analysis, and DeepEval outputs into one report directory.
