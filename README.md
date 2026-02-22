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
```

## Primary Pipeline (`ssqpg`)

The project is stage-driven. Run each stage explicitly and keep intermediate JSONL/JSON metrics.

### Stage 1: Source sampling

```bash
python -m src.ssqpg.cli source sample \
  --out data/processed/source.jsonl \
  --source halueval \
  --max-samples 5000 \
  --include-reference-answer true
```

### Stage 2: Self-sampled answer generation

```bash
python -m src.ssqpg.cli generate answers \
  --in-path data/processed/source.jsonl \
  --out data/processed/generated.jsonl \
  --backend local \
  --model-name Qwen/Qwen3-0.6B \
  --num-samples-per-question 6
```

### Stage 3: Calibrated answer judgment (Method B + Method D)

```bash
python -m src.ssqpg.cli judge answers \
  --in-path data/processed/generated.jsonl \
  --out data/processed/judged.jsonl
```

### Stage 4: Answer filter (drop abstain by default)

```bash
python -m src.ssqpg.cli pair answer-filter \
  --in-path data/processed/judged.jsonl \
  --out data/processed/judged_filtered.jsonl
```

### Stage 5: Group summary (inspection step)

```bash
python -m src.ssqpg.cli pair group \
  --in-path data/processed/judged_filtered.jsonl \
  --out data/processed/grouped.jsonl
```

### Stage 6: Hard augmentation (rejudge + re-filter)

```bash
python -m src.ssqpg.cli pair hard-augment \
  --in-path data/processed/judged_filtered.jsonl \
  --out data/processed/judged_augmented.jsonl \
  --extra-samples-per-hard 0 \
  --enable-api-repair true
```

### Stage 7: Pairing

```bash
python -m src.ssqpg.cli pair pairing \
  --in-path data/processed/judged_augmented.jsonl \
  --out data/processed/pairs.jsonl
```

### Stage 8: Final pair filtering (no NLI re-run)

```bash
python -m src.ssqpg.cli filter apply \
  --in-path data/processed/pairs.jsonl \
  --out data/processed/dpo_pairs.jsonl
```

### Stage 9: Surface-signal audit

```bash
python -m src.ssqpg.cli audit surface \
  --in-path data/processed/dpo_pairs.jsonl \
  --out data/metrics/surface_audit.json
```

### Stage 10: Difficulty bucketing

```bash
python -m src.ssqpg.cli bucket difficulty \
  --in-path data/processed/dpo_pairs.jsonl \
  --out data/processed/dpo_pairs_bucketed.jsonl
```

## Evaluation Utilities (`model_eval`)

Run them from source, for example:

```bash
python -m src.model_eval.generate_answers --help
python -m src.model_eval.eval_nli_faithfulness --help
python -m src.model_eval.eval_deepeval_hallucination --help
```

## Notes on Naming

- Active judge package: `qa_judge`.
- Active data pipeline: `ssqpg`.
- Legacy `meqng` code is preserved but not the primary research route.
