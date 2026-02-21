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

### 3) Optional extras (install only when needed)

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
ssqpg source sample \
  --out data/processed/source.jsonl \
  --source halueval \
  --max-samples 5000 \
  --include-reference-answer true
```

### Stage 2: Self-sampled answer generation

```bash
ssqpg generate answers \
  --in-path data/processed/source.jsonl \
  --out data/processed/generated.jsonl \
  --backend local \
  --model-name Qwen/Qwen3-0.6B \
  --num-samples-per-question 6
```

### Stage 3: NLI-based answer judgment

```bash
ssqpg judge answers \
  --in-path data/processed/generated.jsonl \
  --out data/processed/judged.jsonl
```

### Stage 4: Pair building with hard-question augmentation

```bash
ssqpg pair build \
  --in-path data/processed/judged.jsonl \
  --out data/processed/pairs.jsonl \
  --out-augmented-judged data/processed/judged_augmented.jsonl \
  --extra-samples-per-hard 0 \
  --enable-api-repair true
```

### Stage 5: Final pair filtering (DPO-ready)

```bash
ssqpg filter apply \
  --in-path data/processed/pairs.jsonl \
  --out data/processed/dpo_pairs.jsonl
```

### Stage 6: Surface-signal audit

```bash
ssqpg audit surface \
  --in-path data/processed/dpo_pairs.jsonl \
  --out data/metrics/surface_audit.json
```

### Stage 7: Difficulty bucketing

```bash
ssqpg bucket difficulty \
  --in-path data/processed/dpo_pairs.jsonl \
  --out data/processed/dpo_pairs_bucketed.jsonl
```

## Evaluation Utilities (`model_eval`)

`model_eval` scripts are decoupled tools for:
- answer generation across base/LoRA checkpoints
- NLI faithfulness evaluation
- DeepEval hallucination scoring
- NLI/DeepEval consistency analysis
- threshold search and pairwise preference curves

Run them from source, for example:

```bash
python -m src.model_eval.generate_answers --help
python -m src.model_eval.eval_nli_faithfulness --help
python -m src.model_eval.eval_deepeval_hallucination --help
```

## Notes on Naming

- Package name remains `meqng` to avoid environment breakage.
- Current active implementation is `ssqpg`.
- Legacy `meqng` code is preserved but not the primary research route.
