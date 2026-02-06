# minimal_edit_qa_negative_generator (meqng)

This repository contains a **data construction** pipeline for building **minimal-edit negative answers** for evidence-grounded QA,
intended for downstream preference training (e.g., DPO) in a separate training repo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m spacy download en_core_web_trf
```

Optional grammar filtering:

```bash
pip install -e ".[grammar]"
```

## Stage 1: Sanity check (HaluEval QA) + entity bank

Sanity check measures whether an NLI verifier agrees with the dataset's `right_answer` vs `hallucinated_answer`.

```bash
meqng halu sanity-check --out data/metrics/nli_sanity.json --max-samples 2000
meqng halu build-entity-bank --out data/artifacts/entity_bank.json --max-samples 20000
```

## Stage 2: Sampling

```bash
meqng halu sample   --out data/processed/sampled.jsonl   --max-samples 5000   --max-prompt-tokens 768   --max-total-tokens 1024
```

## Stage 3: Perturbation (candidate generation)

```bash
meqng perturb generate   --in data/processed/sampled.jsonl   --entity-bank data/artifacts/entity_bank.json   --out data/processed/candidates.jsonl   --max-candidates-per-sample 6
```

## Stage 4: Filtering + final DPO pairs export

```bash
meqng filter apply   --in data/processed/candidates.jsonl   --out data/processed/dpo_pairs.jsonl   --max-per-sample 1
```

## Standalone component testing

Run a perturbator on a single example:

```bash
meqng perturb one --perturbator entity_swap --knowledge "..." --question "..." --answer "..." --entity-bank data/artifacts/entity_bank.json
```

Run filters on a single candidate:

```bash
meqng filter one --knowledge "..." --question "..." --chosen "..." --candidate "..."
```

Notes:
- The pipeline is intentionally **stage-driven**. No command automatically runs all stages end-to-end.
- All intermediate artifacts are JSON/JSONL for easy debugging and recomposition.
