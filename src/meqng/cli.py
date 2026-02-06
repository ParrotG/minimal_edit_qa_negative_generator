from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import track

from .config import HaluEvalConfig, NLIConfig
from .halu import build_entity_bank_from_halueval, iter_halueval_qa, sanity_check_nli
from .io import read_jsonl, write_jsonl
from .nli import NLIVerifier
from .sampling import sample_halueval
from .perturb import DEFAULT_PERTURBATORS
from .filter import DEFAULT_FILTERS
from .filter.nli_flip import NLIFlipFilter

app = typer.Typer(add_completion=False)
console = Console()


@app.command("version")
def version() -> None:
    console.print("meqng 0.1.0")


halu_app = typer.Typer()
perturb_app = typer.Typer()
filter_app = typer.Typer()
app.add_typer(halu_app, name="halu")
app.add_typer(perturb_app, name="perturb")
app.add_typer(filter_app, name="filter")


@halu_app.command("sanity-check")
def halu_sanity_check(
    out: str = typer.Option(..., help="Output JSON metrics path."),
    max_samples: int = typer.Option(2000, help="Maximum number of samples."),
    seed: int = typer.Option(0, help="Random seed."),
    nli_model: str = typer.Option(NLIConfig.model_name, help="HF model name for NLI."),
    device: str = typer.Option("cuda", help="Device: cuda or cpu."),
    batch_size: int = typer.Option(16, help="Batch size."),
) -> None:
    """Stage 1a: NLI agreement sanity check on original (right vs hallucinated) pairs."""
    cfg = HaluEvalConfig()
    samples = list(iter_halueval_qa(cfg, max_samples=max_samples, seed=seed))
    nli = NLIVerifier(model_name=nli_model, device=device, batch_size=batch_size)
    metrics = sanity_check_nli(samples, nli=nli, out_path=out)
    console.print_json(json.dumps(metrics))


@halu_app.command("build-entity-bank")
def halu_build_entity_bank(
    out: str = typer.Option(..., help="Output entity bank JSON path."),
    max_samples: int = typer.Option(20000, help="Maximum number of samples."),
    seed: int = typer.Option(0, help="Random seed."),
    spacy_model: str = typer.Option("en_core_web_trf", help="spaCy model name."),
    min_count: int = typer.Option(2, help="Min frequency for an entity surface form."),
) -> None:
    """Stage 1b: Build corpus-level entity bank for same-type replacement."""
    cfg = HaluEvalConfig()
    samples = list(iter_halueval_qa(cfg, max_samples=max_samples, seed=seed))
    bank = build_entity_bank_from_halueval(samples, out_path=out, spacy_model=spacy_model, min_count=min_count)
    console.print(f"Saved entity bank with {len(bank)} labels to {out}")


@halu_app.command("sample")
def halu_sample(
    out: str = typer.Option(..., help="Output sampled JSONL path."),
    max_samples: int = typer.Option(5000, help="Maximum number of sampled records."),
    max_prompt_tokens: int = typer.Option(768, help="Max prompt tokens."),
    max_total_tokens: int = typer.Option(1024, help="Max total tokens (prompt+chosen+rejected_orig)."),
    tokenizer_name: str = typer.Option("gpt2", help="Tokenizer for token counting."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Stage 2: Sample + filter by length constraints."""
    cfg = HaluEvalConfig()
    samples = list(iter_halueval_qa(cfg, max_samples=None, seed=seed))
    sample_halueval(
        samples=samples,
        out_path=out,
        max_samples=max_samples,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens,
        tokenizer_name=tokenizer_name,
        seed=seed,
    )
    console.print(f"Saved sampled records to {out}")


@perturb_app.command("generate")
def perturb_generate(
    in_path: str = typer.Option(..., help="Input sampled JSONL path."),
    out: str = typer.Option(..., help="Output candidates JSONL path."),
    entity_bank: Optional[str] = typer.Option(None, help="Entity bank JSON path."),
    max_candidates_per_sample: int = typer.Option(6, help="Max candidates per sample (across all perturbators)."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Stage 3: Generate candidate negatives by applying multiple perturbators."""
    bank_obj: Optional[Dict[str, Any]] = None
    if entity_bank:
        bank_obj = json.loads(Path(entity_bank).read_text(encoding="utf-8"))

    records = list(read_jsonl(in_path))
    out_recs: List[Dict[str, Any]] = []
    for rec in track(records, description="Generating candidates"):
        knowledge = rec["knowledge"]
        question = rec["question"]
        chosen = rec["chosen"]
        uid = rec["id"]

        seen = set()
        cands: List[Dict[str, Any]] = []
        per_seed = seed + int(uid)

        for p in DEFAULT_PERTURBATORS:
            if len(cands) >= max_candidates_per_sample:
                break
            gen = p.generate(
                knowledge=knowledge,
                question=question,
                answer=chosen,
                entity_bank=bank_obj,
                max_candidates=max(1, max_candidates_per_sample - len(cands)),
                seed=per_seed,
            )
            for g in gen:
                if g.text in seen:
                    continue
                seen.add(g.text)
                cands.append({"text": g.text, "perturbator": g.perturbator, "meta": g.meta})

        for c in cands:
            out_recs.append(
                {
                    "id": uid,
                    "knowledge": knowledge,
                    "question": question,
                    "chosen": chosen,
                    "candidate": c["text"],
                    "perturbator": c["perturbator"],
                    "perturb_meta": c["meta"],
                }
            )

    write_jsonl(out, out_recs)
    console.print(f"Saved {len(out_recs)} candidates to {out}")


@perturb_app.command("one")
def perturb_one(
    perturbator: str = typer.Option(..., help="Perturbator name: entity_swap | numeric_perturb | negation_toggle"),
    knowledge: str = typer.Option(...),
    question: str = typer.Option(...),
    answer: str = typer.Option(...),
    entity_bank: Optional[str] = typer.Option(None),
    max_candidates: int = typer.Option(5),
    seed: int = typer.Option(0),
) -> None:
    """Standalone: run a single perturbator on one sample and print candidates."""
    bank_obj: Optional[Dict[str, Any]] = None
    if entity_bank:
        bank_obj = json.loads(Path(entity_bank).read_text(encoding="utf-8"))

    pmap = {p.name: p for p in DEFAULT_PERTURBATORS}
    if perturbator not in pmap:
        raise typer.BadParameter(f"Unknown perturbator: {perturbator}. Available: {list(pmap.keys())}")

    gen = pmap[perturbator].generate(knowledge, question, answer, bank_obj, max_candidates, seed)
    console.print_json(json.dumps([{"text": g.text, "meta": g.meta} for g in gen], ensure_ascii=False, indent=2))


@filter_app.command("apply")
def filter_apply(
    in_path: str = typer.Option(..., help="Input candidates JSONL path."),
    out: str = typer.Option(..., help="Output DPO pairs JSONL path."),
    nli_model: str = typer.Option(NLIConfig.model_name, help="HF NLI model name."),
    device: str = typer.Option("cuda", help="Device: cuda or cpu."),
    batch_size: int = typer.Option(16, help="NLI batch size."),
    entail_threshold_pos: float = typer.Option(0.60, help="Chosen entailment threshold."),
    entail_threshold_neg: float = typer.Option(0.30, help="Candidate entailment threshold."),
    max_per_sample: int = typer.Option(1, help="Max kept candidates per sample."),
) -> None:
    """Stage 4: Filter candidates, then export (prompt, chosen, rejected) for DPO training."""
    verifier = NLIVerifier(model_name=nli_model, device=device, batch_size=batch_size)
    nli_filter = NLIFlipFilter(verifier=verifier, entail_threshold_pos=entail_threshold_pos, entail_threshold_neg=entail_threshold_neg)

    filters = list(DEFAULT_FILTERS) + [nli_filter]

    rows = list(read_jsonl(in_path))
    kept_by_id: Dict[str, List[Dict[str, Any]]] = {}

    for r in track(rows, description="Filtering candidates"):
        uid = r["id"]
        knowledge = r["knowledge"]
        question = r["question"]
        chosen = r["chosen"]
        cand = r["candidate"]

        decisions = []
        ok = True
        meta = {}
        for flt in filters:
            d = flt.check(knowledge, question, chosen, cand)
            decisions.append({"filter": flt.name, "keep": d.keep, "reason": d.reason, "meta": d.meta})
            if not d.keep:
                ok = False
                break
            meta[flt.name] = d.meta

        if not ok:
            continue

        prompt = f"{knowledge}\nQuestion: {question}"
        out_rec = {
            "id": uid,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": cand,
            "perturbator": r.get("perturbator"),
            "perturb_meta": r.get("perturb_meta"),
            "filter_meta": meta,
            "filter_trace": decisions,
        }
        kept_by_id.setdefault(uid, []).append(out_rec)

    # Keep top-N per sample (currently first-come; you can add ranking later).
    final = []
    for uid, lst in kept_by_id.items():
        final.extend(lst[:max_per_sample])

    write_jsonl(out, final)
    console.print(f"Saved {len(final)} DPO pairs to {out}")


@filter_app.command("one")
def filter_one(
    knowledge: str = typer.Option(...),
    question: str = typer.Option(...),
    chosen: str = typer.Option(...),
    candidate: str = typer.Option(...),
    nli_model: str = typer.Option(NLIConfig.model_name),
    device: str = typer.Option("cuda"),
    batch_size: int = typer.Option(16),
    entail_threshold_pos: float = typer.Option(0.60),
    entail_threshold_neg: float = typer.Option(0.30),
) -> None:
    """Standalone: run the default filter stack on one candidate and print decisions."""
    verifier = NLIVerifier(model_name=nli_model, device=device, batch_size=batch_size)
    nli_filter = NLIFlipFilter(verifier=verifier, entail_threshold_pos=entail_threshold_pos, entail_threshold_neg=entail_threshold_neg)
    filters = list(DEFAULT_FILTERS) + [nli_filter]

    trace = []
    for flt in filters:
        d = flt.check(knowledge, question, chosen, candidate)
        trace.append({"filter": flt.name, "keep": d.keep, "reason": d.reason, "meta": d.meta})
        if not d.keep:
            break
    console.print_json(json.dumps(trace, ensure_ascii=False, indent=2))
