from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import track

from .config import DifficultyConfig, FilterConfig, HaluEvalConfig, NLIConfig, SpanDropConfig, TextAttackConfig
from .difficulty import DifficultyScorer
from .halu import build_entity_bank_from_halueval, iter_halueval_qa, sanity_check_nli
from .io import read_jsonl, write_jsonl
from .nli import NLIVerifier
from .prompt import build_qa_answer_prefix, build_qa_premise
from .ranking import compute_rank_score
from .sampling import sample_halueval
from .perturb import build_perturbators
from .filter import DEFAULT_FILTERS
from .filter.grammar import GrammarFilter
from .filter.nli_flip import NLIFlipFilter
from .filter.qa_consistency import QAConsistencyFilter

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
    tokenizer_name: str = typer.Option("Qwen/Qwen3-0.6B", help="Tokenizer for token counting."),
    seed: int = typer.Option(0, help="Random seed."),
    min_chosen_entail: Optional[float] = typer.Option(
        None,
        help="Optional NLI entailment threshold for chosen answers in sampling stage.",
    ),
    nli_model: str = typer.Option(NLIConfig.model_name, help="HF model name for NLI filtering."),
    nli_device: str = typer.Option(NLIConfig.device, help="NLI device: cuda or cpu."),
    nli_batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size for filtering."),
    nli_max_length: int = typer.Option(NLIConfig.max_length, help="NLI max sequence length."),
    nli_fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 for NLI on CUDA."),
) -> None:
    """Stage 2: Sample + filter by length constraints."""
    cfg = HaluEvalConfig()
    samples = list(iter_halueval_qa(cfg, max_samples=None, seed=seed))
    nli_verifier: Optional[NLIVerifier] = None
    if min_chosen_entail is not None:
        nli_verifier = NLIVerifier(
            model_name=nli_model,
            device=nli_device,
            batch_size=nli_batch_size,
            max_length=nli_max_length,
            fp16=nli_fp16,
        )

    metrics = sample_halueval(
        samples=samples,
        out_path=out,
        max_samples=max_samples,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens,
        tokenizer_name=tokenizer_name,
        seed=seed,
        nli_verifier=nli_verifier,
        min_chosen_entail=min_chosen_entail,
    )
    console.print(f"Saved sampled records to {out}")
    console.print_json(json.dumps(metrics))


@perturb_app.command("generate")
def perturb_generate(
    in_path: str = typer.Option(..., help="Input sampled JSONL path."),
    out: str = typer.Option(..., help="Output candidates JSONL path."),
    entity_bank: Optional[str] = typer.Option(None, help="Entity bank JSON path."),
    max_candidates_per_sample: int = typer.Option(6, help="Max candidates per sample (across all perturbators)."),
    attempts_per_perturbator: int = typer.Option(
        1,
        min=1,
        help="Number of repeated perturbation attempts per perturbator for one sample.",
    ),
    enable_span_drop: bool = typer.Option(False, help="Enable span-level span_drop perturbator."),
    span_drop_min_words: int = typer.Option(SpanDropConfig.min_words, help="Minimum words in a dropped span."),
    span_drop_max_words: int = typer.Option(SpanDropConfig.max_words, help="Maximum words in a dropped span."),
    enable_textattack: bool = typer.Option(False, help="Enable TextAttack-based NLI flip perturbator."),
    textattack_augmenter: str = typer.Option(
        TextAttackConfig.augmenter, help="TextAttack augmenter: embedding or wordnet."
    ),
    textattack_pct_words_to_swap: float = typer.Option(
        TextAttackConfig.pct_words_to_swap, help="Fraction of words to perturb for TextAttack."
    ),
    textattack_transformations_per_example: int = typer.Option(
        TextAttackConfig.transformations_per_example, help="Number of augmented outputs per TextAttack search call."
    ),
    textattack_search_calls: int = typer.Option(
        TextAttackConfig.search_calls, help="How many TextAttack search calls to run per sample."
    ),
    textattack_entail_threshold_neg: float = typer.Option(
        TextAttackConfig.entail_threshold_neg, help="NLI entailment threshold for accepting TextAttack negatives."
    ),
    textattack_contradiction_ratio: float = typer.Option(
        TextAttackConfig.contradiction_ratio, help="Target contradiction ratio among accepted TextAttack negatives."
    ),
    textattack_min_norm_edit: float = typer.Option(
        TextAttackConfig.min_norm_edit, help="Minimum normalized edit distance for TextAttack."
    ),
    textattack_max_norm_edit: float = typer.Option(
        TextAttackConfig.max_norm_edit, help="Maximum normalized edit distance for TextAttack."
    ),
    textattack_semantic_model_name: Optional[str] = typer.Option(
        TextAttackConfig.semantic_model_name,
        help="Sentence-Transformer model for semantic similarity; pass empty string to disable.",
    ),
    textattack_min_semantic_similarity: float = typer.Option(
        TextAttackConfig.min_semantic_similarity, help="Minimum semantic similarity between original and TextAttack candidate."
    ),
    textattack_protect_entities: bool = typer.Option(
        TextAttackConfig.protect_entities, help="Protect entity spans during TextAttack editing."
    ),
    textattack_protect_numbers: bool = typer.Option(
        TextAttackConfig.protect_numbers, help="Protect number spans during TextAttack editing."
    ),
    textattack_spacy_model: str = typer.Option(
        TextAttackConfig.spacy_model, help="spaCy model for entity span protection."
    ),
    textattack_nli_model: str = typer.Option(NLIConfig.model_name, help="NLI model used by TextAttack perturbator."),
    textattack_nli_device: str = typer.Option(NLIConfig.device, help="NLI device used by TextAttack perturbator."),
    textattack_nli_batch_size: int = typer.Option(
        NLIConfig.batch_size, help="NLI batch size used by TextAttack perturbator."
    ),
    textattack_nli_max_length: int = typer.Option(
        NLIConfig.max_length, help="NLI max sequence length used by TextAttack perturbator."
    ),
    textattack_nli_fp16: bool = typer.Option(
        NLIConfig.fp16, help="Whether to enable fp16 for TextAttack perturbator NLI on CUDA."
    ),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Stage 3: Generate candidate negatives by applying multiple perturbators."""
    bank_obj: Optional[Dict[str, Any]] = None
    if entity_bank:
        bank_obj = json.loads(Path(entity_bank).read_text(encoding="utf-8"))

    semantic_model_name = textattack_semantic_model_name or None
    textattack_verifier: Optional[NLIVerifier] = None
    if enable_textattack:
        textattack_verifier = NLIVerifier(
            model_name=textattack_nli_model,
            device=textattack_nli_device,
            batch_size=textattack_nli_batch_size,
            max_length=textattack_nli_max_length,
            fp16=textattack_nli_fp16,
        )

    perturbators = build_perturbators(
        enable_span_drop=enable_span_drop,
        span_drop_min_words=span_drop_min_words,
        span_drop_max_words=span_drop_max_words,
        enable_textattack=enable_textattack,
        textattack_verifier=textattack_verifier,
        textattack_augmenter=textattack_augmenter,
        textattack_pct_words_to_swap=textattack_pct_words_to_swap,
        textattack_transformations_per_example=textattack_transformations_per_example,
        textattack_search_calls=textattack_search_calls,
        textattack_entail_threshold_neg=textattack_entail_threshold_neg,
        textattack_contradiction_ratio=textattack_contradiction_ratio,
        textattack_min_norm_edit=textattack_min_norm_edit,
        textattack_max_norm_edit=textattack_max_norm_edit,
        textattack_semantic_model_name=semantic_model_name,
        textattack_min_semantic_similarity=textattack_min_semantic_similarity,
        textattack_protect_entities=textattack_protect_entities,
        textattack_protect_numbers=textattack_protect_numbers,
        textattack_spacy_model=textattack_spacy_model,
    )

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

        for attempt_idx in range(attempts_per_perturbator):
            for p_idx, p in enumerate(perturbators):
                if len(cands) >= max_candidates_per_sample:
                    break
                attempt_seed = per_seed + p_idx * 1009 + attempt_idx * 7919
                gen = p.generate(
                    knowledge=knowledge,
                    question=question,
                    answer=chosen,
                    entity_bank=bank_obj,
                    max_candidates=1,
                    seed=attempt_seed,
                )
                for g in gen:
                    if g.text in seen:
                        continue
                    seen.add(g.text)
                    cands.append(
                        {
                            "text": g.text,
                            "perturbator": g.perturbator,
                            "meta": {**g.meta, "attempt_index": attempt_idx},
                        }
                    )
                if len(cands) >= max_candidates_per_sample:
                    break
            if len(cands) >= max_candidates_per_sample:
                break

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
    perturbator: str = typer.Option(
        ...,
        help="Perturbator name: entity_swap | numeric_perturb | negation_toggle | span_drop | textattack_nli_flip",
    ),
    knowledge: str = typer.Option(...),
    question: str = typer.Option(...),
    answer: str = typer.Option(...),
    entity_bank: Optional[str] = typer.Option(None),
    max_candidates: int = typer.Option(5),
    attempts: int = typer.Option(1, min=1, help="Repeated attempts for this perturbator."),
    enable_textattack: bool = typer.Option(False, help="Enable TextAttack perturbator config."),
    textattack_augmenter: str = typer.Option(TextAttackConfig.augmenter),
    textattack_pct_words_to_swap: float = typer.Option(TextAttackConfig.pct_words_to_swap),
    textattack_transformations_per_example: int = typer.Option(TextAttackConfig.transformations_per_example),
    textattack_search_calls: int = typer.Option(TextAttackConfig.search_calls),
    textattack_entail_threshold_neg: float = typer.Option(TextAttackConfig.entail_threshold_neg),
    textattack_contradiction_ratio: float = typer.Option(TextAttackConfig.contradiction_ratio),
    textattack_min_norm_edit: float = typer.Option(TextAttackConfig.min_norm_edit),
    textattack_max_norm_edit: float = typer.Option(TextAttackConfig.max_norm_edit),
    textattack_semantic_model_name: Optional[str] = typer.Option(TextAttackConfig.semantic_model_name),
    textattack_min_semantic_similarity: float = typer.Option(TextAttackConfig.min_semantic_similarity),
    textattack_protect_entities: bool = typer.Option(TextAttackConfig.protect_entities),
    textattack_protect_numbers: bool = typer.Option(TextAttackConfig.protect_numbers),
    textattack_spacy_model: str = typer.Option(TextAttackConfig.spacy_model),
    textattack_nli_model: str = typer.Option(NLIConfig.model_name),
    textattack_nli_device: str = typer.Option(NLIConfig.device),
    textattack_nli_batch_size: int = typer.Option(NLIConfig.batch_size),
    textattack_nli_max_length: int = typer.Option(NLIConfig.max_length),
    textattack_nli_fp16: bool = typer.Option(NLIConfig.fp16),
    seed: int = typer.Option(0),
) -> None:
    """Standalone: run a single perturbator on one sample and print candidates."""
    bank_obj: Optional[Dict[str, Any]] = None
    if entity_bank:
        bank_obj = json.loads(Path(entity_bank).read_text(encoding="utf-8"))

    semantic_model_name = textattack_semantic_model_name or None
    textattack_verifier: Optional[NLIVerifier] = None
    if enable_textattack or perturbator == "textattack_nli_flip":
        textattack_verifier = NLIVerifier(
            model_name=textattack_nli_model,
            device=textattack_nli_device,
            batch_size=textattack_nli_batch_size,
            max_length=textattack_nli_max_length,
            fp16=textattack_nli_fp16,
        )

    perturbators = build_perturbators(
        enable_span_drop=True,
        enable_textattack=(enable_textattack or perturbator == "textattack_nli_flip"),
        textattack_verifier=textattack_verifier,
        textattack_augmenter=textattack_augmenter,
        textattack_pct_words_to_swap=textattack_pct_words_to_swap,
        textattack_transformations_per_example=textattack_transformations_per_example,
        textattack_search_calls=textattack_search_calls,
        textattack_entail_threshold_neg=textattack_entail_threshold_neg,
        textattack_contradiction_ratio=textattack_contradiction_ratio,
        textattack_min_norm_edit=textattack_min_norm_edit,
        textattack_max_norm_edit=textattack_max_norm_edit,
        textattack_semantic_model_name=semantic_model_name,
        textattack_min_semantic_similarity=textattack_min_semantic_similarity,
        textattack_protect_entities=textattack_protect_entities,
        textattack_protect_numbers=textattack_protect_numbers,
        textattack_spacy_model=textattack_spacy_model,
    )
    pmap = {p.name: p for p in perturbators}
    if perturbator not in pmap:
        raise typer.BadParameter(f"Unknown perturbator: {perturbator}. Available: {list(pmap.keys())}")

    out: List[Dict[str, Any]] = []
    for attempt_idx in range(attempts):
        attempt_seed = seed + attempt_idx * 7919
        gen = pmap[perturbator].generate(knowledge, question, answer, bank_obj, max_candidates, attempt_seed)
        for g in gen:
            out.append({"text": g.text, "meta": {**g.meta, "attempt_index": attempt_idx}})
    console.print_json(json.dumps(out, ensure_ascii=False, indent=2))


@filter_app.command("apply")
def filter_apply(
    in_path: str = typer.Option(..., help="Input candidates JSONL path."),
    out: str = typer.Option(..., help="Output DPO pairs JSONL path."),
    nli_model: str = typer.Option(NLIConfig.model_name, help="HF NLI model name."),
    device: str = typer.Option(NLIConfig.device, help="Device: cuda or cpu."),
    batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size."),
    entail_threshold_pos: float = typer.Option(NLIConfig.entail_threshold_pos, help="Chosen entailment threshold."),
    entail_threshold_neg: float = typer.Option(NLIConfig.entail_threshold_neg, help="Candidate entailment threshold."),
    enable_qa_consistency_filter: bool = typer.Option(True, help="Enable QA semantic consistency filter."),
    qa_similarity_model_name: str = typer.Option(
        FilterConfig.qa_similarity_model_name, help="Sentence-Transformer model for QA consistency filter."
    ),
    qa_similarity_min: float = typer.Option(
        FilterConfig.qa_similarity_min, help="Minimum QA semantic similarity for QA consistency filter."
    ),
    enable_grammar_filter: bool = typer.Option(False, help="Enable grammar filter (optional dependency)."),
    grammar_language: str = typer.Option("en-US", help="LanguageTool language code."),
    grammar_max_extra_issues: int = typer.Option(
        FilterConfig.grammar_max_extra_issues, help="Max additional grammar issues allowed vs chosen answer."
    ),
    ranking_strategy: str = typer.Option(
        "heuristic", help="Candidate ranking strategy per sample: heuristic or first_come."
    ),
    enable_difficulty_bucket: bool = typer.Option(False, help="Enable policy-logprob difficulty scoring and bucketing."),
    difficulty_model_name: str = typer.Option(
        DifficultyConfig.model_name, help="Causal LM model used for difficulty scoring."
    ),
    difficulty_device: str = typer.Option(DifficultyConfig.device, help="Device for difficulty scoring."),
    difficulty_batch_size: int = typer.Option(DifficultyConfig.batch_size, help="Batch size for difficulty scoring."),
    difficulty_max_length: int = typer.Option(DifficultyConfig.max_length, help="Max sequence length for difficulty scoring."),
    difficulty_fp16: bool = typer.Option(DifficultyConfig.fp16, help="Whether to enable fp16 for difficulty scoring on CUDA."),
    difficulty_hard_max_delta: float = typer.Option(
        DifficultyConfig.hard_max_delta, help="Hard bucket upper bound for delta=logP(chosen)-logP(rejected)."
    ),
    difficulty_medium_max_delta: float = typer.Option(
        DifficultyConfig.medium_max_delta, help="Medium bucket upper bound for delta=logP(chosen)-logP(rejected)."
    ),
    max_per_sample: int = typer.Option(1, help="Max kept candidates per sample."),
) -> None:
    """Stage 4: Filter candidates, rank them, and optionally assign difficulty buckets."""
    ranking_strategy = ranking_strategy.strip().lower()
    if ranking_strategy not in {"heuristic", "first_come"}:
        raise typer.BadParameter("ranking_strategy must be one of: heuristic, first_come")

    verifier = NLIVerifier(model_name=nli_model, device=device, batch_size=batch_size)
    nli_filter = NLIFlipFilter(verifier=verifier, entail_threshold_pos=entail_threshold_pos, entail_threshold_neg=entail_threshold_neg)

    filters = list(DEFAULT_FILTERS)
    if enable_qa_consistency_filter:
        filters.append(QAConsistencyFilter(model_name=qa_similarity_model_name, min_similarity=qa_similarity_min))
    if enable_grammar_filter:
        filters.append(GrammarFilter(language=grammar_language, max_extra_issues=grammar_max_extra_issues))
    filters.append(nli_filter)

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

        rank_score = 0.0
        rank_components: Dict[str, float] = {}
        if ranking_strategy == "heuristic":
            rank_score, rank_components = compute_rank_score(meta)

        prompt = build_qa_premise(knowledge, question)
        out_rec = {
            "id": uid,
            "knowledge": knowledge,
            "question": question,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": cand,
            "perturbator": r.get("perturbator"),
            "perturb_meta": r.get("perturb_meta"),
            "filter_meta": meta,
            "filter_trace": decisions,
            "rank_score": rank_score,
            "rank_components": rank_components,
        }
        kept_by_id.setdefault(uid, []).append(out_rec)

    # Keep top-N per sample.
    final = []
    for uid, lst in kept_by_id.items():
        if ranking_strategy == "heuristic":
            lst = sorted(lst, key=lambda x: x.get("rank_score", 0.0), reverse=True)
        final.extend(lst[:max_per_sample])

    if enable_difficulty_bucket and final:
        scorer = DifficultyScorer(
            model_name=difficulty_model_name,
            device=difficulty_device,
            batch_size=difficulty_batch_size,
            max_length=difficulty_max_length,
            fp16=difficulty_fp16,
        )
        prompt_prefixes = [build_qa_answer_prefix(r["knowledge"], r["question"]) for r in final]
        chosens = [r["chosen"] for r in final]
        rejecteds = [r["rejected"] for r in final]
        diff_records = scorer.score(
            prompt_prefixes=prompt_prefixes,
            chosens=chosens,
            rejecteds=rejecteds,
            hard_max_delta=difficulty_hard_max_delta,
            medium_max_delta=difficulty_medium_max_delta,
        )

        bucket_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for rec, d in zip(final, diff_records):
            rec["difficulty"] = {
                "chosen_avg_logprob": d.chosen_avg_logprob,
                "rejected_avg_logprob": d.rejected_avg_logprob,
                "delta": d.delta,
            }
            rec["difficulty_bucket"] = d.bucket
            bucket_counts[d.bucket] = bucket_counts.get(d.bucket, 0) + 1
        console.print_json(json.dumps({"difficulty_bucket_counts": bucket_counts}))

    write_jsonl(out, final)
    console.print(f"Saved {len(final)} DPO pairs to {out}")


@filter_app.command("bucket")
def filter_bucket(
    in_path: str = typer.Option(..., help="Input DPO pairs JSONL path."),
    out: str = typer.Option(..., help="Output DPO pairs JSONL path with difficulty buckets."),
    difficulty_model_name: str = typer.Option(
        DifficultyConfig.model_name, help="Causal LM model used for difficulty scoring."
    ),
    difficulty_device: str = typer.Option(DifficultyConfig.device, help="Device for difficulty scoring."),
    difficulty_batch_size: int = typer.Option(DifficultyConfig.batch_size, help="Batch size for difficulty scoring."),
    difficulty_max_length: int = typer.Option(DifficultyConfig.max_length, help="Max sequence length for difficulty scoring."),
    difficulty_fp16: bool = typer.Option(DifficultyConfig.fp16, help="Whether to enable fp16 for difficulty scoring on CUDA."),
    difficulty_hard_max_delta: float = typer.Option(
        DifficultyConfig.hard_max_delta, help="Hard bucket upper bound for delta=logP(chosen)-logP(rejected)."
    ),
    difficulty_medium_max_delta: float = typer.Option(
        DifficultyConfig.medium_max_delta, help="Medium bucket upper bound for delta=logP(chosen)-logP(rejected)."
    ),
) -> None:
    """Stage 5: Assign difficulty buckets to existing DPO pairs."""
    rows = list(read_jsonl(in_path))
    if not rows:
        write_jsonl(out, [])
        console.print(f"Saved 0 DPO pairs to {out}")
        return

    scorer = DifficultyScorer(
        model_name=difficulty_model_name,
        device=difficulty_device,
        batch_size=difficulty_batch_size,
        max_length=difficulty_max_length,
        fp16=difficulty_fp16,
    )

    prompt_prefixes = []
    chosens = []
    rejecteds = []
    for r in rows:
        if "knowledge" in r and "question" in r:
            prompt_prefixes.append(build_qa_answer_prefix(r["knowledge"], r["question"]))
        else:
            prompt_prefixes.append(f"{r['prompt']}\nAnswer: ")
        chosens.append(r["chosen"])
        rejecteds.append(r["rejected"])

    diff_records = scorer.score(
        prompt_prefixes=prompt_prefixes,
        chosens=chosens,
        rejecteds=rejecteds,
        hard_max_delta=difficulty_hard_max_delta,
        medium_max_delta=difficulty_medium_max_delta,
    )

    bucket_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for rec, d in zip(rows, diff_records):
        rec["difficulty"] = {
            "chosen_avg_logprob": d.chosen_avg_logprob,
            "rejected_avg_logprob": d.rejected_avg_logprob,
            "delta": d.delta,
        }
        rec["difficulty_bucket"] = d.bucket
        bucket_counts[d.bucket] = bucket_counts.get(d.bucket, 0) + 1

    write_jsonl(out, rows)
    console.print_json(json.dumps({"difficulty_bucket_counts": bucket_counts}))
    console.print(f"Saved {len(rows)} DPO pairs with difficulty buckets to {out}")


@filter_app.command("one")
def filter_one(
    knowledge: str = typer.Option(...),
    question: str = typer.Option(...),
    chosen: str = typer.Option(...),
    candidate: str = typer.Option(...),
    nli_model: str = typer.Option(NLIConfig.model_name),
    device: str = typer.Option(NLIConfig.device),
    batch_size: int = typer.Option(NLIConfig.batch_size),
    entail_threshold_pos: float = typer.Option(NLIConfig.entail_threshold_pos),
    entail_threshold_neg: float = typer.Option(NLIConfig.entail_threshold_neg),
    enable_qa_consistency_filter: bool = typer.Option(True),
    qa_similarity_model_name: str = typer.Option(FilterConfig.qa_similarity_model_name),
    qa_similarity_min: float = typer.Option(FilterConfig.qa_similarity_min),
    enable_grammar_filter: bool = typer.Option(False),
    grammar_language: str = typer.Option("en-US"),
    grammar_max_extra_issues: int = typer.Option(FilterConfig.grammar_max_extra_issues),
) -> None:
    """Standalone: run the default filter stack on one candidate and print decisions."""
    verifier = NLIVerifier(model_name=nli_model, device=device, batch_size=batch_size)
    nli_filter = NLIFlipFilter(verifier=verifier, entail_threshold_pos=entail_threshold_pos, entail_threshold_neg=entail_threshold_neg)
    filters = list(DEFAULT_FILTERS)
    if enable_qa_consistency_filter:
        filters.append(QAConsistencyFilter(model_name=qa_similarity_model_name, min_similarity=qa_similarity_min))
    if enable_grammar_filter:
        filters.append(GrammarFilter(language=grammar_language, max_extra_issues=grammar_max_extra_issues))
    filters.append(nli_filter)

    trace = []
    for flt in filters:
        d = flt.check(knowledge, question, chosen, candidate)
        trace.append({"filter": flt.name, "keep": d.keep, "reason": d.reason, "meta": d.meta})
        if not d.keep:
            break
    console.print_json(json.dumps(trace, ensure_ascii=False, indent=2))
