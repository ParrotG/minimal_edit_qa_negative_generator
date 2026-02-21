from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer, util

from .config import PairFilterConfig
from .ner import NERTagger, extract_numbers
from nli_judge.nli import NLIVerifier
from prompt import build_qa_premise, build_qa_question_answer_text
from .text import length_ratio, normalized_edit_distance


_YEAR_RE = re.compile(r"\b\d{4}\b")


def _answer_type_ok(question: str, answer: str, ner: NERTagger) -> bool:
    q = (question or "").strip().lower()
    ents = ner.extract(answer)
    labels = {e.label for e in ents}
    has_num = len(extract_numbers(answer)) > 0
    has_year = bool(_YEAR_RE.search(answer or ""))

    if q.startswith("who"):
        return "PERSON" in labels or "ORG" in labels
    if q.startswith("where"):
        return "GPE" in labels or "LOC" in labels
    if q.startswith("when"):
        return "DATE" in labels or "TIME" in labels or has_year
    if q.startswith("how many") or q.startswith("how much"):
        return has_num
    return True


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _compute_rank_score(meta: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Compute heuristic rank score for the kept pair."""

    comp: Dict[str, float] = {}
    w: Dict[str, float] = {}

    cand_entail = meta.get("nli", {}).get("rejected_entail")
    if cand_entail is not None:
        comp["nli_hardness"] = _clip01(float(cand_entail))
        w["nli_hardness"] = 0.45

    norm_edit = meta.get("edit_distance", {}).get("norm")
    if norm_edit is not None:
        comp["edit_proximity"] = _clip01(1.0 - float(norm_edit))
        w["edit_proximity"] = 0.30

    ratio = meta.get("length_ratio", {}).get("ratio")
    if ratio is not None:
        comp["length_proximity"] = _clip01(1.0 - abs(1.0 - float(ratio)))
        w["length_proximity"] = 0.15

    qa_sim = meta.get("qa_consistency", {}).get("similarity")
    if qa_sim is not None:
        comp["qa_similarity"] = _clip01(float(qa_sim))
        w["qa_similarity"] = 0.10

    if not comp:
        return 0.0, {}

    total = sum(w[k] for k in comp.keys())
    if total <= 0:
        return 0.0, comp

    score = sum(comp[k] * w[k] for k in comp.keys()) / total
    return float(score), comp


def apply_pair_filters(
    rows: Sequence[Dict[str, Any]],
    cfg: PairFilterConfig,
    verifier: NLIVerifier,
    entail_threshold_pos: float,
    entail_threshold_neg: float,
    include_prompt: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply quality filters and export final DPO-ready pairs."""

    qa_model: Optional[SentenceTransformer] = None
    if cfg.qa_similarity_model_name:
        qa_model = SentenceTransformer(cfg.qa_similarity_model_name)

    ner: Optional[NERTagger] = None
    if cfg.enforce_answer_type:
        ner = NERTagger(cfg.spacy_model)

    kept: List[Dict[str, Any]] = []
    counts = {
        "num_input": 0,
        "num_kept": 0,
        "drop_not_ready": 0,
        "drop_empty": 0,
        "drop_length": 0,
        "drop_edit": 0,
        "drop_answer_type": 0,
        "drop_nli": 0,
        "drop_qa": 0,
    }

    for row in rows:
        counts["num_input"] += 1

        if row.get("status") != "ready":
            counts["drop_not_ready"] += 1
            continue

        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not chosen or not rejected:
            counts["drop_empty"] += 1
            continue

        trace: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        ratio = length_ratio(chosen, rejected)
        keep_len = cfg.min_length_ratio <= ratio <= cfg.max_length_ratio
        trace.append({"filter": "length_ratio", "keep": keep_len, "reason": "ok" if keep_len else "ratio_out_of_range"})
        if not keep_len:
            counts["drop_length"] += 1
            continue
        meta["length_ratio"] = {"ratio": ratio}

        norm = normalized_edit_distance(chosen, rejected)
        keep_edit = cfg.min_norm_edit <= norm <= cfg.max_norm_edit
        trace.append({"filter": "edit_distance", "keep": keep_edit, "reason": "ok" if keep_edit else "edit_out_of_range"})
        if not keep_edit:
            counts["drop_edit"] += 1
            continue
        meta["edit_distance"] = {"norm": norm}

        if ner is not None:
            ok_chosen = _answer_type_ok(row["question"], chosen, ner)
            ok_rejected = _answer_type_ok(row["question"], rejected, ner)
            keep_type = bool(ok_chosen and ok_rejected)
            trace.append({"filter": "answer_type", "keep": keep_type, "reason": "ok" if keep_type else "answer_type_mismatch"})
            if not keep_type:
                counts["drop_answer_type"] += 1
                continue
            meta["answer_type"] = {"chosen_ok": ok_chosen, "rejected_ok": ok_rejected}

        premise = build_qa_premise(row["knowledge"], row["question"])
        pos = verifier.score([premise], [chosen])[0]
        neg = verifier.score([premise], [rejected])[0]
        keep_nli = pos.entail >= entail_threshold_pos and neg.entail <= entail_threshold_neg
        trace.append(
            {
                "filter": "nli",
                "keep": keep_nli,
                "reason": "ok" if keep_nli else "nli_not_flipped",
            }
        )
        if not keep_nli:
            counts["drop_nli"] += 1
            continue
        meta["nli"] = {
            "chosen_entail": pos.entail,
            "rejected_entail": neg.entail,
            "chosen_neutral": pos.neutral,
            "rejected_neutral": neg.neutral,
            "chosen_contradict": pos.contradict,
            "rejected_contradict": neg.contradict,
        }

        if qa_model is not None:
            ref = build_qa_question_answer_text(question=row["question"], answer=chosen)
            cand = build_qa_question_answer_text(question=row["question"], answer=rejected)
            embs = qa_model.encode([ref, cand], convert_to_tensor=True, normalize_embeddings=True)
            sim = float(util.cos_sim(embs[0], embs[1]).item())
            keep_qa = sim >= cfg.qa_similarity_min
            trace.append(
                {
                    "filter": "qa_consistency",
                    "keep": keep_qa,
                    "reason": "ok" if keep_qa else "qa_semantic_shifted",
                }
            )
            if not keep_qa:
                counts["drop_qa"] += 1
                continue
            meta["qa_consistency"] = {"similarity": sim, "min_similarity": cfg.qa_similarity_min}

        rank_score, rank_comp = _compute_rank_score(meta)

        out = {
            "id": row["id"],
            "task": "qa",
            "knowledge": row["knowledge"],
            "question": row["question"],
            "chosen": chosen,
            "rejected": rejected,
            "difficulty_label": row.get("difficulty_label"),
            "trial_count": row.get("trial_count"),
            "correct_count": row.get("correct_count"),
            "accuracy": row.get("accuracy"),
            "chosen_origin": row.get("chosen_origin"),
            "pair_meta": row.get("pair_meta", {}),
            "filter_meta": meta,
            "filter_trace": trace,
            "rank_score": rank_score,
            "rank_components": rank_comp,
        }
        if include_prompt:
            out["prompt"] = build_qa_premise(row["knowledge"], row["question"])
        kept.append(out)

    counts["num_kept"] = len(kept)
    return kept, counts
