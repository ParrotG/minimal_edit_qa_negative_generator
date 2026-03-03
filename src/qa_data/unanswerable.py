from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .construct import ConstructionConfig, build_answerable_example
from .records import ContextDocument, QaExample, SupportingSentence


@dataclass(frozen=True)
class UnanswerableBuildConfig:
    """Configuration for v1 unanswerable raw-example construction."""

    paired_fraction: float = 0.5
    max_total_examples: int = -1
    replace_supporting_facts_min: int = 1
    replace_supporting_facts_max: int = 1
    same_doc_candidate_radius: int = 1
    allow_same_doc_non_adjacent: bool = True
    adjacent_doc_sentence_limit: int = 1
    include_title_prefix: bool = True
    seed: int = 42


def _format_sentence_block(title: str, sentence: str, include_title_prefix: bool) -> str:
    text = str(sentence or "").strip()
    if not include_title_prefix or not title:
        return text
    return f"{title}: {text}".strip()


def _stable_rng(seed: int, *parts: object) -> random.Random:
    joined = "::".join(str(part) for part in parts)
    return random.Random(f"{seed}:{joined}")


def _support_index(example: QaExample) -> set[tuple[str, int]]:
    return {(item.title, item.sent_id) for item in example.supporting_sentences}


def _context_map(example: QaExample) -> dict[str, ContextDocument]:
    return {doc.title: doc for doc in example.context_documents if doc.title}


def _context_order(example: QaExample) -> dict[str, int]:
    return {doc.title: idx for idx, doc in enumerate(example.context_documents)}


def _same_doc_candidates(
    *,
    support: SupportingSentence,
    example: QaExample,
    cfg: UnanswerableBuildConfig,
) -> list[tuple[str, int, str]]:
    docs = _context_map(example)
    doc = docs.get(support.title)
    if doc is None:
        return []

    support_idx = _support_index(example)
    candidates: list[tuple[str, int, str]] = []

    for sent_id, sentence in enumerate(doc.sentences):
        if (support.title, sent_id) in support_idx:
            continue
        if abs(sent_id - support.sent_id) <= cfg.same_doc_candidate_radius:
            candidates.append((support.title, sent_id, sentence))

    if candidates or not cfg.allow_same_doc_non_adjacent:
        return candidates

    for sent_id, sentence in enumerate(doc.sentences):
        if (support.title, sent_id) in support_idx:
            continue
        candidates.append((support.title, sent_id, sentence))
    return candidates


def _adjacent_doc_candidates(
    *,
    support: SupportingSentence,
    example: QaExample,
    cfg: UnanswerableBuildConfig,
) -> list[tuple[str, int, str]]:
    if cfg.adjacent_doc_sentence_limit <= 0:
        return []

    order = _context_order(example)
    center = order.get(support.title)
    if center is None:
        return []

    candidates: list[tuple[str, int, str]] = []
    for delta in (-1, 1):
        idx = center + delta
        if idx < 0 or idx >= len(example.context_documents):
            continue
        doc = example.context_documents[idx]
        for sent_id, sentence in enumerate(doc.sentences[: cfg.adjacent_doc_sentence_limit]):
            candidates.append((doc.title, sent_id, sentence))
    return candidates


def _choose_replacement(
    *,
    support: SupportingSentence,
    example: QaExample,
    cfg: UnanswerableBuildConfig,
    rng: random.Random,
) -> tuple[str, int, str] | None:
    same_doc = _same_doc_candidates(support=support, example=example, cfg=cfg)
    if same_doc:
        return rng.choice(same_doc)

    adjacent = _adjacent_doc_candidates(support=support, example=example, cfg=cfg)
    if adjacent:
        return rng.choice(adjacent)
    return None


def build_unanswerable_from_answerable_example(
    example: QaExample,
    cfg: UnanswerableBuildConfig,
    *,
    origin_track: str = "paired_answerable",
    paired_answerable_id: str | None = None,
    paired_answerable_source_id: str | None = None,
    raw_source_split: str | None = None,
) -> Optional[QaExample]:
    """Derive one v1 unanswerable example from an answerable scaffold."""

    if example.answerability_label != "answerable":
        return None
    if not example.supporting_sentences or not example.context_documents:
        return None

    rng = _stable_rng(cfg.seed, example.source_id, example.variant_id, origin_track)
    max_replace = min(cfg.replace_supporting_facts_max, len(example.supporting_sentences))
    min_replace = max(1, min(cfg.replace_supporting_facts_min, max_replace))
    num_replace = rng.randint(min_replace, max_replace)
    replaced_supports = rng.sample(list(example.supporting_sentences), k=num_replace)

    replacement_by_support: dict[tuple[str, int], tuple[str, int, str]] = {}
    replacement_sentences: list[dict[str, object]] = []
    replaced_meta: list[dict[str, object]] = []
    for support in replaced_supports:
        replacement = _choose_replacement(support=support, example=example, cfg=cfg, rng=rng)
        if replacement is None:
            return None
        replacement_by_support[(support.title, support.sent_id)] = replacement
        replacement_sentences.append(
            {
                "title": replacement[0],
                "sent_id": int(replacement[1]),
                "sentence": str(replacement[2]).strip(),
            }
        )
        replaced_meta.append(
            {
                "title": support.title,
                "sent_id": int(support.sent_id),
            }
        )

    knowledge_blocks: list[str] = []
    for support in example.supporting_sentences:
        key = (support.title, support.sent_id)
        if key in replacement_by_support:
            rep_title, rep_sent_id, rep_sentence = replacement_by_support[key]
            block = _format_sentence_block(rep_title, rep_sentence, cfg.include_title_prefix)
        else:
            rep_sent_id = None
            block = str(support.window_text or "").strip()
        _ = rep_sent_id
        if block:
            knowledge_blocks.append(block)

    dedup_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for block in knowledge_blocks:
        if block not in seen_blocks:
            seen_blocks.add(block)
            dedup_blocks.append(block)

    knowledge = "\n\n".join(dedup_blocks).strip()
    if not knowledge or knowledge == example.knowledge:
        return None

    metadata = dict(example.metadata)
    metadata["origin_track"] = origin_track
    metadata["paired_answerable_id"] = paired_answerable_id
    metadata["paired_answerable_source_id"] = paired_answerable_source_id
    metadata["raw_source_split"] = raw_source_split or example.split
    metadata["negative_strategy"] = {
        "mode": "support_replace_v1",
        "replaced_supports": replaced_meta,
        "replacement_sentences": replacement_sentences,
        "construction_from": "answerable_example" if origin_track == "paired_answerable" else "raw_hotpot_row",
        "same_doc_candidate_radius": cfg.same_doc_candidate_radius,
        "adjacent_doc_sentence_limit": cfg.adjacent_doc_sentence_limit,
    }
    variant_id = "unanswerable-paired-v1" if origin_track == "paired_answerable" else "unanswerable-external-v1"

    return QaExample(
        id=f"{example.source_id}:{variant_id}",
        source_id=example.source_id,
        variant_id=variant_id,
        split=example.split,
        question=example.question,
        knowledge=knowledge,
        reference_answer=example.reference_answer,
        answerability_label="unanswerable",
        difficulty=example.difficulty,
        supporting_sentences=example.supporting_sentences,
        context_documents=example.context_documents,
        metadata=metadata,
    )


def build_unanswerable_from_raw_hotpot_row(
    row: Dict[str, object],
    construct_cfg: ConstructionConfig,
    cfg: UnanswerableBuildConfig,
) -> Optional[QaExample]:
    """Build one unanswerable example from a raw Hotpot row via an answerable scaffold."""

    scaffold = build_answerable_example(row, construct_cfg)
    if scaffold is None:
        return None
    scaffold = QaExample(
        id=scaffold.id,
        source_id=scaffold.source_id,
        variant_id=scaffold.variant_id,
        split=str(row.get("split") or scaffold.split or "").strip() or scaffold.split,
        question=scaffold.question,
        knowledge=scaffold.knowledge,
        reference_answer=scaffold.reference_answer,
        answerability_label=scaffold.answerability_label,
        difficulty=scaffold.difficulty,
        supporting_sentences=scaffold.supporting_sentences,
        context_documents=scaffold.context_documents,
        metadata=dict(scaffold.metadata),
    )
    return build_unanswerable_from_answerable_example(
        scaffold,
        cfg,
        origin_track="external_raw",
        paired_answerable_id=None,
        paired_answerable_source_id=None,
        raw_source_split=str(row.get("split") or "").strip() or scaffold.split,
    )


def select_unanswerable_input_pools(
    *,
    paired_examples: Sequence[QaExample],
    raw_hotpot_rows: Sequence[Dict[str, object]],
    target_split: str,
    cfg: UnanswerableBuildConfig,
) -> tuple[list[QaExample], list[Dict[str, object]]]:
    """Select paired and external input pools for unanswerable raw construction."""

    rng = _stable_rng(cfg.seed, target_split, "unanswerable-pools")
    paired_pool = [example for example in paired_examples if str(example.split or "") == target_split]
    external_pool = [dict(row) for row in raw_hotpot_rows if str(row.get("split") or "") == target_split]

    rng.shuffle(paired_pool)
    rng.shuffle(external_pool)

    if cfg.max_total_examples > 0:
        paired_target = max(0, min(len(paired_pool), round(cfg.max_total_examples * cfg.paired_fraction)))
        external_target = max(0, cfg.max_total_examples - paired_target)
    else:
        paired_target = len(paired_pool)
        external_target = paired_target

    paired_selected = paired_pool[:paired_target]
    paired_source_ids = {item.source_id for item in paired_selected}
    external_selected: list[Dict[str, object]] = []
    for row in external_pool:
        source_id = str(row.get("_id") or row.get("id") or row.get("_source_index") or "").strip()
        if source_id in paired_source_ids:
            continue
        external_selected.append(row)
        if len(external_selected) >= external_target:
            break

    return paired_selected, external_selected


def build_unanswerable_examples_from_pools(
    *,
    paired_examples: Sequence[QaExample],
    raw_hotpot_rows: Sequence[Dict[str, object]],
    target_split: str,
    construct_cfg: ConstructionConfig,
    cfg: UnanswerableBuildConfig,
) -> list[QaExample]:
    """Build unanswerable examples from paired answerable and external raw pools."""

    paired_selected, external_selected = select_unanswerable_input_pools(
        paired_examples=paired_examples,
        raw_hotpot_rows=raw_hotpot_rows,
        target_split=target_split,
        cfg=cfg,
    )

    out: list[QaExample] = []
    seen_ids: set[str] = set()
    for example in paired_selected:
        built = build_unanswerable_from_answerable_example(
            example,
            cfg,
            origin_track="paired_answerable",
            paired_answerable_id=example.id,
            paired_answerable_source_id=example.source_id,
            raw_source_split=example.split,
        )
        if built is None or built.id in seen_ids:
            continue
        seen_ids.add(built.id)
        out.append(built)

    for row in external_selected:
        built = build_unanswerable_from_raw_hotpot_row(row, construct_cfg, cfg)
        if built is None or built.id in seen_ids:
            continue
        seen_ids.add(built.id)
        out.append(built)

    return out
