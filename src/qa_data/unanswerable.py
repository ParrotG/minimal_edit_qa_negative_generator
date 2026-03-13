from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from project_config import PROJECT_SETTINGS

from .construct import ConstructionConfig, build_answerable_example
from .records import ContextDocument, QaExample, SupportingSentence


@dataclass(frozen=True)
class UnanswerableBuildConfig:
    """Configuration for unanswerable example construction."""

    replace_supporting_facts_min: int = PROJECT_SETTINGS.source.replace_supporting_facts_min
    replace_supporting_facts_max: int = PROJECT_SETTINGS.source.replace_supporting_facts_max
    same_doc_candidate_radius: int = PROJECT_SETTINGS.source.same_doc_candidate_radius
    allow_same_doc_non_adjacent: bool = PROJECT_SETTINGS.source.allow_same_doc_non_adjacent
    adjacent_doc_sentence_limit: int = PROJECT_SETTINGS.source.adjacent_doc_sentence_limit
    include_title_prefix: bool = PROJECT_SETTINGS.source.include_title_prefix
    seed: int = PROJECT_SETTINGS.source.seed


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
    origin_track: str,
    raw_source_id: str,
    data_split: str | None,
    answerability_split: str | None,
) -> Optional[QaExample]:
    """Derive one unanswerable example from an answerable scaffold."""

    if example.answerability_label != "answerable":
        return None
    if not example.supporting_sentences or not example.context_documents:
        return None

    rng = _stable_rng(cfg.seed, example.source_id, example.variant_id, origin_track, answerability_split or "")
    max_replace = min(cfg.replace_supporting_facts_max, len(example.supporting_sentences))
    if max_replace <= 0:
        return None
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
        replaced_meta.append({"title": support.title, "sent_id": int(support.sent_id)})

    knowledge_blocks: list[str] = []
    for support in example.supporting_sentences:
        key = (support.title, support.sent_id)
        if key in replacement_by_support:
            rep_title, _, rep_sentence = replacement_by_support[key]
            block = _format_sentence_block(rep_title, rep_sentence, cfg.include_title_prefix)
        else:
            block = str(support.window_text or "").strip()
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

    variant_id = "unanswerable-paired-v1" if origin_track == "paired_answerable" else "unanswerable-external-v1"
    metadata = dict(example.metadata)
    metadata.update(
        {
            "origin_track": origin_track,
            "raw_source_id": raw_source_id,
            "data_split": data_split,
            "answerability_split": answerability_split,
            "negative_strategy": {
                "mode": "support_replace_v1",
                "replaced_supports": replaced_meta,
                "replacement_sentences": replacement_sentences,
                "same_doc_candidate_radius": cfg.same_doc_candidate_radius,
                "adjacent_doc_sentence_limit": cfg.adjacent_doc_sentence_limit,
            },
        }
    )

    return QaExample(
        id=f"{example.source_id}:{variant_id}",
        source_id=example.source_id,
        variant_id=variant_id,
        data_split=data_split,
        answerability_split=answerability_split,
        question=example.question,
        knowledge=knowledge,
        reference_answer=example.reference_answer,
        answerability_label="unanswerable",
        difficulty=example.difficulty,
        supporting_sentences=example.supporting_sentences,
        context_documents=example.context_documents,
        metadata=metadata,
    )


def _clone_answerable_example(
    example: QaExample,
    *,
    data_split: str | None,
    answerability_split: str | None,
) -> QaExample:
    metadata = dict(example.metadata)
    metadata.update(
        {
            "origin_track": "answerable",
            "raw_source_id": example.source_id,
            "data_split": data_split,
            "answerability_split": answerability_split,
        }
    )
    return QaExample(
        id=example.id,
        source_id=example.source_id,
        variant_id=example.variant_id,
        data_split=data_split,
        answerability_split=answerability_split,
        question=example.question,
        knowledge=example.knowledge,
        reference_answer=example.reference_answer,
        answerability_label=example.answerability_label,
        difficulty=example.difficulty,
        supporting_sentences=example.supporting_sentences,
        context_documents=example.context_documents,
        metadata=metadata,
    )


def build_examples_from_tagged_row(
    row: Dict[str, object],
    construct_cfg: ConstructionConfig,
    unanswerable_cfg: UnanswerableBuildConfig,
) -> list[QaExample]:
    """Build concrete examples from one tagged Hotpot row."""

    answerability_split = str(row.get("answerability_split") or "").strip()
    data_split = str(row.get("data_split") or "").strip() or None
    hotpot_source_split = str(row.get("hotpot_source_split") or "").strip() or None
    scaffold = build_answerable_example(row, construct_cfg)
    if scaffold is None:
        return []

    out: list[QaExample] = []
    if answerability_split in {"answerable", "both"}:
        out.append(
            _clone_answerable_example(
                scaffold,
                data_split=data_split,
                answerability_split=answerability_split or None,
            )
        )
        if hotpot_source_split:
            out[-1].metadata["hotpot_source_split"] = hotpot_source_split

    if answerability_split in {"unanswerable", "both"}:
        origin_track = "paired_answerable" if answerability_split == "both" else "external_raw"
        derived = build_unanswerable_from_answerable_example(
            scaffold,
            unanswerable_cfg,
            origin_track=origin_track,
            raw_source_id=str(row.get("_id") or row.get("id") or row.get("_source_index") or scaffold.source_id),
            data_split=data_split,
            answerability_split=answerability_split or None,
        )
        if derived is not None:
            if hotpot_source_split:
                derived.metadata["hotpot_source_split"] = hotpot_source_split
            out.append(derived)

    return out


def build_prepared_examples(
    rows: Sequence[Dict[str, object]],
    construct_cfg: ConstructionConfig,
    unanswerable_cfg: UnanswerableBuildConfig,
) -> list[QaExample]:
    """Build mixed concrete examples from tagged raw rows."""

    out: list[QaExample] = []
    for row in rows:
        out.extend(build_examples_from_tagged_row(row, construct_cfg, unanswerable_cfg))
    return out
