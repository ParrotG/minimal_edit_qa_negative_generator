from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .records import QaExample, SupportingSentence


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Configuration for the first-pass simple unanswerable builder."""

    drop_supporting_facts: int = 1
    max_same_doc_sentences: int = 2
    max_adjacent_doc_sentences: int = 2
    include_title_prefix: bool = True


def _format_sentences(title: str, sentences: List[str], include_title_prefix: bool) -> str:
    text = " ".join(str(item).strip() for item in sentences if str(item).strip()).strip()
    if not include_title_prefix or not title:
        return text
    return f"{title}: {text}".strip()


def derive_simple_unanswerable(
    example: QaExample,
    cfg: NegativeSamplingConfig,
) -> Optional[QaExample]:
    """Derive a simple unanswerable example by removing support and keeping nearby context."""

    if example.answerability_label != "answerable":
        return None
    if not example.supporting_sentences or not example.context_documents:
        return None

    num_drop = max(1, min(cfg.drop_supporting_facts, len(example.supporting_sentences)))
    kept_supports = list(example.supporting_sentences[:-num_drop])
    dropped_supports = list(example.supporting_sentences[-num_drop:])

    support_index = {(item.title, item.sent_id) for item in example.supporting_sentences}
    dropped_titles = {item.title for item in dropped_supports}
    context_order = {doc.title: idx for idx, doc in enumerate(example.context_documents)}

    knowledge_parts: List[str] = []
    for item in kept_supports:
        if item.window_text:
            knowledge_parts.append(item.window_text)

    for doc in example.context_documents:
        if doc.title not in dropped_titles:
            continue
        non_support_sentences = [
            sentence
            for sent_id, sentence in enumerate(doc.sentences)
            if (doc.title, sent_id) not in support_index
        ][: cfg.max_same_doc_sentences]
        if non_support_sentences:
            text = _format_sentences(doc.title, non_support_sentences, cfg.include_title_prefix)
            if text:
                knowledge_parts.append(text)

    if cfg.max_adjacent_doc_sentences > 0:
        for title in dropped_titles:
            center = context_order.get(title)
            if center is None:
                continue
            for delta in (-1, 1):
                idx = center + delta
                if idx < 0 or idx >= len(example.context_documents):
                    continue
                doc = example.context_documents[idx]
                sent_block = list(doc.sentences[: cfg.max_adjacent_doc_sentences])
                if not sent_block:
                    continue
                text = _format_sentences(doc.title, sent_block, cfg.include_title_prefix)
                if text:
                    knowledge_parts.append(text)

    dedup_parts: List[str] = []
    seen: set[str] = set()
    for part in knowledge_parts:
        if part and part not in seen:
            seen.add(part)
            dedup_parts.append(part)

    knowledge = "\n\n".join(dedup_parts).strip()
    if not knowledge:
        return None

    metadata = dict(example.metadata)
    metadata["negative_strategy"] = {
        "mode": "simple_support_drop",
        "drop_supporting_facts": num_drop,
        "max_same_doc_sentences": cfg.max_same_doc_sentences,
        "max_adjacent_doc_sentences": cfg.max_adjacent_doc_sentences,
    }

    return QaExample(
        id=f"{example.source_id}:unanswerable-simple",
        source_id=example.source_id,
        variant_id="unanswerable-simple",
        split=example.split,
        question=example.question,
        knowledge=knowledge,
        reference_answer=example.reference_answer,
        answerability_label="unanswerable",
        difficulty=example.difficulty,
        supporting_sentences=tuple(kept_supports),
        context_documents=example.context_documents,
        metadata=metadata,
    )
