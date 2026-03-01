from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .records import ContextDocument, QaExample, SupportingSentence


@dataclass(frozen=True)
class ConstructionConfig:
    """Configuration for answerable-example construction from HotpotQA."""

    window_size: int = 1
    max_supporting_facts: int = 4
    drop_over_max_supporting_facts: bool = True
    include_title_prefix: bool = True


def _as_context_documents(row: Dict[str, object]) -> tuple[ContextDocument, ...]:
    context = row.get("context") or {}
    titles = list((context or {}).get("title") or [])
    sentence_lists = list((context or {}).get("sentences") or [])
    docs: List[ContextDocument] = []
    for title, sentences in zip(titles, sentence_lists):
        clean_sentences = tuple(str(item).strip() for item in list(sentences or []) if str(item).strip())
        docs.append(ContextDocument(title=str(title).strip(), sentences=clean_sentences))
    return tuple(docs)


def _window(sentences: Sequence[str], center: int, window_size: int) -> tuple[str, ...]:
    left = max(0, center - window_size)
    right = min(len(sentences), center + window_size + 1)
    return tuple(str(item).strip() for item in sentences[left:right] if str(item).strip())


def _format_window(title: str, window_sentences: Sequence[str], include_title_prefix: bool) -> str:
    text = " ".join(str(item).strip() for item in window_sentences if str(item).strip()).strip()
    if not include_title_prefix or not title:
        return text
    return f"{title}: {text}".strip()


def build_answerable_example(
    row: Dict[str, object],
    cfg: ConstructionConfig,
) -> Optional[QaExample]:
    """Build one answerable grounded-QA example from a HotpotQA row."""

    question = str(row.get("question") or "").strip()
    reference_answer = str(row.get("answer") or "").strip()
    difficulty = str(row.get("level") or "").strip() or "unknown"
    source_id = str(row.get("_id") or row.get("id") or row.get("_source_index") or "").strip()
    question_type = str(row.get("type") or "").strip()

    docs = _as_context_documents(row)
    doc_map = {doc.title: doc for doc in docs if doc.title}

    supporting_facts = row.get("supporting_facts") or {}
    support_titles = list((supporting_facts or {}).get("title") or [])
    support_sent_ids = list((supporting_facts or {}).get("sent_id") or [])
    support_pairs = list(dict.fromkeys((str(title).strip(), int(sent_id)) for title, sent_id in zip(support_titles, support_sent_ids)))

    if not question or not reference_answer or not source_id or not support_pairs:
        return None
    if cfg.drop_over_max_supporting_facts and len(support_pairs) > cfg.max_supporting_facts:
        return None

    supporting_sentences: List[SupportingSentence] = []
    knowledge_parts: List[str] = []
    seen_knowledge_parts: set[str] = set()

    for title, sent_id in support_pairs:
        doc = doc_map.get(title)
        if doc is None or sent_id < 0 or sent_id >= len(doc.sentences):
            return None

        sentence = doc.sentences[sent_id]
        window_sentences = _window(doc.sentences, sent_id, cfg.window_size)
        window_text = _format_window(title, window_sentences, cfg.include_title_prefix)
        supporting_sentences.append(
            SupportingSentence(
                title=title,
                sent_id=sent_id,
                sentence=sentence,
                window_sentences=window_sentences,
                window_text=window_text,
            )
        )
        if window_text not in seen_knowledge_parts:
            seen_knowledge_parts.add(window_text)
            knowledge_parts.append(window_text)

    knowledge = "\n\n".join(part for part in knowledge_parts if part).strip()
    if not knowledge:
        return None

    return QaExample(
        id=f"{source_id}:answerable",
        source_id=source_id,
        variant_id="answerable",
        split=None,
        question=question,
        knowledge=knowledge,
        reference_answer=reference_answer,
        answerability_label="answerable",
        difficulty=difficulty,
        supporting_sentences=tuple(supporting_sentences),
        context_documents=docs,
        metadata={
            "question_type": question_type,
            "knowledge_strategy": {
                "window_size": cfg.window_size,
                "include_title_prefix": cfg.include_title_prefix,
            },
            "negative_strategy": None,
        },
    )
