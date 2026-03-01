from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .records import ContextDocument, QaExample, SupportingSentence


def example_to_dict(example: QaExample) -> Dict[str, Any]:
    """Convert one QA example dataclass into a JSON-serializable dictionary."""

    return asdict(example)


def _load_supporting_sentences(items: List[Dict[str, Any]]) -> tuple[SupportingSentence, ...]:
    out: List[SupportingSentence] = []
    for item in items:
        out.append(
            SupportingSentence(
                title=str(item.get("title") or "").strip(),
                sent_id=int(item.get("sent_id") or 0),
                sentence=str(item.get("sentence") or "").strip(),
                window_sentences=tuple(str(x).strip() for x in list(item.get("window_sentences") or []) if str(x).strip()),
                window_text=str(item.get("window_text") or "").strip(),
            )
        )
    return tuple(out)


def _load_context_documents(items: List[Dict[str, Any]]) -> tuple[ContextDocument, ...]:
    out: List[ContextDocument] = []
    for item in items:
        out.append(
            ContextDocument(
                title=str(item.get("title") or "").strip(),
                sentences=tuple(str(x).strip() for x in list(item.get("sentences") or []) if str(x).strip()),
            )
        )
    return tuple(out)


def example_from_dict(row: Dict[str, Any]) -> QaExample:
    """Restore one QA example dataclass from a JSON row."""

    return QaExample(
        id=str(row.get("id") or "").strip(),
        source_id=str(row.get("source_id") or "").strip(),
        variant_id=str(row.get("variant_id") or "").strip(),
        split=None if row.get("split") is None else str(row.get("split")).strip(),
        question=str(row.get("question") or "").strip(),
        knowledge=str(row.get("knowledge") or "").strip(),
        reference_answer=str(row.get("reference_answer") or "").strip(),
        answerability_label=str(row.get("answerability_label") or "").strip(),
        difficulty=str(row.get("difficulty") or "").strip(),
        supporting_sentences=_load_supporting_sentences(list(row.get("supporting_sentences") or [])),
        context_documents=_load_context_documents(list(row.get("context_documents") or [])),
        metadata=dict(row.get("metadata") or {}),
    )
