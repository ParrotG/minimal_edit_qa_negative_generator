from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy


@dataclass(frozen=True)
class EntitySpan:
    """Entity span extracted from text."""

    text: str
    label: str
    start: int
    end: int


class NERTagger:
    """Lightweight wrapper around spaCy named-entity recognition."""

    def __init__(self, model_name: str = "en_core_web_trf") -> None:
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> List[EntitySpan]:
        """Extract entity spans from text."""

        doc = self.nlp(text)
        spans: List[EntitySpan] = []
        for ent in doc.ents:
            spans.append(EntitySpan(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char))
        return spans


_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|)(?!\w)")
_YEAR_RE = re.compile(r"\b\d{4}\b")


def extract_numbers(text: str) -> List[Tuple[str, int, int]]:
    """Extract numeric spans with offsets."""

    return [(m.group(0), m.start(), m.end()) for m in _NUMBER_RE.finditer(text)]


def _question_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if q.startswith("who"):
        return "who"
    if q.startswith("where"):
        return "where"
    if q.startswith("when"):
        return "when"
    if q.startswith("how many") or q.startswith("how much"):
        return "how_many"
    return "other"


def check_answer_type(question: str, answer: str, ner: Optional[NERTagger]) -> Dict[str, Any]:
    """Run question-type/answer-type consistency checks for one answer."""

    if ner is None:
        return {
            "enabled": False,
            "question_type": _question_type(question),
            "answer_type_ok": None,
            "reason": "disabled",
            "entity_labels": [],
            "has_number": None,
            "has_year": None,
        }

    q_type = _question_type(question)
    ents = ner.extract(answer)
    labels = sorted({e.label for e in ents})
    has_num = len(extract_numbers(answer)) > 0
    has_year = bool(_YEAR_RE.search(answer or ""))

    if q_type == "who":
        ok = "PERSON" in labels or "ORG" in labels
        reason = "expects_person_or_org"
    elif q_type == "where":
        ok = "GPE" in labels or "LOC" in labels
        reason = "expects_location"
    elif q_type == "when":
        ok = "DATE" in labels or "TIME" in labels or has_year
        reason = "expects_time"
    elif q_type == "how_many":
        ok = has_num
        reason = "expects_number"
    else:
        ok = True
        reason = "not_type_constrained"

    return {
        "enabled": True,
        "question_type": q_type,
        "answer_type_ok": bool(ok),
        "reason": reason,
        "entity_labels": labels,
        "has_number": bool(has_num),
        "has_year": bool(has_year),
    }
