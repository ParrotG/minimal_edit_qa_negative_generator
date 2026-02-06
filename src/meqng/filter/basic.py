from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rapidfuzz.distance import Levenshtein

from ..ner import NERTagger, extract_numbers
from .base import FilterDecision


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class LengthRatioFilter:
    """Keep candidates with similar length to the chosen answer."""
    name: str = "length_ratio"
    min_ratio: float = 0.85
    max_ratio: float = 1.15

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        c1 = len(_norm_ws(chosen))
        c2 = len(_norm_ws(candidate))
        if c1 == 0:
            return FilterDecision(False, "empty_chosen", {})
        ratio = c2 / c1
        keep = self.min_ratio <= ratio <= self.max_ratio
        return FilterDecision(keep, "ok" if keep else "ratio_out_of_range", {"ratio": ratio})


@dataclass
class EditDistanceFilter:
    """Enforce minimal but non-zero edits via normalized Levenshtein distance."""
    name: str = "edit_distance"
    min_norm: float = 0.01
    max_norm: float = 0.25

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        a = _norm_ws(chosen)
        b = _norm_ws(candidate)
        if a == b:
            return FilterDecision(False, "identical", {"norm": 0.0})
        dist = Levenshtein.distance(a, b)
        norm = dist / max(1, max(len(a), len(b)))
        keep = self.min_norm <= norm <= self.max_norm
        return FilterDecision(keep, "ok" if keep else "edit_out_of_range", {"norm": norm, "dist": dist})


@dataclass
class AnswerTypeFilter:
    """Very lightweight answer-type constraint based on the question form."""
    name: str = "answer_type"
    spacy_model: str = "en_core_web_trf"

    def __post_init__(self) -> None:
        self.ner = NERTagger(self.spacy_model)

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        q = question.strip().lower()

        # Extract signals from candidate only.
        ents = self.ner.extract(candidate)
        has_num = len(extract_numbers(candidate)) > 0
        labels = {e.label for e in ents}

        def ok_person() -> bool:
            return "PERSON" in labels or "ORG" in labels

        def ok_place() -> bool:
            return "GPE" in labels or "LOC" in labels

        def ok_date() -> bool:
            return "DATE" in labels or "TIME" in labels or any(re.fullmatch(r"\d{4}", n[0].replace(",", "").replace("%", "")) for n in extract_numbers(candidate))

        if q.startswith("who"):
            keep = ok_person()
            return FilterDecision(keep, "ok" if keep else "expect_person", {"labels": list(labels)})
        if q.startswith("where"):
            keep = ok_place()
            return FilterDecision(keep, "ok" if keep else "expect_place", {"labels": list(labels)})
        if q.startswith("when"):
            keep = ok_date()
            return FilterDecision(keep, "ok" if keep else "expect_date", {"labels": list(labels)})
        if q.startswith("how many") or q.startswith("how much"):
            keep = has_num
            return FilterDecision(keep, "ok" if keep else "expect_number", {"has_num": has_num})

        # Default: no constraint.
        return FilterDecision(True, "ok", {})
