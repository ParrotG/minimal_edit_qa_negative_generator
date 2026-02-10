from __future__ import annotations

import re
from typing import Dict

from rapidfuzz.distance import Levenshtein


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")
_DIGIT_RE = re.compile(r"\d")
_UPPER_RE = re.compile(r"[A-Z]")
_PUNCT_RE = re.compile(r"[.,;:!?\-()\[\]{}'\"]")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim text boundaries."""

    return _WS_RE.sub(" ", text or "").strip()


def normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized Levenshtein distance on whitespace-normalized strings."""

    a2 = normalize_whitespace(a)
    b2 = normalize_whitespace(b)
    if not a2 and not b2:
        return 0.0
    dist = Levenshtein.distance(a2, b2)
    return float(dist / max(1, max(len(a2), len(b2))))


def length_ratio(reference: str, candidate: str) -> float:
    """Compute candidate/reference length ratio on normalized strings."""

    ref = normalize_whitespace(reference)
    cand = normalize_whitespace(candidate)
    if not ref:
        return 0.0
    return float(len(cand) / len(ref))


def word_count(text: str) -> int:
    """Count words using a lightweight regex tokenizer."""

    return len(_WORD_RE.findall(text or ""))


def basic_surface_features(text: str) -> Dict[str, float]:
    """Extract lightweight surface features for artifact auditing."""

    t = text or ""
    n = max(1, len(t))
    words = word_count(t)
    return {
        "char_len": float(len(t)),
        "word_len": float(words),
        "avg_word_len": float(len(t) / max(1, words)),
        "digit_ratio": float(len(_DIGIT_RE.findall(t)) / n),
        "upper_ratio": float(len(_UPPER_RE.findall(t)) / n),
        "punct_ratio": float(len(_PUNCT_RE.findall(t)) / n),
        "comma_count": float(t.count(",")),
        "quote_count": float(t.count('"') + t.count("'")),
    }
