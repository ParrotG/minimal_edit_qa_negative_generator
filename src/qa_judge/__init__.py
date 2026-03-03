from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NLIConfig",
    "JudgeConfig",
    "NLIScores",
    "NLIVerifier",
    "AnswerJudge",
    "EntitySpan",
    "NERTagger",
    "extract_numbers",
    "check_answer_type",
    "build_qa_premise",
]


def __getattr__(name: str) -> Any:
    if name in {"NLIConfig", "JudgeConfig"}:
        return getattr(import_module(".config", __name__), name)
    if name in {"NLIScores", "NLIVerifier"}:
        return getattr(import_module(".nli", __name__), name)
    if name == "AnswerJudge":
        return getattr(import_module(".judge", __name__), name)
    if name in {"EntitySpan", "NERTagger", "check_answer_type", "extract_numbers"}:
        return getattr(import_module(".qa_consistency", __name__), name)
    if name == "build_qa_premise":
        return getattr(import_module("prompt"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
