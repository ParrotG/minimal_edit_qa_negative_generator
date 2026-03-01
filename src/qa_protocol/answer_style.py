from __future__ import annotations

from enum import StrEnum


class AnswerStylePolicy(StrEnum):
    """Answer style policy for teacher prompting."""

    PRESERVE = "preserve"
    SHORT_PHRASE = "short_phrase"
    SHORT_SENTENCE = "short_sentence"


def choose_answer_style(question: str, reference_answer: str) -> AnswerStylePolicy:
    """Choose a conservative answer style policy.

    The default policy is to preserve the original answer surface when possible.
    """

    _ = question
    _ = reference_answer
    return AnswerStylePolicy.PRESERVE
