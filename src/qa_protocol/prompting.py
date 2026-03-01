from __future__ import annotations

import json
from typing import Optional

from .answer_style import AnswerStylePolicy, choose_answer_style
from .schema import StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


def _schema_example() -> str:
    payload = StructuredQaOutput.model_json_schema()
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_teacher_prompt(
    *,
    knowledge: str,
    question: str,
    answerability_label: str,
    reference_answer: Optional[str] = None,
    support_hint: Optional[str] = None,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
    answer_style: Optional[AnswerStylePolicy] = None,
) -> str:
    """Build the teacher-generation prompt."""

    resolved_style = answer_style or choose_answer_style(question=question, reference_answer=reference_answer or "")
    answer_guidance = "Preserve the natural answer form when possible."
    if resolved_style == AnswerStylePolicy.SHORT_SENTENCE:
        answer_guidance = "Return a short sentence."
    elif resolved_style == AnswerStylePolicy.SHORT_PHRASE:
        answer_guidance = "Return a short phrase when the question naturally allows it."

    reference_block = ""
    if reference_answer:
        reference_block = (
            "Reference answer:\n"
            f"{reference_answer}\n\n"
            "Keep your final answer semantically consistent with the reference answer.\n\n"
        )

    support_block = ""
    if support_hint:
        support_block = f"Support hint:\n{support_hint}\n\n"

    return (
        "You are preparing structured grounded-QA supervision data.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Requirements:\n"
        "1) Use the field order: answerability, evidence, rationale, answer, confidence.\n"
        "2) Every evidence quote must be copied from the provided knowledge.\n"
        "3) The rationale must be a brief explanation of how the evidence leads to the answer.\n"
        "4) The rationale must not be identical to the final answer text.\n"
        "5) When the question is answerable, cite evidence and answer directly.\n"
        "6) When the question is unanswerable, use an allowed refusal template and keep evidence as [].\n"
        f"7) {answer_guidance}\n\n"
        f"Target answerability label: {answerability_label}\n\n"
        f"{reference_block}"
        f"{support_block}"
        "JSON schema:\n"
        f"{_schema_example()}\n\n"
        f"Allowed refusal templates: {list(spec.refusal_templates)}\n\n"
        f"Knowledge:\n{knowledge}\n\n"
        f"Question: {question}\n"
    )


def build_infer_prompt(
    *,
    knowledge: str,
    question: str,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> str:
    """Build the inference-time prompt."""

    return (
        "Return exactly one JSON object with fields in this order:\n"
        f"{list(spec.field_order)}\n\n"
        "Rules:\n"
        "- If answerable, cite 1 to 4 quotes copied from the knowledge.\n"
        "- Provide a short rationale showing how the evidence supports the answer.\n"
        "- The rationale must not be identical to the answer.\n"
        "- If unanswerable, set evidence to [] and use an allowed refusal template.\n"
        "- Keep the answer concise and directly responsive.\n"
        "- Do not add explanations outside the JSON.\n\n"
        f"Allowed refusal templates: {list(spec.refusal_templates)}\n\n"
        f"Knowledge:\n{knowledge}\n\n"
        f"Question: {question}\n"
    )
