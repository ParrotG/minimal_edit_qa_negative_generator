from __future__ import annotations

from typing import Any, Dict, Optional

from qa_protocol import build_infer_prompt, parse_structured_output, to_canonical_json, validate_structured_payload


def _resolve_structured_output(row: Dict[str, Any]) -> Optional[tuple[Dict[str, Any], str]]:
    payload = row.get("parsed_output")
    if payload:
        parsed = validate_structured_payload(payload)
        return parsed.model_dump(mode="json"), to_canonical_json(parsed)

    canonical_output = str(row.get("canonical_output") or "").strip()
    if canonical_output:
        parse_result = parse_structured_output(canonical_output)
        if parse_result.ok and parse_result.parsed is not None:
            return parse_result.parsed.model_dump(mode="json"), to_canonical_json(parse_result.parsed)
    return None


def build_sft_record(row: Dict[str, Any], prompt_style: str) -> Optional[Dict[str, Any]]:
    """Convert one validated structured row into an SFT prompt-completion record."""

    structured_pair = _resolve_structured_output(row)
    if structured_pair is None:
        return None

    structured_payload, canonical_output = structured_pair
    answerability = str(structured_payload.get("answerability") or "").strip()
    confidence = str(structured_payload.get("confidence") or "").strip()
    validation_report = dict(row.get("validation_report") or {})
    derived_confidence = str(validation_report.get("derived_confidence") or "").strip()
    if answerability == "answerable" and not derived_confidence:
        return None
    if answerability == "answerable":
        structured_payload["confidence"] = derived_confidence
        canonical_output = to_canonical_json(validate_structured_payload(structured_payload))
    if answerability == "unanswerable" and not confidence:
        return None
    if prompt_style != "infer_v1":
        raise ValueError(f"Unsupported SFT prompt style: {prompt_style}")

    prompt = build_infer_prompt(
        knowledge=str(row.get("knowledge") or "").strip(),
        question=str(row.get("question") or "").strip(),
    )

    return {
        "id": str(row.get("id") or "").strip(),
        "source_id": str(row.get("source_id") or "").strip(),
        "split": row.get("split"),
        "prompt_style": prompt_style,
        "prompt": prompt,
        "completion": canonical_output,
        "target_structured": structured_payload,
        "metadata": {
            "reference_answer": str(row.get("reference_answer") or "").strip(),
            "answerability_label": str(row.get("answerability_label") or "").strip(),
            "difficulty": str(row.get("difficulty") or "").strip(),
            "validation_report": validation_report,
        },
    }
