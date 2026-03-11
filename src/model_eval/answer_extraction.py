from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_textgen.api_client import ApiGenerationConfig, OpenAICompatibleTextGenerator
from project_config import PROJECT_SETTINGS


@dataclass(frozen=True)
class AnswerExtractionConfig:
    """Configuration for LLM-based answer extraction and refusal detection."""

    api_model_name: str = PROJECT_SETTINGS.answer_extraction_api.model_name
    api_base_url: str = PROJECT_SETTINGS.answer_extraction_api.base_url
    api_key_env: str = PROJECT_SETTINGS.answer_extraction_api.api_key_env
    api_timeout_seconds: float = PROJECT_SETTINGS.answer_extraction_api.timeout_seconds
    api_max_concurrency: int = PROJECT_SETTINGS.answer_extraction_api.max_concurrency
    api_max_retries: int = PROJECT_SETTINGS.answer_extraction_api.max_retries
    api_backoff_base_seconds: float = PROJECT_SETTINGS.answer_extraction_api.backoff_base_seconds
    api_backoff_max_seconds: float = PROJECT_SETTINGS.answer_extraction_api.backoff_max_seconds
    max_new_tokens: int = PROJECT_SETTINGS.answer_extraction_api.max_tokens
    temperature: float = PROJECT_SETTINGS.answer_extraction_api.temperature
    top_p: float = PROJECT_SETTINGS.answer_extraction_api.top_p
    seed: int = PROJECT_SETTINGS.answer_extraction_api.seed
    error_log_dir: str = PROJECT_SETTINGS.paths.error_log_dir
    max_attempts: int = PROJECT_SETTINGS.answer_extraction_api.max_attempts

    def to_api_config(self) -> ApiGenerationConfig:
        return ApiGenerationConfig(
            model_name=self.api_model_name,
            base_url=self.api_base_url,
            timeout_seconds=self.api_timeout_seconds,
            max_concurrency=self.api_max_concurrency,
            max_retries=self.api_max_retries,
            backoff_base_seconds=self.api_backoff_base_seconds,
            backoff_max_seconds=self.api_backoff_max_seconds,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            error_log_dir=self.error_log_dir,
        )


@dataclass(frozen=True)
class ExtractionParseResult:
    """Parsed extraction output for one model answer."""

    ok: bool
    short_answer: str = ""
    refusal_detected: Optional[bool] = None
    refusal_reason: str = ""
    errors: List[str] = field(default_factory=list)


def build_answer_extraction_prompt(*, question: str, raw_answer: str) -> str:
    """Build the prompt for short-answer extraction and refusal detection."""

    return (
        "Read the model answer and return exactly one JSON object.\n"
        "The JSON must use this schema and field order:\n"
        '{\n  "short_answer": string,\n  "refusal_detected": boolean,\n  "refusal_reason": string\n}\n\n'
        "Instructions:\n"
        "- Determine whether the model answer refuses to answer, says it does not know, or states that the information is missing.\n"
        "- If the answer is a refusal or non-answer, set refusal_detected to true and short_answer to an empty string.\n"
        "- If the answer contains a real answer, set refusal_detected to false and extract the shortest complete answer span needed to answer the question.\n"
        "- Do not include explanations outside the JSON.\n\n"
        f"Question: {question}\n"
        "Model answer:\n"
        f"{raw_answer}\n"
    )


def parse_answer_extraction_output(raw_text: str) -> ExtractionParseResult:
    """Parse the extractor JSON output."""

    text = str(raw_text or "").strip()
    if not text:
        return ExtractionParseResult(ok=False, errors=["Empty extraction output."])

    candidate = text
    if not candidate.startswith("{"):
        left = candidate.find("{")
        right = candidate.rfind("}")
        if left >= 0 and right > left:
            candidate = candidate[left : right + 1]

    try:
        payload = json.loads(candidate)
    except Exception as exc:
        return ExtractionParseResult(ok=False, errors=[f"Failed to parse extraction JSON: {exc}"])

    if not isinstance(payload, dict):
        return ExtractionParseResult(ok=False, errors=["Extraction payload is not a JSON object."])

    refusal_value = payload.get("refusal_detected")
    if isinstance(refusal_value, bool):
        refusal_detected = refusal_value
    elif isinstance(refusal_value, str):
        lowered = refusal_value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            refusal_detected = True
        elif lowered in {"false", "no", "0"}:
            refusal_detected = False
        else:
            return ExtractionParseResult(ok=False, errors=["refusal_detected is not a valid boolean value."])
    else:
        return ExtractionParseResult(ok=False, errors=["Missing refusal_detected boolean field."])

    short_answer = str(payload.get("short_answer") or "").strip()
    refusal_reason = str(payload.get("refusal_reason") or "").strip()
    if refusal_detected:
        short_answer = ""

    return ExtractionParseResult(
        ok=True,
        short_answer=short_answer,
        refusal_detected=refusal_detected,
        refusal_reason=refusal_reason,
        errors=[],
    )


def extract_answers_with_llm(
    *,
    rows: Sequence[Dict[str, Any]],
    generator: OpenAICompatibleTextGenerator,
    cfg: AnswerExtractionConfig,
) -> List[Dict[str, Any]]:
    """Extract one structured answer signal for each raw-answer row."""

    if not rows:
        return []

    prompts = [
        build_answer_extraction_prompt(
            question=str(row.get("question") or ""),
            raw_answer=str(row.get("answer") or ""),
        )
        for row in rows
    ]
    states: List[Dict[str, Any]] = [
        {
            "raw_response": "",
            "attempt_count": 0,
            "parse_result": None,
            "api_ok": False,
            "error_type": None,
            "error_message": None,
        }
        for _ in rows
    ]

    active_indices = list(range(len(rows)))
    max_attempts = max(1, int(cfg.max_attempts))
    for attempt in range(1, max_attempts + 1):
        if not active_indices:
            break
        result_batch = generator.generate_many_results([prompts[idx] for idx in active_indices])
        next_active: List[int] = []
        for state_idx, api_result in zip(active_indices, result_batch):
            parse_result = parse_answer_extraction_output(api_result.text) if api_result.ok else ExtractionParseResult(
                ok=False,
                errors=[str(api_result.error_message or "Extraction API call failed.")],
            )
            state = states[state_idx]
            state["raw_response"] = api_result.text
            state["attempt_count"] = attempt
            state["parse_result"] = parse_result
            state["api_ok"] = bool(api_result.ok)
            state["error_type"] = api_result.error_type
            state["error_message"] = api_result.error_message
            if attempt < max_attempts and (not api_result.ok or not parse_result.ok):
                next_active.append(state_idx)
        active_indices = next_active

    outputs: List[Dict[str, Any]] = []
    for row, state in zip(rows, states):
        parse_result = state["parse_result"]
        parse_ok = bool(parse_result is not None and parse_result.ok)
        outputs.append(
            {
                **row,
                "extraction_raw_response": state["raw_response"],
                "extraction_attempt_count": int(state["attempt_count"]),
                "extraction_api_ok": bool(state["api_ok"]),
                "extraction_parse_ok": parse_ok,
                "extraction_error_type": state["error_type"],
                "extraction_error_message": state["error_message"],
                "extracted_answer": "" if parse_result is None else parse_result.short_answer,
                "refusal_detected": None if parse_result is None else parse_result.refusal_detected,
                "refusal_reason": "" if parse_result is None else parse_result.refusal_reason,
                "extraction_errors": [] if parse_result is None else list(parse_result.errors),
            }
        )
    return outputs
