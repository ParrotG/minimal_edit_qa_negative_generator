from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounded_qa.config import ValidationConfig
from grounded_qa.workflow import validate_mixed_teacher_candidates
from qa_judge.structured import StructuredAnswerJudge
from qa_protocol import Answerability, ConfidenceLevel, EvidenceQuote, StructuredQaOutput, parse_structured_output


class FakeAnswerJudge:
    def __init__(self) -> None:
        self.rows = []

    def judge(self, rows):
        self.rows = list(rows)
        judged = []
        for row in rows:
            judged.append(
                {
                    **row,
                    "judge": {
                        "margin": 0.9,
                        "full_binary": {"decision": "yes", "supported_by_nli": True},
                        "reject_aware": {"decision": "yes"},
                        "qa_consistent": True,
                    },
                }
            )
        return judged, {"num_rows": float(len(rows))}


class FakeStructuredJudge:
    def __init__(self, decisions_by_question):
        self.decisions_by_question = decisions_by_question

    def judge_rows(self, rows, *, decision_source="full_binary"):
        judged = []
        for row in rows:
            question = str(row["question"])
            margin, decision = self.decisions_by_question.get(question, (None, None))
            judged.append(
                {
                    **row,
                    "structured_judge": {
                        "ok": True if decision == "yes" else (False if decision == "no" else None),
                        "supported": True if decision == "yes" else False if decision == "no" else None,
                        "refusal_ok": None,
                        "decision_source": decision_source,
                        "decision": decision,
                        "margin": margin,
                        "issues": [] if decision == "yes" else ["Semantic judge did not mark the rationale-answer chain as supported."],
                        "answer_judge": {
                            "margin": margin,
                            "qa_consistent": True,
                            "full_binary": {"decision": "yes" if decision == "yes" else "no", "supported_by_nli": decision == "yes"},
                            "reject_aware": {"decision": decision or "abstain"},
                        },
                    },
                }
            )
        return judged, {"num_rows": float(len(rows))}


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CanonicalizationTests(unittest.TestCase):
    def test_parse_structured_output_canonicalizes_fields(self) -> None:
        raw = (
            "\ufeff```json\r\n"
            '{"answerability":"answerable","evidence":[{"quote":"  Tom &amp; Jerry  "}],'
            '"rationale":"  Based on ＡＢＣ\\u200b.  ","answer":"  ＡＢＣ  ","confidence":"high"}\r\n```'
        )
        result = parse_structured_output(raw)
        self.assertTrue(result.ok)
        self.assertEqual(result.parsed.evidence[0].quote, "Tom & Jerry")
        self.assertEqual(result.parsed.rationale, "Based on ABC.")
        self.assertEqual(result.parsed.answer, "ABC")


class StructuredJudgeTests(unittest.TestCase):
    def test_answerable_semantics_use_evidence_only_and_hypothesis_chain(self) -> None:
        fake = FakeAnswerJudge()
        judge = StructuredAnswerJudge(answer_judge=fake)
        output = StructuredQaOutput(
            answerability=Answerability.ANSWERABLE,
            evidence=[EvidenceQuote(quote="Evidence sentence.")],
            rationale="The evidence states the entity directly.",
            answer="Entity",
            confidence=ConfidenceLevel.HIGH,
        )

        judged_rows, _ = judge.judge_rows(
            [
                {
                    "question": "Who is the entity?",
                    "knowledge": "Full knowledge that should not be used.",
                    "parsed_output": output,
                }
            ]
        )

        self.assertEqual(fake.rows[0]["knowledge"], "Evidence sentence.")
        self.assertEqual(
            fake.rows[0]["answer"],
            "The evidence states the entity directly.\nTherefore the answer is Entity",
        )
        self.assertEqual(judged_rows[0]["structured_judge"]["selected_knowledge"], "Evidence sentence.")

    def test_unanswerable_semantics_skip_nli(self) -> None:
        fake = FakeAnswerJudge()
        judge = StructuredAnswerJudge(answer_judge=fake)
        output = StructuredQaOutput(
            answerability=Answerability.UNANSWERABLE,
            evidence=[],
            rationale="The knowledge does not provide enough information.",
            answer="I don't know based on the provided knowledge.",
            confidence=ConfidenceLevel.LOW,
        )

        judged_rows, metrics = judge.judge_rows([{"question": "Q", "knowledge": "K", "parsed_output": output}])
        self.assertEqual(fake.rows, [])
        self.assertEqual(metrics["num_rows"], 0.0)
        self.assertIsNone(judged_rows[0]["structured_judge"]["ok"])


class ValidateWorkflowTests(unittest.TestCase):
    @patch("grounded_qa.workflow.count_text_tokens", side_effect=lambda text, _: len(str(text).split()))
    def test_validate_keeps_soft_supporting_fact_penalty_and_assigns_quantile_confidence(self, _mock_tokens) -> None:
        rows = [
            {
                "id": "a",
                "source_id": "a",
                "candidate_id": 0,
                "question": "Q-low",
                "knowledge": "Alpha is the capital of Country A. Distractor sentence.",
                "reference_answer": "Alpha",
                "answerability_label": "answerable",
                "supporting_sentences": [{"sentence": "A different support sentence."}],
                "raw_output": json.dumps(
                    {
                        "answerability": "answerable",
                        "evidence": [{"quote": "Alpha is the capital of Country A."}],
                        "rationale": "The evidence states that Alpha is the capital.",
                        "answer": "Alpha",
                        "confidence": "high",
                    }
                ),
            },
            {
                "id": "b",
                "source_id": "b",
                "candidate_id": 0,
                "question": "Q-mid",
                "knowledge": "Beta is the capital of Country B. Distractor sentence.",
                "reference_answer": "Beta",
                "answerability_label": "answerable",
                "supporting_sentences": [{"sentence": "Another unrelated support sentence."}],
                "raw_output": json.dumps(
                    {
                        "answerability": "answerable",
                        "evidence": [{"quote": "Beta is the capital of Country B."}],
                        "rationale": "The evidence identifies Beta as the capital.",
                        "answer": "Beta",
                        "confidence": "high",
                    }
                ),
            },
            {
                "id": "c",
                "source_id": "c",
                "candidate_id": 0,
                "question": "Q-high",
                "knowledge": "Gamma is the capital of Country C. Distractor sentence.",
                "reference_answer": "Gamma",
                "answerability_label": "answerable",
                "supporting_sentences": [{"sentence": "Support sentence still does not contain the quote."}],
                "raw_output": json.dumps(
                    {
                        "answerability": "answerable",
                        "evidence": [{"quote": "Gamma is the capital of Country C."}],
                        "rationale": "The evidence directly names Gamma as the capital.",
                        "answer": "Gamma",
                        "confidence": "high",
                    }
                ),
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "teacher.jsonl"
            out_path = Path(tmpdir) / "validated.jsonl"
            selected_path = Path(tmpdir) / "selected.jsonl"
            _write_jsonl(in_path, rows)

            fake_judge = FakeStructuredJudge(
                {
                    "Q-low": (0.1, "yes"),
                    "Q-mid": (0.5, "yes"),
                    "Q-high": (0.9, "yes"),
                }
            )
            with patch("grounded_qa.workflow.StructuredAnswerJudge.from_defaults", return_value=fake_judge):
                validate_mixed_teacher_candidates(
                    in_path=str(in_path),
                    out_path=str(out_path),
                    selected_out_path=str(selected_path),
                    metrics_out=None,
                    validation_cfg=ValidationConfig(enable_semantics=True),
                )

            validated = _read_jsonl(out_path)
            selected = _read_jsonl(selected_path)
            self.assertEqual(len(selected), 3)
            by_id = {row["id"]: row for row in selected}
            self.assertTrue(by_id["a"]["validation_report"]["hard_pass"])
            self.assertEqual(by_id["a"]["validation_report"]["soft_metrics"]["supporting_fact_match_rate"], 0.0)
            self.assertEqual(by_id["a"]["validation_report"]["derived_confidence"], "low")
            self.assertEqual(by_id["b"]["validation_report"]["derived_confidence"], "medium")
            self.assertEqual(by_id["c"]["validation_report"]["derived_confidence"], "high")
            self.assertEqual(by_id["c"]["parsed_output"]["confidence"], "high")
            self.assertEqual(len(validated), 3)

    @patch("grounded_qa.workflow.count_text_tokens", side_effect=lambda text, _: len(str(text).split()))
    def test_validate_semantic_negative_only_drops_when_enabled(self, _mock_tokens) -> None:
        row = {
            "id": "neg",
            "source_id": "neg",
            "candidate_id": 0,
            "question": "Q-neg",
            "knowledge": "Delta is the capital of Country D.",
            "reference_answer": "Delta",
            "answerability_label": "answerable",
            "supporting_sentences": [{"sentence": "Delta is the capital of Country D."}],
            "raw_output": json.dumps(
                {
                    "answerability": "answerable",
                    "evidence": [{"quote": "Delta is the capital of Country D."}],
                    "rationale": "The evidence names Delta as the capital.",
                    "answer": "Delta",
                    "confidence": "high",
                }
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "teacher.jsonl"
            out_path = Path(tmpdir) / "validated.jsonl"
            _write_jsonl(in_path, [row])

            fake_judge = FakeStructuredJudge({"Q-neg": (0.2, "no")})
            with patch("grounded_qa.workflow.StructuredAnswerJudge.from_defaults", return_value=fake_judge):
                validate_mixed_teacher_candidates(
                    in_path=str(in_path),
                    out_path=str(out_path),
                    selected_out_path=None,
                    metrics_out=None,
                    validation_cfg=ValidationConfig(enable_semantics=True, semantic_drop_by_nli=False),
                )
            validated = _read_jsonl(out_path)
            self.assertTrue(validated[0]["validation_report"]["hard_pass"])

            with patch("grounded_qa.workflow.StructuredAnswerJudge.from_defaults", return_value=fake_judge):
                validate_mixed_teacher_candidates(
                    in_path=str(in_path),
                    out_path=str(out_path),
                    selected_out_path=None,
                    metrics_out=None,
                    validation_cfg=ValidationConfig(
                        enable_semantics=True,
                        semantic_drop_by_nli=True,
                        semantic_decision_source="full_binary",
                    ),
                )
            validated = _read_jsonl(out_path)
            self.assertFalse(validated[0]["validation_report"]["hard_pass"])

if __name__ == "__main__":
    unittest.main()
