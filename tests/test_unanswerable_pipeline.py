from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounded_qa.config import ValidationConfig
from grounded_qa.workflow import validate_unanswerable_teacher_candidates
from qa_checks.unanswerable_prefilter import build_reference_answer_hypothesis, check_reference_answer_unsupported
from qa_data.records import ContextDocument, QaExample, SupportingSentence
from qa_data.unanswerable import UnanswerableBuildConfig, build_unanswerable_from_answerable_example
from qa_protocol.prompting import build_infer_prompt, build_teacher_prompt
from sft_trainer.formatting import build_sft_record


class FakePrefilterJudge:
    def __init__(self, decision: str, margin: float) -> None:
        self.decision = decision
        self.margin = margin

    def judge(self, rows):
        return (
            [
                {
                    **row,
                    "judge": {
                        "margin": self.margin,
                        "full_binary": {"decision": self.decision},
                    },
                }
                for row in rows
            ],
            {"num_rows": float(len(rows))},
        )


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class UnanswerableBuilderTests(unittest.TestCase):
    def test_build_unanswerable_replaces_support_with_neighbor_sentence(self) -> None:
        example = QaExample(
            id="s1:answerable",
            source_id="s1",
            variant_id="answerable",
            split="train_sft_raw",
            question="Who founded Acme?",
            knowledge="Doc1: Support sentence one.\n\nDoc2: Support sentence two.",
            reference_answer="Alice",
            answerability_label="answerable",
            difficulty="medium",
            supporting_sentences=(
                SupportingSentence(
                    title="Doc1",
                    sent_id=1,
                    sentence="Support sentence one.",
                    window_sentences=("Neighbor sentence.", "Support sentence one."),
                    window_text="Doc1: Support sentence one.",
                ),
                SupportingSentence(
                    title="Doc2",
                    sent_id=0,
                    sentence="Support sentence two.",
                    window_sentences=("Support sentence two.",),
                    window_text="Doc2: Support sentence two.",
                ),
            ),
            context_documents=(
                ContextDocument(title="Doc1", sentences=("Intro.", "Support sentence one.", "Neighbor sentence.")),
                ContextDocument(title="Doc2", sentences=("Support sentence two.", "Tail.")),
            ),
            metadata={},
        )

        built = build_unanswerable_from_answerable_example(example, UnanswerableBuildConfig(seed=7))
        self.assertIsNotNone(built)
        self.assertEqual(built.answerability_label, "unanswerable")
        self.assertNotEqual(built.knowledge, example.knowledge)
        self.assertIn("origin_track", built.metadata)
        self.assertIn("replacement_sentences", built.metadata["negative_strategy"])


class UnanswerablePrefilterTests(unittest.TestCase):
    def test_reference_answer_hypothesis_uses_fixed_template(self) -> None:
        self.assertEqual(build_reference_answer_hypothesis("Alpha"), "The answer is Alpha.")

    def test_prefilter_keeps_only_full_binary_no(self) -> None:
        keep_report = check_reference_answer_unsupported(
            knowledge="K",
            question="Q",
            reference_answer="A",
            judge=FakePrefilterJudge(decision="no", margin=-0.5),
        )
        drop_report = check_reference_answer_unsupported(
            knowledge="K",
            question="Q",
            reference_answer="A",
            judge=FakePrefilterJudge(decision="yes", margin=0.5),
        )
        self.assertTrue(keep_report.keep)
        self.assertFalse(drop_report.keep)


class UnanswerablePromptTests(unittest.TestCase):
    def test_teacher_prompt_mentions_missing_or_ambiguous_rationale(self) -> None:
        prompt = build_teacher_prompt(knowledge="Knowledge", question="Question?", reference_answer=None)
        self.assertIn("missing, ambiguous, or contradictory", prompt)
        self.assertNotIn("Reference answer:", prompt)

    def test_infer_prompt_mentions_missing_or_ambiguous_rationale(self) -> None:
        prompt = build_infer_prompt(knowledge="Knowledge", question="Question?")
        self.assertIn("missing, ambiguous, or contradictory", prompt)


class UnanswerableValidateTests(unittest.TestCase):
    @patch("grounded_qa.workflow.count_text_tokens", side_effect=lambda text, _: len(str(text).split()))
    def test_validate_unanswerable_selects_first_valid_candidate(self, _mock_tokens) -> None:
        rows = [
            {
                "id": "u1",
                "source_id": "u1",
                "candidate_id": 1,
                "question": "Question?",
                "knowledge": "Knowledge.",
                "answerability_label": "unanswerable",
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The knowledge misses the needed fact.",
                        "answer": "I don't know based on the provided knowledge.",
                        "confidence": "medium",
                    }
                ),
            },
            {
                "id": "u1",
                "source_id": "u1",
                "candidate_id": 0,
                "question": "Question?",
                "knowledge": "Knowledge.",
                "answerability_label": "unanswerable",
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The knowledge does not identify the missing entity.",
                        "answer": "I don't know based on the provided knowledge.",
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

            validate_unanswerable_teacher_candidates(
                in_path=str(in_path),
                out_path=str(out_path),
                selected_out_path=str(selected_path),
                metrics_out=None,
                validation_cfg=ValidationConfig(enable_semantics=False, semantic_drop_by_nli=False),
            )

            selected = _read_jsonl(selected_path)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_id"], 0)
            self.assertIsNone(selected[0]["validation_report"]["correctness"])
            self.assertIsNone(selected[0]["validation_report"]["semantics"])


class MixedSftPackingTests(unittest.TestCase):
    def test_unanswerable_sft_record_keeps_teacher_confidence(self) -> None:
        row = {
            "id": "u2",
            "source_id": "u2",
            "split": "train_sft_raw",
            "knowledge": "Knowledge.",
            "question": "Question?",
            "parsed_output": {
                "answerability": "unanswerable",
                "evidence": [],
                "rationale": "The key fact is missing from the knowledge.",
                "answer": "I don't know based on the provided knowledge.",
                "confidence": "low",
            },
            "validation_report": {
                "overall_ok": True,
                "derived_confidence": None,
            },
        }

        record = build_sft_record(row=row, prompt_style="infer_v1")
        self.assertIsNotNone(record)
        self.assertEqual(record["target_structured"]["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
