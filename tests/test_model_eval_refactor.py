from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    from qa_checks import check_answer_correctness
    from qa_checks.correctness import CorrectnessConfig

    from model_eval.answer_extraction import parse_answer_extraction_output
    from model_eval.common import (
        extract_knowledge_question_from_infer_prompt,
        load_dataset_split,
        load_structured_generation_items,
        normalize_generated_row,
    )
    from model_eval.correctness_bem import AnswerEquivalenceBemJudge, BemConfig
    from model_eval.generate_structured_answers import _build_prompt, _generate_batch_with_retry
    from model_eval.merge_eval_curves import _merge_rows
except ModuleNotFoundError:  # pragma: no cover
    check_answer_correctness = None
    CorrectnessConfig = None
    parse_answer_extraction_output = None
    extract_knowledge_question_from_infer_prompt = None
    load_dataset_split = None
    load_structured_generation_items = None
    normalize_generated_row = None
    AnswerEquivalenceBemJudge = None
    BemConfig = None
    _build_prompt = None
    _generate_batch_with_retry = None
    _merge_rows = None


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@unittest.skipUnless(
    all(
        item is not None
        for item in (
            extract_knowledge_question_from_infer_prompt,
            load_dataset_split,
            load_structured_generation_items,
            normalize_generated_row,
            check_answer_correctness,
            parse_answer_extraction_output,
            AnswerEquivalenceBemJudge,
            BemConfig,
            _build_prompt,
            _generate_batch_with_retry,
            _merge_rows,
        )
    ),
    "model_eval dependencies are not available",
)
class ModelEvalRefactorTests(unittest.TestCase):
    def test_extract_knowledge_question_from_infer_prompt(self) -> None:
        prompt = (
            "Return exactly one JSON object.\n\n"
            "Knowledge:\nDocA: Alpha is true.\n\nDocB: Beta is true.\n\n"
            "Question: Who is true?\n"
        )
        knowledge, question = extract_knowledge_question_from_infer_prompt(prompt)
        self.assertIn("DocA: Alpha is true.", knowledge)
        self.assertEqual(question, "Who is true?")

    def test_normalize_generated_row_supports_rationale_plus_answer(self) -> None:
        row = {
            "model_tag": "base",
            "model_step": 0,
            "model_path": "Qwen/Qwen3-0.6B",
            "source_id": "s1",
            "question": "Who founded Acme?",
            "knowledge": "Acme was founded by Alice.",
            "parsed_output": {
                "answerability": "answerable",
                "evidence": [{"quote": "Acme was founded by Alice."}],
                "rationale": "The knowledge explicitly names the founder as Alice.",
                "answer": "Alice",
                "confidence": "high",
            },
        }
        normalized = normalize_generated_row(row, idx=0, answer_source="rationale_plus_answer")
        self.assertIsNotNone(normalized)
        self.assertIn("Therefore the answer is Alice", normalized["answer"])

    def test_load_dataset_split_handles_jsonl_with_mixed_nested_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "a",
                        "prompt": "Knowledge:\nK1\n\nQuestion: Q1",
                        "completion": '{"answer":"A1"}',
                        "metadata": {"source_prefilter": {"infer_prompt_tokens": 128}},
                    },
                    {
                        "id": "b",
                        "prompt": "Knowledge:\nK2\n\nQuestion: Q2",
                        "completion": '{"answer":"A2"}',
                        "metadata": {"source_prefilter": {"infer_prompt_tokens": 64, "issues": []}, "nli_prefilter": {"keep": True}},
                    },
                ],
            )
            ds = load_dataset_split(str(path), split="train")
            self.assertEqual(len(ds), 2)

    def test_load_structured_generation_items_from_prompt_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "x1",
                        "source_id": "s1",
                        "prompt": "Instruction\n\nKnowledge:\nDoc1: Alice founded Acme.\n\nQuestion: Who founded Acme?\n",
                        "metadata": {"reference_answer": "Alice", "answerability_label": "answerable"},
                    }
                ],
            )
            items = load_structured_generation_items(
                data_path=str(path),
                split="train",
                max_samples=-1,
                seed=42,
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["question"], "Who founded Acme?")
            self.assertIn("Alice founded Acme", items[0]["knowledge"])
            self.assertEqual(items[0]["reference_answer"], "Alice")

    def test_answer_extraction_parser_handles_json_payload(self) -> None:
        result = parse_answer_extraction_output(
            '{"short_answer":"Magic Johnson","refusal_detected":false,"refusal_reason":""}'
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.short_answer, "Magic Johnson")
        self.assertFalse(result.refusal_detected)

    def test_answer_extraction_parser_clears_answer_on_refusal(self) -> None:
        result = parse_answer_extraction_output(
            '{"short_answer":"something","refusal_detected":true,"refusal_reason":"missing info"}'
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.short_answer, "")
        self.assertTrue(result.refusal_detected)

    def test_bem_judge_reviews_batch_with_mocked_model(self) -> None:
        class DummyTokenizer:
            def __call__(self, **kwargs):
                import torch

                return {
                    "input_ids": torch.ones((2, 4), dtype=torch.long),
                    "attention_mask": torch.ones((2, 4), dtype=torch.long),
                }

        class DummyModel:
            def __init__(self) -> None:
                self.config = Namespace(id2label={0: "not_equivalent", 1: "equivalent"}, num_labels=2)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **kwargs):
                import torch

                return Namespace(logits=torch.tensor([[0.1, 1.2], [1.4, 0.2]], dtype=torch.float32))

        with patch("model_eval.correctness_bem.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
            with patch("model_eval.correctness_bem.AutoModelForSequenceClassification.from_pretrained", return_value=DummyModel()):
                judge = AnswerEquivalenceBemJudge(BemConfig(device="cpu", batch_size=2))
                reports = judge.review_batch(
                    [
                        {"question": "Q1", "reference_answer": "A1", "answer": "B1"},
                        {"question": "Q2", "reference_answer": "A2", "answer": "B2"},
                    ]
                )
        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0].ok)
        self.assertFalse(reports[1].ok)

    def test_teacher_fewshot_prompt_contains_examples(self) -> None:
        sample = {
            "question": "Who founded Acme?",
            "knowledge": "Acme was founded by Alice.",
            "reference_answer": "Alice",
            "prompt": "",
        }
        prompt = _build_prompt(sample, prompt_mode="teacher_fewshot", fewshot_k=2)
        self.assertIn("Few-shot format examples", prompt)
        self.assertIn("Example 1", prompt)
        self.assertIn("Example 2", prompt)
        self.assertIn("Now solve the next case.", prompt)

    def test_retry_on_protocol_fail_keeps_sample_and_updates_attempt_count(self) -> None:
        class DummyGenerator:
            def __init__(self) -> None:
                self.calls = 0

            def generate_many(self, prompts, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ["not json at all" for _ in prompts]
                return [
                    '{"answerability":"answerable","evidence":[{"quote":"Acme was founded by Alice."}],'
                    '"rationale":"The quote states the founder.","answer":"Alice","confidence":"high"}'
                    for _ in prompts
                ]

        args = Namespace(
            prompt_mode="teacher_fewshot",
            fewshot_k=2,
            retry_on_protocol_fail=True,
            max_attempts=3,
            batch_size=1,
            max_new_tokens=64,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            min_p=None,
            repetition_penalty=1.0,
            use_chat_template=False,
            enable_thinking=False,
            strip_think_tags=True,
            strip_role_markers=True,
            split="validation",
        )
        sample = {
            "id": "s1",
            "sample_id": 0,
            "source_id": "src-1",
            "question": "Who founded Acme?",
            "knowledge": "Acme was founded by Alice.",
            "reference_answer": "Alice",
            "answerability_label": "answerable",
            "data_split": "validation",
            "prompt": "",
        }
        out = _generate_batch_with_retry(generator=DummyGenerator(), batch=[sample], args=args)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["parse_ok"])
        self.assertTrue(out[0]["protocol_passed"])
        self.assertFalse(out[0]["generation_failed"])
        self.assertEqual(out[0]["attempt_count"], 2)

    def test_merge_rows_prefixes_metrics_and_fills_missing_later(self) -> None:
        merged = {}
        _merge_rows(
            merged,
            rows=[{"model_tag": "base", "model_step": "0", "model_path": "m", "metric_a": "1.0"}],
            prefix="base_protocol",
        )
        _merge_rows(
            merged,
            rows=[{"model_tag": "sft", "model_step": "100", "model_path": "m2", "metric_b": "2.0"}],
            prefix="sft_structured",
        )
        self.assertIn(("base", "0", "m"), merged)
        self.assertEqual(merged[("base", "0", "m")]["base_protocol__metric_a"], "1.0")
        self.assertEqual(merged[("sft", "100", "m2")]["sft_structured__metric_b"], "2.0")


if __name__ == "__main__":
    unittest.main()
