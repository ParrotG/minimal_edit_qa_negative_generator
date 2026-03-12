from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    from calibrate.common import build_annotation_pack_rows, group_rows, load_annotation_rows
    from qa_checks import check_answer_correctness
    from qa_checks.correctness import CorrectnessConfig

    from model_eval.answer_extraction import parse_answer_extraction_output
    from model_eval.common import (
        extract_knowledge_question_from_infer_prompt,
        flat_content_eval_skipped_reason,
        is_flat_answerable_for_content_eval,
        is_structured_answerable_for_semantic_eval,
        extract_structured_output_text,
        load_dataset_split,
        load_structured_generation_items,
        normalize_generated_row,
    )
    from model_eval.correctness_transformer_matcher import (
        AnswerEquivalenceTransformerMatcher,
        TransformerMatcherConfig,
    )
    from model_eval.eval_grounded_qa import _binary_auroc
    from model_eval.generate_structured_answers import _build_prompt, _generate_batch_with_retry
    from model_eval.merge_eval_curves import _project_rows
    from model_eval.run_sft_eval_report import _select_best_checkpoint, run_sft_eval_report
    from llm_textgen.api_client import ApiGenerationConfig, ApiGenerationResult, OpenAICompatibleTextGenerator
    from llm_textgen.generator import UnifiedTextGenerator
    from project_config import PROJECT_SETTINGS
except ModuleNotFoundError:  # pragma: no cover
    build_annotation_pack_rows = None
    group_rows = None
    load_annotation_rows = None
    check_answer_correctness = None
    CorrectnessConfig = None
    parse_answer_extraction_output = None
    extract_knowledge_question_from_infer_prompt = None
    flat_content_eval_skipped_reason = None
    is_flat_answerable_for_content_eval = None
    is_structured_answerable_for_semantic_eval = None
    extract_structured_output_text = None
    load_dataset_split = None
    load_structured_generation_items = None
    normalize_generated_row = None
    AnswerEquivalenceTransformerMatcher = None
    TransformerMatcherConfig = None
    _binary_auroc = None
    _build_prompt = None
    _generate_batch_with_retry = None
    _project_rows = None
    _select_best_checkpoint = None
    run_sft_eval_report = None
    ApiGenerationConfig = None
    ApiGenerationResult = None
    OpenAICompatibleTextGenerator = None
    PROJECT_SETTINGS = None
    UnifiedTextGenerator = None


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@unittest.skipUnless(
    all(
        item is not None
        for item in (
            extract_knowledge_question_from_infer_prompt,
            flat_content_eval_skipped_reason,
            is_flat_answerable_for_content_eval,
            is_structured_answerable_for_semantic_eval,
            build_annotation_pack_rows,
            group_rows,
            load_annotation_rows,
            load_dataset_split,
            load_structured_generation_items,
            normalize_generated_row,
            check_answer_correctness,
            parse_answer_extraction_output,
            extract_structured_output_text,
            AnswerEquivalenceTransformerMatcher,
            TransformerMatcherConfig,
            _binary_auroc,
            _build_prompt,
            _generate_batch_with_retry,
            _project_rows,
            _select_best_checkpoint,
            run_sft_eval_report,
            ApiGenerationConfig,
            ApiGenerationResult,
            OpenAICompatibleTextGenerator,
            PROJECT_SETTINGS,
            UnifiedTextGenerator,
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
        self.assertEqual(normalized["eval_track"], "sft_structured")
        self.assertEqual(normalized["eval_variant"], "checkpoint")

    def test_extract_structured_output_text_parses_raw_output(self) -> None:
        row = {
            "raw_output": (
                '{"answerability":"answerable","evidence":[{"quote":"Acme was founded by Alice."}],'
                '"rationale":"The quote identifies the founder.","answer":"Alice","confidence":"high"}'
            )
        }
        text, meta = extract_structured_output_text(row)
        self.assertIn("Therefore the answer is Alice", text)
        self.assertTrue(meta["structured_parse_ok"])
        self.assertTrue(meta["structured_answer_present"])
        self.assertTrue(meta["structured_rationale_present"])
        self.assertFalse(meta["structured_extract_failed"])

    def test_shared_gate_helpers_respect_answerability_and_refusal(self) -> None:
        flat_row = {
            "extraction_parse_ok": True,
            "refusal_detected": False,
            "extracted_answer": "Alice",
            "pred_answerability": "answerable",
        }
        self.assertTrue(is_flat_answerable_for_content_eval(flat_row))
        self.assertEqual(flat_content_eval_skipped_reason(flat_row), "")
        refusal_row = {
            "extraction_parse_ok": True,
            "refusal_detected": True,
            "extracted_answer": "",
            "pred_answerability": "unanswerable",
        }
        self.assertFalse(is_flat_answerable_for_content_eval(refusal_row))
        self.assertEqual(flat_content_eval_skipped_reason(refusal_row), "predicted_refusal")

        structured_row = {
            "parsed_output": {
                "answerability": "answerable",
                "evidence": [{"quote": "Acme was founded by Alice."}],
                "rationale": "The quote names Alice as the founder.",
                "answer": "Alice",
                "confidence": "high",
            }
        }
        self.assertTrue(is_structured_answerable_for_semantic_eval(structured_row))
        structured_unanswerable = {
            "parsed_output": {
                "answerability": "unanswerable",
                "evidence": [],
                "rationale": "The founder is not stated.",
                "answer": "I don't know based on the provided knowledge.",
                "confidence": "low",
            }
        }
        self.assertFalse(is_structured_answerable_for_semantic_eval(structured_unanswerable))

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

    def test_transformer_matcher_reviews_batch_with_mocked_backend(self) -> None:
        class DummyMatcher:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def get_score(self, reference_answer: str, candidate_answer: str, question: str) -> float:
                return 0.9 if question == "Q1" else 0.1

            def transformer_match(self, reference_answers, candidate_answer: str, question: str) -> bool:
                return question == "Q1"

        with patch("model_eval.correctness_transformer_matcher.QaMetricsTransformerMatcher", DummyMatcher):
            judge = AnswerEquivalenceTransformerMatcher(
                TransformerMatcherConfig(model_name="zli12321/answer_equivalence_roberta-large")
            )
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

            def generate_many_results(self, prompts, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [type("Result", (), {"text": "not json at all", "token_usage": None})() for _ in prompts]
                return [
                    type(
                        "Result",
                        (),
                        {
                            "text": (
                                '{"answerability":"answerable","evidence":[{"quote":"Acme was founded by Alice."}],'
                                '"rationale":"The quote states the founder.","answer":"Alice","confidence":"high"}'
                            ),
                            "token_usage": None,
                        },
                    )()
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

    def test_build_annotation_pack_rows_supports_structured_and_matcher_tasks(self) -> None:
        rows = [
            {
                "sample_id": 1,
                "source_id": "s1",
                "model_tag": "base",
                "model_step": 0,
                "model_path": "model",
                "question": "Who founded Acme?",
                "knowledge": "Acme was founded by Alice.",
                "reference_answer": "Alice",
                "parsed_output": {
                    "answerability": "answerable",
                    "evidence": [{"quote": "Acme was founded by Alice."}],
                    "rationale": "The evidence states that Alice founded Acme.",
                    "answer": "Alice",
                    "confidence": "high",
                },
                "correctness_ok": False,
            }
        ]
        pack_rows, metrics = build_annotation_pack_rows(
            rows=rows,
            task_types=["nli_structured", "matcher"],
            max_samples_per_task=-1,
            seed=42,
            pack_id="pack-1",
        )
        self.assertEqual(len(pack_rows), 2)
        structured_row = next(row for row in pack_rows if row["task_type"] == "nli_structured")
        self.assertIn("Therefore the answer is Alice", structured_row["hypothesis_text"])
        matcher_row = next(row for row in pack_rows if row["task_type"] == "matcher")
        self.assertEqual(matcher_row["reference_answer"], "Alice")
        self.assertEqual(metrics["tasks"]["nli_structured"]["num_kept"], 1)

    def test_build_annotation_pack_rows_skips_flat_refusal_without_eval_entry(self) -> None:
        rows = [
            {
                "sample_id": 1,
                "source_id": "s1",
                "model_tag": "base",
                "model_step": 0,
                "model_path": "model",
                "question": "Who founded Acme?",
                "knowledge": "Acme was founded by Alice.",
                "reference_answer": "Alice",
                "answer": "I do not know based on the knowledge.",
                "extraction_parse_ok": True,
                "refusal_detected": True,
                "extracted_answer": "",
                "pred_answerability": "unanswerable",
            }
        ]
        pack_rows, metrics = build_annotation_pack_rows(
            rows=rows,
            task_types=["nli_flat", "matcher"],
            max_samples_per_task=-1,
            seed=42,
            pack_id="pack-2",
        )
        self.assertEqual(pack_rows, [])
        self.assertEqual(metrics["tasks"]["nli_flat"]["num_skipped_refusal_or_unanswerable"], 1)
        self.assertEqual(metrics["tasks"]["matcher"]["num_skipped_refusal_or_unanswerable"], 1)

    def test_build_annotation_pack_rows_skips_raw_flat_rows_without_eval_state(self) -> None:
        rows = [
            {
                "sample_id": 1,
                "source_id": "s1",
                "model_tag": "base",
                "model_step": 0,
                "model_path": "model",
                "question": "Who founded Acme?",
                "knowledge": "Acme was founded by Alice.",
                "reference_answer": "Alice",
                "answer": "Alice",
            }
        ]
        pack_rows, metrics = build_annotation_pack_rows(
            rows=rows,
            task_types=["nli_flat", "matcher"],
            max_samples_per_task=-1,
            seed=42,
            pack_id="pack-3",
        )
        self.assertEqual(pack_rows, [])
        self.assertEqual(metrics["tasks"]["nli_flat"]["num_skipped_missing_eval_state"], 1)
        self.assertEqual(metrics["tasks"]["matcher"]["num_skipped_missing_eval_state"], 1)

    def test_load_annotation_rows_parses_binary_human_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "annotations.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "task_type": "matcher",
                        "sample_id": 1,
                        "source_id": "s1",
                        "question": "Q",
                        "knowledge": "K",
                        "reference_answer": "A",
                        "answer": "B",
                        "human_label": "yes",
                    }
                ],
            )
            rows = load_annotation_rows(str(path), split="train", seed=42, max_samples=-1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["human_label_bool"])

    def test_calibration_grouping_defaults_to_task_level(self) -> None:
        rows = [
            {"task_type": "nli_structured", "model_tag": "checkpoint-100", "model_step": 100, "model_path": "ckpt-100"},
            {"task_type": "nli_structured", "model_tag": "checkpoint-200", "model_step": 200, "model_path": "ckpt-200"},
        ]
        grouped = group_rows(rows, group_by="task")
        self.assertEqual(list(grouped.keys()), [("nli_structured",)])
        self.assertEqual(len(grouped[("nli_structured",)]), 2)

    def test_project_settings_use_task_level_f1_calibration_defaults(self) -> None:
        self.assertEqual(PROJECT_SETTINGS.calibration.group_by, "task")
        self.assertEqual(PROJECT_SETTINGS.calibration.search_objective, "f1")

    def test_api_generator_estimates_missing_token_usage(self) -> None:
        cfg = ApiGenerationConfig(
            model_name="demo-model",
            base_url="https://example.invalid",
            record_token_usage=True,
            token_usage_tokenizer_name=PROJECT_SETTINGS.model.default_tokenizer_name,
        )
        generator = OpenAICompatibleTextGenerator(cfg=cfg, api_key="dummy")

        async def fake_generate_many_async(prompts):
            return [ApiGenerationResult(ok=True, text="Alpha", index=0, token_usage=None)]

        with patch.object(generator, "_generate_many_async", side_effect=fake_generate_many_async), patch(
            "llm_textgen.api_client.count_text_tokens_batch",
            side_effect=[[3], [2]],
        ):
            results = generator.generate_many_results(["Prompt"])
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].token_usage)
        self.assertEqual(results[0].token_usage.prompt_tokens, 3)
        self.assertEqual(results[0].token_usage.completion_tokens, 2)
        self.assertEqual(results[0].token_usage.source, "estimated")

    def test_project_rows_uses_sft_schema_and_fills_nan(self) -> None:
        schema = ["model_tag", "model_step", "model_path", "eval_track", "eval_variant", "num_rows", "parse_ok_rate"]
        rows = [{"model_tag": "base", "model_step": "0", "model_path": "m", "eval_track": "base_task", "eval_variant": "no_think", "num_rows": "10"}]
        projected = _project_rows(rows, schema)
        self.assertEqual(projected[0]["model_tag"], "base")
        self.assertEqual(projected[0]["eval_track"], "base_task")
        self.assertEqual(projected[0]["num_rows"], "10")
        self.assertEqual(projected[0]["parse_ok_rate"], "nan")

    def test_qa_prompt_can_append_refusal_instruction(self) -> None:
        prompt = UnifiedTextGenerator._build_qa_prompt(
            knowledge="Doc: Alice founded Acme.",
            question="Who founded Acme?",
            encourage_refusal=True,
        )
        self.assertIn("If the provided knowledge is insufficient", prompt)
        self.assertTrue(prompt.endswith("Answer: "))

    def test_binary_auroc_uses_confidence_order(self) -> None:
        score = _binary_auroc([1.0, 2.0, 3.0, 1.0], [0, 0, 1, 1])
        self.assertIsNotNone(score)
        self.assertGreaterEqual(float(score), 0.5)

    def test_select_best_checkpoint_prefers_hard_constraint_then_main_targets(self) -> None:
        selection = _select_best_checkpoint(
            eval_rows=[
                {
                    "model_tag": "checkpoint-100",
                    "model_step": 100,
                    "model_path": "ckpt-100",
                    "eval_track": "sft_structured",
                    "eval_variant": "checkpoint",
                    "parse_ok_rate": 0.94,
                    "protocol_ok_rate_given_parse_ok": 0.99,
                    "evidence_substring_ok_rate": 0.99,
                    "correctness_reviewed_rate": 0.95,
                    "answerability_accuracy": 0.95,
                    "semantic_yes_rate": 0.90,
                },
                {
                    "model_tag": "checkpoint-200",
                    "model_step": 200,
                    "model_path": "ckpt-200",
                    "eval_track": "sft_structured",
                    "eval_variant": "checkpoint",
                    "parse_ok_rate": 0.97,
                    "protocol_ok_rate_given_parse_ok": 0.99,
                    "evidence_substring_ok_rate": 0.97,
                    "correctness_reviewed_rate": 0.90,
                    "answerability_accuracy": 0.96,
                    "semantic_yes_rate": 0.89,
                },
            ],
            loss_rows=[
                {
                    "model_tag": "checkpoint-100",
                    "model_step": 100,
                    "model_path": "ckpt-100",
                    "eval_track": "sft_structured",
                    "eval_variant": "checkpoint",
                    "mean_loss": 1.1,
                },
                {
                    "model_tag": "checkpoint-200",
                    "model_step": 200,
                    "model_path": "ckpt-200",
                    "eval_track": "sft_structured",
                    "eval_variant": "checkpoint",
                    "mean_loss": 1.2,
                },
            ],
            args=Namespace(
                parse_ok_threshold=0.95,
                protocol_ok_threshold=0.98,
                evidence_ok_threshold=0.95,
            ),
        )
        self.assertTrue(selection["constraint_satisfied"])
        self.assertEqual(selection["selected_model_path"], "ckpt-200")

    def test_project_settings_define_default_training_model(self) -> None:
        self.assertEqual(PROJECT_SETTINGS.model.target_training_llm, "Qwen/Qwen3-0.6B")

    def test_run_sft_eval_report_keeps_validation_ckpt_only(self) -> None:
        structured_calls = []
        task_calls = []

        def fake_run_sft_loss_curve(args):
            return (
                [
                    {
                        "model_tag": "checkpoint-100",
                        "model_step": 100,
                        "model_path": "ckpt-100",
                        "eval_track": "sft_structured",
                        "eval_variant": "checkpoint",
                        "mean_loss": 1.0,
                    }
                ],
                [],
            )

        def fake_run_structured_generation(args):
            structured_calls.append((args.data_path, args.split, args.prompt_mode))
            return []

        def fake_run_grounded_eval(args):
            return (
                [
                    {
                        "model_tag": "checkpoint-100",
                        "model_step": 100,
                        "model_path": "ckpt-100",
                        "eval_track": "sft_structured" if "best_ckpt" not in args.generated_path and "base_protocol" not in args.generated_path else ("base_protocol" if "base_protocol" in args.generated_path else "sft_structured"),
                        "eval_variant": "checkpoint" if "base_protocol" not in args.generated_path else "fewshot_retry",
                        "parse_ok_rate": 0.99,
                        "protocol_ok_rate_given_parse_ok": 0.99,
                        "evidence_substring_ok_rate": 0.99,
                        "correctness_reviewed_rate": 0.9,
                        "answerability_accuracy": 0.9,
                        "semantic_yes_rate": 0.9,
                    }
                ],
                [],
                {},
            )

        def fake_select_best_checkpoint(eval_rows, loss_rows, args):
            return {
                "selected_model_path": "ckpt-100",
                "constraint_satisfied": True,
                "selection_metrics": {
                    "correctness_reviewed_rate": 0.9,
                    "answerability_accuracy": 0.9,
                    "semantic_yes_rate": 0.9,
                    "mean_loss": 1.0,
                },
            }

        def fake_run_answer_generation(args):
            task_calls.append((args.data_path, args.split, args.eval_variant))
            return []

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "model_eval.run_sft_eval_report.run_sft_loss_curve", side_effect=fake_run_sft_loss_curve
        ), patch(
            "model_eval.run_sft_eval_report.run_structured_generation", side_effect=fake_run_structured_generation
        ), patch(
            "model_eval.run_sft_eval_report.run_grounded_qa_evaluation", side_effect=fake_run_grounded_eval
        ), patch(
            "model_eval.run_sft_eval_report._select_best_checkpoint", side_effect=fake_select_best_checkpoint
        ), patch(
            "model_eval.run_sft_eval_report.run_answer_generation", side_effect=fake_run_answer_generation
        ), patch(
            "model_eval.run_sft_eval_report.run_task_content_baseline", return_value=([], [])
        ), patch(
            "model_eval.run_sft_eval_report.run_deepeval_hallucination", return_value=([], [])
        ), patch(
            "model_eval.run_sft_eval_report.merge_curve_rows", return_value=[]
        ):
            run_sft_eval_report(
                Namespace(
                    validation_data_path="val-ds",
                    test_data_path="test-ds",
                    validation_split="validation",
                    test_split="test",
                    validation_max_samples=10,
                    test_max_samples=10,
                    seed=42,
                    base_model="Qwen/Qwen3-0.6B",
                    lora_ckpt_path=None,
                    lora_ckpt_list_path="ckpts",
                    out_dir=tmpdir,
                    structured_batch_size=1,
                    task_batch_size=1,
                    max_length=1024,
                    structured_max_new_tokens=64,
                    base_protocol_max_new_tokens=64,
                    base_task_max_new_tokens=64,
                    base_task_think_max_new_tokens=128,
                    record_token_usage=False,
                    fewshot_k=2,
                    protocol_max_attempts=2,
                    protocol_temperature=0.2,
                    structured_temperature=0.0,
                    task_temperature=0.0,
                    parse_ok_threshold=0.95,
                    protocol_ok_threshold=0.98,
                    evidence_ok_threshold=0.95,
                    enable_semantics=True,
                    semantic_decision_source="full_binary",
                    semantic_match_f1_threshold=0.85,
                    nli_model_name="nli",
                    nli_device="cpu",
                    nli_batch_size=1,
                    nli_max_length=128,
                    nli_fp16=False,
                    temperature=3.0,
                    full_margin_threshold=0.6,
                    reject_margin_threshold=0.9,
                    reject_band_half_width=0.95,
                    qa_fail_as_negative=True,
                    qa_check_answer_type=True,
                    qa_spacy_model="en_core_web_trf",
                    matcher_model_name="matcher",
                    api_model_name="api",
                    api_base_url="https://example.invalid",
                    api_key_env="DASHSCOPE_API_KEY",
                    api_timeout_seconds=30.0,
                    api_max_concurrency=1,
                    api_max_retries=1,
                    api_backoff_base_seconds=0.1,
                    api_backoff_max_seconds=0.2,
                    api_extraction_max_new_tokens=64,
                    api_temperature=0.0,
                    api_top_p=1.0,
                    api_seed=42,
                    error_log_dir="log",
                    extraction_max_attempts=1,
                    deepeval_judge_model="judge",
                    deepeval_threshold=0.5,
                    deepeval_max_concurrent=1,
                    deepeval_throttle_value=0.0,
                )
            )

        self.assertEqual(structured_calls, [("val-ds", "validation", "infer"), ("test-ds", "test", "infer"), ("test-ds", "test", "teacher_fewshot")])
        self.assertEqual(task_calls, [("test-ds", "test", "no_think"), ("test-ds", "test", "think")])


if __name__ == "__main__":
    unittest.main()
