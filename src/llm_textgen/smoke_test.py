from __future__ import annotations

import argparse
from typing import List

from .config import UnifiedLLMConfig
from .generator import UnifiedTextGenerator
from .types import LocalGenerationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run smoke test for unified LLM text generation.")
    parser.add_argument("--base_model_name", type=str, default=UnifiedLLMConfig.base_model_name)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=UnifiedLLMConfig.device)
    parser.add_argument("--device_map", type=str, default=UnifiedLLMConfig.device_map)

    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for single and batch generation.")
    parser.add_argument(
        "--second_prompt",
        type=str,
        default="Write one short sentence about reliable model loading.",
        help="Optional second prompt used for batch generation.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--expect_non_empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, fail if any output is empty.",
    )
    return parser.parse_args()


def run_smoke_test(args: argparse.Namespace) -> int:
    cfg = UnifiedLLMConfig(
        base_model_name=args.base_model_name,
        lora_path=args.lora_path,
        device=args.device,
        device_map=args.device_map,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    generator = UnifiedTextGenerator(config=cfg)
    generator.load_model()

    single_output = generator.generate_one(args.prompt)
    batch_prompts: List[str] = [args.prompt, args.second_prompt]
    batch_results: List[LocalGenerationResult] = generator.generate_many_results(batch_prompts)
    batch_outputs = [item.text for item in batch_results]

    if len(batch_outputs) != len(batch_prompts):
        raise AssertionError(
            f"Batch generation length mismatch: expected {len(batch_prompts)}, got {len(batch_outputs)}."
        )
    if args.expect_non_empty:
        if not single_output.strip():
            raise AssertionError("Single output is empty.")
        if any(not item.strip() for item in batch_outputs):
            raise AssertionError("At least one batch output is empty.")

    print("Smoke test passed.")
    print(f"[single] {single_output}")
    for idx, text in enumerate(batch_outputs):
        print(f"[batch:{idx}] {text}")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_smoke_test(args))


if __name__ == "__main__":
    main()
