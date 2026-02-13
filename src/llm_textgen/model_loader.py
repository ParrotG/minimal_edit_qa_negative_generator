from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import UnifiedLLMConfig
from .generator import UnifiedTextGenerator


@dataclass(frozen=True)
class GeneratorModelSpec:
    """Model specification for one generation run."""

    tag: str
    step: int
    display_name: str
    lora_path: Optional[str]
    lora_is_merged: bool


def _discover_checkpoints(path: str) -> List[Tuple[int, str]]:
    """Discover checkpoint paths from a directory or a text list file."""

    out: List[Tuple[int, str]] = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            match = re.match(r"^checkpoint-(\d+)$", name)
            if match:
                out.append((int(match.group(1)), full))
    elif os.path.isfile(path):
        base_dir = os.path.dirname(os.path.abspath(path))
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for idx, item in enumerate(lines):
            ckpt_path = item if os.path.isabs(item) else os.path.join(base_dir, item)
            if not os.path.isdir(ckpt_path):
                raise ValueError(f"Checkpoint path from list does not exist: {ckpt_path}")
            name = os.path.basename(ckpt_path.rstrip("/"))
            match = re.match(r"^checkpoint-(\d+)$", name)
            step = int(match.group(1)) if match else idx
            out.append((step, ckpt_path))
    else:
        raise ValueError(f"Checkpoint path does not exist: {path}")

    out.sort(key=lambda x: x[0])
    return out


def build_generator_model_specs(
    *,
    base_model_name: str,
    lora_ckpt_path: Optional[str] = None,
    lora_ckpt_list_path: Optional[str] = None,
    lora_is_merged: bool = False,
    lora_list_are_merged: bool = False,
    include_base: bool = True,
) -> List[GeneratorModelSpec]:
    """Build model specs from base model and optional LoRA or checkpoint list."""

    if lora_ckpt_path and lora_ckpt_list_path:
        raise ValueError("--lora_ckpt_path and --lora_ckpt_list_path are mutually exclusive.")

    specs: List[GeneratorModelSpec] = []
    if include_base:
        specs.append(
            GeneratorModelSpec(
                tag="base",
                step=0,
                display_name=base_model_name,
                lora_path=None,
                lora_is_merged=False,
            )
        )

    if lora_ckpt_path:
        specs.append(
            GeneratorModelSpec(
                tag="lora",
                step=0,
                display_name=lora_ckpt_path,
                lora_path=lora_ckpt_path,
                lora_is_merged=bool(lora_is_merged),
            )
        )
        return specs

    if lora_ckpt_list_path:
        checkpoints = _discover_checkpoints(lora_ckpt_list_path)
        if not checkpoints:
            raise RuntimeError(f"No checkpoint-* directories found under: {lora_ckpt_list_path}")
        for step, ckpt_path in checkpoints:
            specs.append(
                GeneratorModelSpec(
                    tag="checkpoint",
                    step=int(step),
                    display_name=ckpt_path,
                    lora_path=ckpt_path,
                    lora_is_merged=bool(lora_list_are_merged),
                )
            )
        return specs

    if not include_base:
        raise ValueError("No model selected: set include_base=True or provide LoRA path/list path.")
    return specs


def load_generator_from_spec(
    *,
    spec: GeneratorModelSpec,
    base_model_name: str,
    merge_lora: bool = False,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    strip_think_tags: bool = True,
    strip_role_markers: bool = True,
    seed: int = 42,
) -> UnifiedTextGenerator:
    """Load UnifiedTextGenerator from one model spec."""

    cfg = UnifiedLLMConfig(
        base_model_name=base_model_name,
        lora_path=spec.lora_path,
        lora_is_merged=bool(spec.lora_is_merged),
        merge_lora=bool(merge_lora),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        strip_think_tags=strip_think_tags,
        strip_role_markers=strip_role_markers,
        seed=seed,
    )
    generator = UnifiedTextGenerator(config=cfg)
    generator.load_model()
    return generator
