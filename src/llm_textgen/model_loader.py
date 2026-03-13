from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from project_config import PROJECT_SETTINGS

from .config import UnifiedLLMConfig
from .generator import UnifiedTextGenerator


@dataclass(frozen=True)
class GeneratorModelSpec:
    """Model specification for one generation run."""

    tag: str
    step: int
    display_name: str
    lora_path: Optional[str]


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
    include_base: bool = True,
) -> List[GeneratorModelSpec]:
    """Build model specs from base model and optional LoRA adapter checkpoint(s)."""

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
            )
        )

    if lora_ckpt_path:
        specs.append(
            GeneratorModelSpec(
                tag="lora",
                step=0,
                display_name=lora_ckpt_path,
                lora_path=lora_ckpt_path,
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
    batch_size: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    use_chat_template: Optional[bool] = None,
    enable_thinking: Optional[bool] = None,
    strip_think_tags: Optional[bool] = None,
    strip_role_markers: Optional[bool] = None,
    record_token_usage: Optional[bool] = None,
    seed: Optional[int] = None,
) -> UnifiedTextGenerator:
    """Load UnifiedTextGenerator from one model spec."""

    cfg = UnifiedLLMConfig(
        base_model_name=base_model_name,
        lora_path=spec.lora_path,
        batch_size=PROJECT_SETTINGS.generation.batch_size if batch_size is None else batch_size,
        max_new_tokens=PROJECT_SETTINGS.generation.max_new_tokens if max_new_tokens is None else max_new_tokens,
        temperature=PROJECT_SETTINGS.generation.temperature if temperature is None else temperature,
        top_p=PROJECT_SETTINGS.generation.top_p if top_p is None else top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=PROJECT_SETTINGS.generation.repetition_penalty if repetition_penalty is None else repetition_penalty,
        use_chat_template=PROJECT_SETTINGS.generation.use_chat_template if use_chat_template is None else use_chat_template,
        enable_thinking=PROJECT_SETTINGS.generation.enable_thinking if enable_thinking is None else enable_thinking,
        strip_think_tags=PROJECT_SETTINGS.generation.strip_think_tags if strip_think_tags is None else strip_think_tags,
        strip_role_markers=PROJECT_SETTINGS.generation.strip_role_markers if strip_role_markers is None else strip_role_markers,
        record_token_usage=PROJECT_SETTINGS.token_budget.record_token_usage if record_token_usage is None else record_token_usage,
        seed=PROJECT_SETTINGS.generation.seed if seed is None else seed,
    )
    generator = UnifiedTextGenerator(config=cfg)
    generator.load_model()
    return generator
