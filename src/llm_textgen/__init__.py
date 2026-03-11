"""Unified local/LoRA LLM text generation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "UnifiedLLMConfig",
    "UnifiedTextGenerator",
    "TokenUsage",
    "LocalGenerationResult",
    "GeneratorModelSpec",
    "build_generator_model_specs",
    "load_generator_from_spec",
]


def __getattr__(name: str) -> Any:
    if name == "UnifiedLLMConfig":
        return getattr(import_module(".config", __name__), name)
    if name == "UnifiedTextGenerator":
        return getattr(import_module(".generator", __name__), name)
    if name in {"TokenUsage", "LocalGenerationResult"}:
        return getattr(import_module(".types", __name__), name)
    if name in {"GeneratorModelSpec", "build_generator_model_specs", "load_generator_from_spec"}:
        return getattr(import_module(".model_loader", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
