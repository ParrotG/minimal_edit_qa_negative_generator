from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import UnifiedLLMConfig
from .types import LocalGenerationResult, TokenUsage
from prompt import build_qa_answer_prefix, build_qa_premise


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ROLE_PREFIX_RE = re.compile(r"^(?:\s*(?:system|user|assistant)\s*\n)+", flags=re.IGNORECASE)

def _sanitize_generated_text(text: str, strip_think_tags: bool, strip_role_markers: bool) -> str:
    """Strip common role/thinking artifacts from generated text."""

    out = (text or "").strip()
    if strip_role_markers:
        lower = out.lower()
        marker = "assistant\n"
        if marker in lower:
            idx = lower.rfind(marker)
            out = out[idx + len(marker) :].strip()
        out = _ROLE_PREFIX_RE.sub("", out).strip()
    if strip_think_tags:
        out = _THINK_BLOCK_RE.sub("", out).strip()
    return out


class UnifiedTextGenerator:
    """Unified local generation wrapper for base and LoRA models."""

    def __init__(self, config: Optional[UnifiedLLMConfig] = None) -> None:
        self.config = config or UnifiedLLMConfig()
        self.model: Any = None
        self.tokenizer: Any = None

    @staticmethod
    def _resolve(value: Any, default: Any) -> Any:
        return default if value is None else value

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_tokenizer(self, model_name_or_path: str) -> Any:
        tok = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=self.config.padding_side,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tok.pad_token_id is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            else:
                raise ValueError("Tokenizer has no pad/eos/unk token; cannot run generation.")
        tok.padding_side = self.config.padding_side
        return tok

    def _load_causal_lm(self, model_name_or_path: str, device_map: Optional[str]) -> Any:
        kwargs: Dict[str, Any] = {"trust_remote_code": self.config.trust_remote_code}
        if device_map is not None:
            kwargs["device_map"] = device_map
        if self.config.dtype:
            kwargs["dtype"] = self.config.dtype

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except TypeError:
            dtype = kwargs.pop("dtype", None)
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

        if device_map is None:
            model.to(self.config.device)
        model.eval()
        return model

    def load_model(
        self,
        *,
        base_model_name: Optional[str] = None,
        lora_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> None:
        """
        Load a generation model.

        Modes:
        - Base model only
        - PEFT LoRA adapter over base model
        """

        resolved_base_model = self._resolve(base_model_name, self.config.base_model_name)
        resolved_lora_path = self._resolve(lora_path, self.config.lora_path)
        resolved_device_map = self._resolve(device_map, self.config.device_map)
        resolved_tokenizer = self._resolve(tokenizer_name_or_path, self.config.tokenizer_name_or_path)

        if resolved_lora_path is None:
            tokenizer_source = resolved_tokenizer or resolved_base_model
            self.tokenizer = self._load_tokenizer(tokenizer_source)
            self.model = self._load_causal_lm(resolved_base_model, resolved_device_map)
            self._set_seed(self.config.seed)
            return

        tokenizer_source = resolved_tokenizer or resolved_base_model
        self.tokenizer = self._load_tokenizer(tokenizer_source)
        base_model = self._load_causal_lm(resolved_base_model, resolved_device_map)
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise ImportError("LoRA adapter loading requires `peft` to be installed.") from exc

        model = PeftModel.from_pretrained(base_model, resolved_lora_path)
        model.eval()
        self.model = model
        self._set_seed(self.config.seed)

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load_model()

    def _build_prompt(self, prompt: str, use_chat_template: bool, enable_thinking: bool) -> str:
        if not use_chat_template:
            return prompt
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    @staticmethod
    def _build_qa_prompt(knowledge: str, question: str, *, encourage_refusal: bool = False) -> str:
        """Build a QA-style prompt from knowledge and question fields."""

        question_text = str(question or "").strip()
        if not question_text:
            raise ValueError("question must not be empty.")
        if not encourage_refusal:
            return build_qa_answer_prefix(knowledge=knowledge, question=question_text)
        premise = build_qa_premise(knowledge=knowledge, question=question_text)
        return (
            "Use only the provided knowledge to answer the question.\n"
            "If the knowledge is insufficient, answer exactly: I do not know based on the knowledge.\n"
            "Do not repeat or paraphrase this instruction in the answer.\n"
            f"{premise}\n"
            "Answer: "
        )

    def _generate_batch_results(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        min_p: Optional[float],
        repetition_penalty: float,
        record_token_usage: bool,
    ) -> List[LocalGenerationResult]:
        prepared_prompts = list(prompts)
        inputs = self.tokenizer(
            prepared_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        do_sample = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if min_p is not None:
                gen_kwargs["min_p"] = min_p
        else:
            # Reset sampling params so they are not considered "active"
            self.model.generation_config.temperature = 1.0
            self.model.generation_config.top_p = 1.0
            self.model.generation_config.top_k = 50
            # (optional) also clear min_p if it exists
            if hasattr(self.model.generation_config, "min_p"):
                self.model.generation_config.min_p = None


        with torch.inference_mode():
            try:
                gen_ids = self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("min_p", None)
                gen_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_padded_len = int(inputs["input_ids"].shape[1])
        prompt_lengths = [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
        out: List[LocalGenerationResult] = []
        for idx in range(len(prepared_prompts)):
            text_ids = gen_ids[idx][prompt_padded_len:]
            token_usage = None
            if record_token_usage:
                prompt_tokens = int(prompt_lengths[idx])
                completion_tokens = int(text_ids.shape[0])
                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    source="estimated",
                )
            out.append(
                LocalGenerationResult(
                    text=self.tokenizer.decode(text_ids, skip_special_tokens=True),
                    token_usage=token_usage,
                )
            )
        return out

    def generate_one_result(
        self,
        prompt: str,
        *,
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
    ) -> LocalGenerationResult:
        """Generate one result object for a single prompt."""

        return self.generate_many_results(
            [prompt],
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
        )[0]

    def generate_one(
        self,
        prompt: str,
        *,
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
    ) -> str:
        """Generate text for one input prompt."""

        if prompt is None:
            raise ValueError("prompt must not be None.")
        output = self.generate_one_result(
            prompt,
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
        )
        return output.text

    def generate_many_results(
        self,
        prompts: Sequence[str],
        *,
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
    ) -> List[LocalGenerationResult]:
        """Generate result objects for a list of input prompts."""

        self._ensure_model_loaded()
        if not prompts:
            return []

        resolved_batch_size = int(self._resolve(batch_size, self.config.batch_size))
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        resolved_max_new_tokens = int(self._resolve(max_new_tokens, self.config.max_new_tokens))
        resolved_temperature = float(self._resolve(temperature, self.config.temperature))
        resolved_top_p = float(self._resolve(top_p, self.config.top_p))
        resolved_top_k = self._resolve(top_k, self.config.top_k)
        resolved_min_p = self._resolve(min_p, self.config.min_p)
        resolved_repetition_penalty = float(self._resolve(repetition_penalty, self.config.repetition_penalty))
        resolved_use_chat_template = bool(self._resolve(use_chat_template, self.config.use_chat_template))
        resolved_enable_thinking = bool(self._resolve(enable_thinking, self.config.enable_thinking))
        resolved_strip_think_tags = bool(self._resolve(strip_think_tags, self.config.strip_think_tags))
        resolved_strip_role_markers = bool(self._resolve(strip_role_markers, self.config.strip_role_markers))

        text_prompts = [str(prompt) for prompt in prompts]
        prepared_prompts = [
            self._build_prompt(
                prompt=prompt,
                use_chat_template=resolved_use_chat_template,
                enable_thinking=resolved_enable_thinking,
            )
            for prompt in text_prompts
        ]

        raw_outputs: List[LocalGenerationResult] = []
        for start in range(0, len(prepared_prompts), resolved_batch_size):
            batch_prompts = prepared_prompts[start : start + resolved_batch_size]
            raw_outputs.extend(
                self._generate_batch_results(
                    batch_prompts,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                    top_p=resolved_top_p,
                    top_k=resolved_top_k,
                    min_p=resolved_min_p,
                    repetition_penalty=resolved_repetition_penalty,
                    record_token_usage=bool(self.config.record_token_usage),
                )
            )

        return [
            LocalGenerationResult(
                text=_sanitize_generated_text(
                    output.text,
                    strip_think_tags=resolved_strip_think_tags,
                    strip_role_markers=resolved_strip_role_markers,
                ),
                token_usage=output.token_usage,
            )
            for output in raw_outputs
        ]

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
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
    ) -> List[str]:
        """Generate texts for a list of input prompts."""

        return [
            result.text
            for result in self.generate_many_results(
                prompts,
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
            )
        ]

    def generate_one_from_qa(
        self,
        *,
        knowledge: str,
        question: str,
        encourage_refusal: bool = False,
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
    ) -> str:
        """Generate one answer using (knowledge, question) inputs."""

        prompt = self._build_qa_prompt(knowledge=knowledge, question=question, encourage_refusal=encourage_refusal)
        return self.generate_one(
            prompt,
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
        )

    def generate_many_from_qa(
        self,
        qa_items: Sequence[Dict[str, str]],
        *,
        encourage_refusal: bool = False,
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
    ) -> List[str]:
        """Generate answers for a list of QA items with keys: knowledge, question."""

        prompts: List[str] = []
        for idx, item in enumerate(qa_items):
            if "question" not in item:
                raise ValueError(f"qa_items[{idx}] is missing required key: question")
            prompts.append(
                self._build_qa_prompt(
                    knowledge=str(item.get("knowledge", "")),
                    question=str(item["question"]),
                    encourage_refusal=encourage_refusal,
                )
            )

        return self.generate_many(
            prompts,
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
        )

    def generate_many_results_from_qa(
        self,
        qa_items: Sequence[Dict[str, str]],
        *,
        encourage_refusal: bool = False,
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
    ) -> List[LocalGenerationResult]:
        """Generate result objects for QA items with keys: knowledge, question."""

        prompts: List[str] = []
        for idx, item in enumerate(qa_items):
            if "question" not in item:
                raise ValueError(f"qa_items[{idx}] is missing required key: question")
            prompts.append(
                self._build_qa_prompt(
                    knowledge=str(item.get("knowledge", "")),
                    question=str(item["question"]),
                    encourage_refusal=encourage_refusal,
                )
            )

        return self.generate_many_results(
            prompts,
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
        )
