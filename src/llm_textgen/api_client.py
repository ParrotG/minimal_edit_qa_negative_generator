from __future__ import annotations

import asyncio
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

from dataio import write_jsonl
from project_config import PROJECT_SETTINGS
from qa_protocol.token_budget import count_text_tokens_batch

from .types import TokenUsage


def _extract_chat_content(content: Any) -> str:
    """Extract a text payload from an OpenAI-compatible message content field."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content or "").strip()


@dataclass(frozen=True)
class ApiGenerationConfig:
    """Generic OpenAI-compatible text generation configuration."""

    model_name: str
    base_url: str
    timeout_seconds: float = 90.0
    max_concurrency: int = 64
    max_retries: int = 4
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42
    error_log_dir: str = PROJECT_SETTINGS.paths.error_log_dir
    record_token_usage: bool = PROJECT_SETTINGS.token_budget.record_token_usage
    token_usage_tokenizer_name: str = PROJECT_SETTINGS.model.default_tokenizer_name


@dataclass(frozen=True)
class ApiGenerationResult:
    """One API generation result, including per-item failure state."""

    ok: bool
    text: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    index: int = 0
    token_usage: Optional[TokenUsage] = None


API_ERROR_PLACEHOLDER = "__API_GENERATION_ERROR__"


class _RetryableApiError(RuntimeError):
    """Internal exception used to trigger exponential backoff."""


class OpenAICompatibleTextGenerator:
    """Minimal reusable wrapper around an OpenAI-compatible chat-completions API."""

    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, cfg: ApiGenerationConfig, api_key: str) -> None:
        if not api_key:
            raise ValueError("Missing API key for OpenAI-compatible generation.")
        self.cfg = cfg
        self.api_key = api_key
        self._rng = random.Random(cfg.seed)
        self._endpoint = f"{cfg.base_url.rstrip('/')}/chat/completions"
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._error_log_path = Path(cfg.error_log_dir) / f"api_generation_errors_{self._run_timestamp}.jsonl"

    def _build_payload(self, prompt: str, seed: int) -> Dict[str, Any]:
        return {
            "model": self.cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "seed": seed,
            "enable_thinking": False,
        }

    @staticmethod
    def _usage_from_provider_payload(data: Dict[str, Any]) -> Optional[TokenUsage]:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None
        if total_tokens is None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        return TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            source="provider",
        )

    async def _post_with_retry(self, client: httpx.AsyncClient, prompt: str, seed: int) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt=prompt, seed=seed)

        for retry in range(self.cfg.max_retries + 1):
            try:
                resp = await client.post(self._endpoint, headers=headers, json=payload)
                if int(resp.status_code) in self._RETRYABLE_STATUS:
                    raise _RetryableApiError(f"Retryable status code: {resp.status_code}")
                if int(resp.status_code) >= 400:
                    raise RuntimeError(f"API request failed with status={resp.status_code}: {resp.text[:300]}")

                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("API response is not a JSON object.")
                return data
            except (_RetryableApiError, httpx.TimeoutException, httpx.NetworkError):
                if retry >= self.cfg.max_retries:
                    raise
                base = self.cfg.backoff_base_seconds * (2**retry)
                sleep_s = min(self.cfg.backoff_max_seconds, base) * self._rng.uniform(0.8, 1.2)
                await asyncio.sleep(sleep_s)

        raise RuntimeError("API retry loop exited unexpectedly.")

    async def _generate_one(self, client: httpx.AsyncClient, prompt: str, index: int) -> ApiGenerationResult:
        data = await self._post_with_retry(client=client, prompt=prompt, seed=self.cfg.seed + index)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("API response has no choices.")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        return ApiGenerationResult(
            ok=True,
            text=_extract_chat_content(message.get("content")),
            index=index,
            token_usage=self._usage_from_provider_payload(data),
        )

    async def _generate_many_async(self, prompts: Sequence[str]) -> List[ApiGenerationResult]:
        concurrency = max(1, int(self.cfg.max_concurrency))
        limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
        timeout = httpx.Timeout(self.cfg.timeout_seconds)

        async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=True) as client:
            sem = asyncio.Semaphore(concurrency)

            async def _task(index: int, prompt: str) -> ApiGenerationResult:
                async with sem:
                    try:
                        return await self._generate_one(client=client, prompt=prompt, index=index)
                    except Exception as exc:
                        return ApiGenerationResult(
                            ok=False,
                            text=API_ERROR_PLACEHOLDER,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            index=index,
                        )

            return list(await asyncio.gather(*[_task(idx, prompt) for idx, prompt in enumerate(prompts)]))

    def _estimate_missing_token_usage(
        self,
        prompts: Sequence[str],
        results: Sequence[ApiGenerationResult],
    ) -> List[ApiGenerationResult]:
        if not self.cfg.record_token_usage:
            return list(results)

        missing_indices = [idx for idx, result in enumerate(results) if result.ok and result.token_usage is None]
        if not missing_indices:
            return list(results)

        prompt_counts = count_text_tokens_batch(
            [str(prompts[idx]) for idx in missing_indices],
            tokenizer_name=self.cfg.token_usage_tokenizer_name,
            batch_size=PROJECT_SETTINGS.token_budget.count_batch_size,
        )
        completion_counts = count_text_tokens_batch(
            [str(results[idx].text) for idx in missing_indices],
            tokenizer_name=self.cfg.token_usage_tokenizer_name,
            batch_size=PROJECT_SETTINGS.token_budget.count_batch_size,
        )
        updated = list(results)
        for offset, result_idx in enumerate(missing_indices):
            prompt_tokens = int(prompt_counts[offset])
            completion_tokens = int(completion_counts[offset])
            updated[result_idx] = ApiGenerationResult(
                ok=updated[result_idx].ok,
                text=updated[result_idx].text,
                error_type=updated[result_idx].error_type,
                error_message=updated[result_idx].error_message,
                index=updated[result_idx].index,
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    source="estimated",
                ),
            )
        return updated

    def _write_error_log(self, prompts: Sequence[str], results: Sequence[ApiGenerationResult]) -> Optional[str]:
        error_rows: List[Dict[str, Any]] = []
        for result in results:
            if result.ok:
                continue
            prompt = prompts[result.index] if 0 <= result.index < len(prompts) else ""
            error_rows.append(
                {
                    "timestamp": self._run_timestamp,
                    "index": int(result.index),
                    "model_name": self.cfg.model_name,
                    "base_url": self.cfg.base_url,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                    "prompt_preview": str(prompt)[:500],
                }
            )

        if not error_rows:
            return None
        write_jsonl(self._error_log_path, error_rows)
        return str(self._error_log_path)

    def generate_many_results(self, prompts: Sequence[str]) -> List[ApiGenerationResult]:
        """Generate one result object for each prompt without aborting on per-item failures."""

        if not prompts:
            return []
        prompt_list = list(prompts)
        results = asyncio.run(self._generate_many_async(prompt_list))
        results = self._estimate_missing_token_usage(prompt_list, results)
        self._write_error_log(prompt_list, results)
        return results

    def generate_many(self, prompts: Sequence[str]) -> List[str]:
        """Generate one response for each prompt."""

        if not prompts:
            return []
        return [result.text for result in self.generate_many_results(prompts)]

    def generate_one(self, prompt: str) -> str:
        """Generate one response for a single prompt."""

        return self.generate_many([prompt])[0]
