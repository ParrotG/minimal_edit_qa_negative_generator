from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GenerationConfig
from .prompt import build_generation_prompt
from .text import normalize_whitespace


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ROLE_PREFIX_RE = re.compile(r"^(?:\s*(?:system|user|assistant)\s*\n)+", flags=re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:：]?\s*(.+)$", flags=re.IGNORECASE | re.DOTALL)
_REASONING_HINT_RE = re.compile(
    r"(?:because|reasoning|let's|therefore|so,|explain|option|why)",
    flags=re.IGNORECASE,
)


def _build_chat_prompt(tokenizer: AutoTokenizer, plain_prompt: str, enable_thinking: bool) -> str:
    """Build chat-formatted prompt with graceful fallback when template is unavailable."""

    if not hasattr(tokenizer, "apply_chat_template"):
        return plain_prompt

    messages = [{"role": "user", "content": plain_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return plain_prompt


def _collapse_repeated_lines(text: str) -> str:
    """Collapse consecutive duplicate non-empty lines."""

    lines = text.splitlines()
    out: List[str] = []
    for line in lines:
        line2 = line.rstrip()
        if out and line2 and line2 == out[-1]:
            continue
        out.append(line2)
    return "\n".join(out).strip()


def _extract_short_answer(text: str) -> str:
    """Extract a concise answer span from verbose outputs when possible."""

    out = text.strip()
    if not out:
        return out

    m = _FINAL_ANSWER_RE.search(out)
    if m:
        cand = m.group(1).strip()
        if cand:
            first = next((x.strip() for x in cand.splitlines() if x.strip()), cand)
            return first

    lines = [x.strip() for x in out.splitlines() if x.strip()]
    if len(lines) >= 2:
        first = lines[0]
        second = lines[1]
        if len(first.split()) <= 20 and (_REASONING_HINT_RE.search(second) or second.lower().startswith("options:")):
            return first

    return out


def _sanitize_generated_text(text: str, cfg: GenerationConfig) -> str:
    """Remove common generation artifacts and keep answer text concise."""

    out = (text or "").strip()

    if cfg.strip_role_markers:
        lower = out.lower()
        marker = "assistant\n"
        if marker in lower:
            idx = lower.rfind(marker)
            out = out[idx + len(marker) :].strip()
        out = _ROLE_PREFIX_RE.sub("", out).strip()

    if cfg.strip_think_tags:
        out = _THINK_BLOCK_RE.sub("", out).strip()

    if cfg.strip_markdown_fences:
        out = out.replace("```", "\n")

    out = _collapse_repeated_lines(out)

    if cfg.extract_short_answer:
        out = _extract_short_answer(out)

    out = normalize_whitespace(out)
    return out


def _build_jobs(source_rows: Sequence[Dict[str, Any]], samples_per_question: int) -> List[Dict[str, Any]]:
    """Expand source rows into per-sample generation jobs."""

    jobs: List[Dict[str, Any]] = []
    for row in source_rows:
        for sid in range(samples_per_question):
            rec: Dict[str, Any] = {
                "id": row["id"],
                "knowledge": row["knowledge"],
                "question": row["question"],
                "sample_id": sid,
                "attempt": 0,
            }
            if row.get("reference_answer"):
                rec["reference_answer"] = row["reference_answer"]
            jobs.append(rec)
    return jobs


def _build_base_metrics(num_source_rows: int, num_requested: int) -> Dict[str, int]:
    """Build the common metric payload shared by local/API generation."""

    return {
        "num_source_rows": num_source_rows,
        "num_requested": num_requested,
        "num_generated": 0,
        "num_retried": 0,
        "num_drop_empty": 0,
        "num_drop_overlength": 0,
        "num_drop_truncated": 0,
        "num_drop_exhausted": 0,
    }


class SelfSampler:
    """Generate multiple sampled answers per QA item with a local causal LM."""

    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad/eos token; cannot run generation.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype="auto",
            device_map="auto" if cfg.device.startswith("cuda") else None,
        )
        self.model.eval()

    def _build_prompt(self, knowledge: str, question: str) -> str:
        plain_prompt = build_generation_prompt(knowledge, question)
        if not self.cfg.use_chat_template:
            return plain_prompt
        return _build_chat_prompt(self.tokenizer, plain_prompt=plain_prompt, enable_thinking=self.cfg.enable_thinking)

    def _generate_batch(self, batch: Sequence[Dict[str, Any]]) -> Tuple[List[List[int]], int]:
        prompts = [self._build_prompt(x["knowledge"], x["question"]) for x in batch]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        do_sample = self.cfg.temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.cfg.repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p
            gen_kwargs["top_k"] = self.cfg.top_k
            if self.cfg.min_p is not None:
                gen_kwargs["min_p"] = self.cfg.min_p

        with torch.inference_mode():
            try:
                gen_ids = self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("min_p", None)
                gen_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_padded_len = int(inputs["input_ids"].shape[1])
        generated_ids = [ids[prompt_padded_len:].tolist() for ids in gen_ids]
        return generated_ids, prompt_padded_len

    def generate(self, source_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Generate sampled answers for each source row with retry-on-invalid behavior."""

        if not source_rows:
            return [], _build_base_metrics(num_source_rows=0, num_requested=0)

        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        pending = _build_jobs(source_rows, self.cfg.num_samples_per_question)
        accepted: List[Dict[str, Any]] = []
        metrics = _build_base_metrics(num_source_rows=len(source_rows), num_requested=len(pending))

        eos_id = self.tokenizer.eos_token_id

        while pending:
            next_round: List[Dict[str, Any]] = []
            num_batches = int(math.ceil(len(pending) / self.cfg.batch_size))
            for bidx in range(num_batches):
                batch = pending[bidx * self.cfg.batch_size : (bidx + 1) * self.cfg.batch_size]
                generated_ids, _ = self._generate_batch(batch)

                for job, out_ids in zip(batch, generated_ids):
                    token_count_raw = len(out_ids)
                    ended_with_eos = bool(token_count_raw > 0 and eos_id is not None and out_ids[-1] == eos_id)
                    is_truncated = bool(token_count_raw >= self.cfg.max_new_tokens and not ended_with_eos)

                    raw_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                    answer = _sanitize_generated_text(raw_text, self.cfg)
                    answer_tokens = len(self.tokenizer.encode(answer, add_special_tokens=False)) if answer else 0
                    is_empty = not bool(answer)
                    is_overlength = bool(answer_tokens > self.cfg.max_answer_tokens)

                    invalid = is_empty or is_overlength or is_truncated
                    if not invalid:
                        row: Dict[str, Any] = {
                            "id": job["id"],
                            "knowledge": job["knowledge"],
                            "question": job["question"],
                            "sample_id": int(job["sample_id"]),
                            "answer": answer,
                        }
                        if job.get("reference_answer"):
                            row["reference_answer"] = job["reference_answer"]

                        if self.cfg.include_generation_meta:
                            row["generation_meta"] = {
                                "model_name": self.cfg.model_name,
                                "attempt": int(job["attempt"]),
                                "raw_tokens": token_count_raw,
                                "answer_tokens": answer_tokens,
                                "ended_with_eos": ended_with_eos,
                            }
                        accepted.append(row)
                        continue

                    attempt = int(job["attempt"])
                    if attempt + 1 < self.cfg.max_attempts_per_sample:
                        retry_job = dict(job)
                        retry_job["attempt"] = attempt + 1
                        next_round.append(retry_job)
                        metrics["num_retried"] += 1
                    else:
                        if is_empty:
                            metrics["num_drop_empty"] += 1
                        if is_overlength:
                            metrics["num_drop_overlength"] += 1
                        if is_truncated:
                            metrics["num_drop_truncated"] += 1
                        metrics["num_drop_exhausted"] += 1

            pending = next_round

        metrics["num_generated"] = len(accepted)
        return accepted, metrics


class _RetryableApiError(RuntimeError):
    """Transient API failure that should be retried with backoff."""


def _extract_chat_content(content: Any) -> str:
    """Extract plain text from OpenAI-compatible message content payloads."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
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
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


class ApiSelfSampler:
    """Generate sampled answers via an OpenAI-compatible chat-completions API."""

    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, cfg: GenerationConfig, api_key: str) -> None:
        if not api_key:
            raise ValueError("Missing API key for API generation backend.")

        self.cfg = cfg
        self.api_key = api_key
        self._rng = random.Random(cfg.seed)
        self._model_name = (cfg.api_model_name or "").strip() or cfg.model_name
        self._endpoint = f"{cfg.api_base_url.rstrip('/')}/chat/completions"

        self._tokenizer: Optional[AutoTokenizer] = None
        try:
            # Token counting remains compatible with local route when tokenizer is available.
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        except Exception:
            self._tokenizer = None

    def _job_seed(self, job: Dict[str, Any]) -> int:
        payload = f"{job['id']}::{job['sample_id']}::{job['attempt']}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        stable = int(digest[:8], 16)
        return int((self.cfg.seed + stable) % (2**31 - 1))

    def _build_payload(self, prompt: str, seed: int, include_extra_sampling_params: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "seed": seed,
        }

        if include_extra_sampling_params:
            payload["repetition_penalty"] = self.cfg.repetition_penalty
            payload["top_k"] = self.cfg.top_k
            if self.cfg.min_p is not None:
                payload["min_p"] = self.cfg.min_p
        return payload

    async def _post_with_retry(self, client: httpx.AsyncClient, prompt: str, seed: int) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._build_payload(prompt=prompt, seed=seed, include_extra_sampling_params=True)
        using_extra = True

        for retry in range(self.cfg.api_max_retries + 1):
            try:
                resp = await client.post(self._endpoint, headers=headers, json=payload)
                status = int(resp.status_code)
                if status in self._RETRYABLE_STATUS:
                    raise _RetryableApiError(f"retryable status={status}")

                if status >= 400:
                    body = resp.text
                    lower = body.lower()
                    if (
                        status == 400
                        and using_extra
                        and ("unknown" in lower or "invalid" in lower or "unsupported" in lower)
                    ):
                        payload = self._build_payload(prompt=prompt, seed=seed, include_extra_sampling_params=False)
                        using_extra = False
                        continue
                    raise RuntimeError(f"API request failed with status={status}: {body[:300]}")

                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("API response is not a JSON object.")
                return data
            except (_RetryableApiError, httpx.TimeoutException, httpx.NetworkError):
                if retry >= self.cfg.api_max_retries:
                    raise
                base = self.cfg.api_retry_backoff_base * (2**retry)
                sleep_s = min(self.cfg.api_retry_backoff_max, base) * self._rng.uniform(0.8, 1.2)
                await asyncio.sleep(sleep_s)

        raise RuntimeError("API retry loop exited unexpectedly.")

    def _answer_token_count(self, answer: str, completion_tokens: Optional[int]) -> int:
        if not answer:
            return 0
        if self._tokenizer is not None:
            return int(len(self._tokenizer.encode(answer, add_special_tokens=False)))
        if completion_tokens is not None:
            return int(completion_tokens)
        return int(len(answer.split()))

    async def _generate_one(self, client: httpx.AsyncClient, job: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_generation_prompt(job["knowledge"], job["question"])
        seed = self._job_seed(job)
        data = await self._post_with_retry(client=client, prompt=prompt, seed=seed)

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("API response has no choices.")

        choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = str(choice.get("finish_reason") or "")
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        raw_text = _extract_chat_content(message.get("content"))
        answer = _sanitize_generated_text(raw_text, self.cfg)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        completion_tokens_raw: Optional[int] = None
        if isinstance(usage.get("completion_tokens"), int):
            completion_tokens_raw = int(usage["completion_tokens"])

        answer_tokens = self._answer_token_count(answer, completion_tokens_raw)
        is_empty = not bool(answer)
        is_overlength = bool(answer_tokens > self.cfg.max_answer_tokens)
        is_truncated = finish_reason == "length"

        row: Dict[str, Any] = {
            "id": job["id"],
            "knowledge": job["knowledge"],
            "question": job["question"],
            "sample_id": int(job["sample_id"]),
            "answer": answer,
        }
        if job.get("reference_answer"):
            row["reference_answer"] = job["reference_answer"]

        if self.cfg.include_generation_meta:
            row["generation_meta"] = {
                "backend": "api",
                "model_name": self._model_name,
                "attempt": int(job["attempt"]),
                "finish_reason": finish_reason or None,
                "raw_tokens": completion_tokens_raw,
                "answer_tokens": answer_tokens,
                "seed": seed,
            }

        return {
            "row": row,
            "is_empty": is_empty,
            "is_overlength": is_overlength,
            "is_truncated": is_truncated,
        }

    async def _generate_async(self, source_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not source_rows:
            return [], _build_base_metrics(num_source_rows=0, num_requested=0)

        pending_jobs = _build_jobs(source_rows, self.cfg.num_samples_per_question)
        metrics = _build_base_metrics(num_source_rows=len(source_rows), num_requested=len(pending_jobs))
        metrics["num_drop_api_error"] = 0

        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        for job in pending_jobs:
            queue.put_nowait(job)

        accepted: List[Dict[str, Any]] = []
        concurrency = max(1, int(self.cfg.api_max_concurrency))
        limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
        timeout = httpx.Timeout(self.cfg.api_timeout_seconds)

        async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=True) as client:

            async def worker() -> None:
                while True:
                    job = await queue.get()
                    if job is None:
                        queue.task_done()
                        return

                    try:
                        result = await self._generate_one(client=client, job=job)
                    except Exception:
                        attempt = int(job["attempt"])
                        if attempt + 1 < self.cfg.max_attempts_per_sample:
                            retry_job = dict(job)
                            retry_job["attempt"] = attempt + 1
                            metrics["num_retried"] += 1
                            await queue.put(retry_job)
                        else:
                            metrics["num_drop_api_error"] += 1
                            metrics["num_drop_exhausted"] += 1
                        queue.task_done()
                        continue

                    invalid = bool(result["is_empty"] or result["is_overlength"] or result["is_truncated"])
                    if not invalid:
                        accepted.append(result["row"])
                        queue.task_done()
                        continue

                    attempt = int(job["attempt"])
                    if attempt + 1 < self.cfg.max_attempts_per_sample:
                        retry_job = dict(job)
                        retry_job["attempt"] = attempt + 1
                        metrics["num_retried"] += 1
                        await queue.put(retry_job)
                    else:
                        if result["is_empty"]:
                            metrics["num_drop_empty"] += 1
                        if result["is_overlength"]:
                            metrics["num_drop_overlength"] += 1
                        if result["is_truncated"]:
                            metrics["num_drop_truncated"] += 1
                        metrics["num_drop_exhausted"] += 1
                    queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        metrics["num_generated"] = len(accepted)
        return accepted, metrics

    def generate(self, source_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Synchronous wrapper for async API generation."""

        return asyncio.run(self._generate_async(source_rows))
