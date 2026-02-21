from __future__ import annotations

import asyncio
import hashlib
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from .answer_filter import apply_answer_filters
from .config import AnswerFilterConfig, GenerationConfig, HardAugmentConfig
from .generation import ApiSelfSampler, SelfSampler
from nli_judge.judge import AnswerJudge
from .pairing import summarize_question_groups
from prompt import build_repair_prompt
from .text import normalize_whitespace


def _support_score(row: Dict[str, Any]) -> float:
    judge = row.get("judge") or {}
    return float(judge.get("candidate_entail_primary", 0.0))


def _hard_question_views(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups, _ = summarize_question_groups(rows)
    return [g for g in groups if str(g.get("group_label")) == "hard"]


def _clean_repair_output(text: str) -> str:
    out = (text or "").replace("```", "\n").strip()
    marker = "revised answer:"
    lower = out.lower()
    if marker in lower:
        idx = lower.rfind(marker)
        out = out[idx + len(marker) :].strip()

    for line in out.splitlines():
        line = line.strip()
        if line:
            return normalize_whitespace(line)
    return ""


def _extract_chat_content(content: Any) -> str:
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
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _resample_generation_cfg(cfg: HardAugmentConfig) -> GenerationConfig:
    return GenerationConfig(
        backend=cfg.resample_backend,
        model_name=cfg.resample_model_name,
        device=cfg.resample_device,
        batch_size=cfg.resample_batch_size,
        max_new_tokens=cfg.resample_max_new_tokens,
        max_answer_tokens=cfg.resample_max_new_tokens,
        max_attempts_per_sample=3,
        num_samples_per_question=cfg.extra_samples_per_hard,
        temperature=cfg.resample_temperature,
        top_p=cfg.resample_top_p,
        top_k=cfg.resample_top_k,
        min_p=cfg.resample_min_p,
        repetition_penalty=cfg.resample_repetition_penalty,
        use_chat_template=False,
        enable_thinking=False,
        strip_think_tags=True,
        strip_role_markers=True,
        strip_markdown_fences=True,
        extract_short_answer=True,
        include_generation_meta=False,
        api_model_name=cfg.resample_api_model_name,
        api_base_url=cfg.resample_api_base_url,
        api_key_env=cfg.resample_api_key_env,
        api_timeout_seconds=cfg.resample_api_timeout_seconds,
        api_max_concurrency=cfg.resample_api_max_concurrency,
        api_max_retries=cfg.resample_api_max_retries,
        api_retry_backoff_base=cfg.resample_api_retry_backoff_base,
        api_retry_backoff_max=cfg.resample_api_retry_backoff_max,
        seed=42,
    )


def resample_hard_answers(
    rows: Sequence[Dict[str, Any]],
    hard_cfg: HardAugmentConfig,
    answer_filter_cfg: AnswerFilterConfig,
    judge: AnswerJudge,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate extra answers for hard questions and keep only filtered judged rows."""

    metrics: Dict[str, Any] = {
        "num_hard_questions": 0,
        "num_generated_rows": 0,
        "num_judged_rows": 0,
        "num_kept_rows": 0,
    }
    if hard_cfg.extra_samples_per_hard <= 0:
        metrics["skipped"] = "extra_samples_per_hard<=0"
        return [], metrics

    hard_views = _hard_question_views(rows)
    metrics["num_hard_questions"] = len(hard_views)
    if not hard_views:
        return [], metrics

    source_rows: List[Dict[str, Any]] = []
    for view in hard_views:
        row = {
            "id": view["id"],
            "knowledge": view["knowledge"],
            "question": view["question"],
        }
        ref = str(view.get("reference_answer") or "").strip()
        if ref:
            row["reference_answer"] = ref
        source_rows.append(row)

    gen_cfg = _resample_generation_cfg(hard_cfg)
    if gen_cfg.backend == "api":
        api_key = os.getenv(hard_cfg.resample_api_key_env, "").strip()
        if not api_key:
            metrics["skipped"] = f"missing_env:{hard_cfg.resample_api_key_env}"
            return [], metrics
        sampler = ApiSelfSampler(cfg=gen_cfg, api_key=api_key)
    else:
        sampler = SelfSampler(gen_cfg)

    generated, gen_metrics = sampler.generate(source_rows)
    metrics["generation_metrics"] = gen_metrics
    metrics["num_generated_rows"] = len(generated)
    if not generated:
        return [], metrics

    for rec in generated:
        rec["sample_id"] = int(rec.get("sample_id", -1)) + 100000

    judged_rows, judge_metrics = judge.judge(generated)
    metrics["judge_metrics"] = judge_metrics
    metrics["num_judged_rows"] = len(judged_rows)

    filtered_rows, filter_metrics = apply_answer_filters(judged_rows, answer_filter_cfg)
    metrics["answer_filter_metrics"] = filter_metrics
    metrics["num_kept_rows"] = len(filtered_rows)
    return filtered_rows, metrics


class _RetryableApiError(RuntimeError):
    """Transient HTTP failure for strong-model repair calls."""


def _job_seed(row_id: str, attempt: int) -> int:
    payload = f"{row_id}::{attempt}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


async def _post_repair_with_retry(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    retries: int,
    backoff_base: float,
    backoff_max: float,
    rng: random.Random,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for retry in range(retries + 1):
        try:
            resp = await client.post(endpoint, headers=headers, json=payload)
            status = int(resp.status_code)
            if status in {408, 429, 500, 502, 503, 504}:
                raise _RetryableApiError(f"status={status}")
            if status >= 400:
                raise RuntimeError(f"API request failed with status={status}: {resp.text[:400]}")
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("API response is not a JSON object.")
            return data
        except (_RetryableApiError, httpx.TimeoutException, httpx.NetworkError):
            if retry >= retries:
                raise
            base = backoff_base * (2**retry)
            sleep_s = min(backoff_max, base) * rng.uniform(0.8, 1.2)
            await asyncio.sleep(sleep_s)

    raise RuntimeError("Repair API retry loop exited unexpectedly.")


def _repair_payload(
    *,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }


async def _run_repair_api(
    jobs: Sequence[Dict[str, Any]],
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_concurrency: int,
    retries: int,
    backoff_base: float,
    backoff_max: float,
    seed: int,
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    limits = httpx.Limits(max_connections=max_concurrency * 2, max_keepalive_connections=max_concurrency)
    timeout = httpx.Timeout(timeout_seconds)
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(dict(job))

    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=True) as client:

        async def worker() -> None:
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return

                try:
                    payload = _repair_payload(
                        model_name=model_name,
                        prompt=job["prompt"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=_job_seed(str(job["id"]), int(job["attempt"])),
                    )
                    data = await _post_repair_with_retry(
                        client,
                        endpoint=endpoint,
                        api_key=api_key,
                        payload=payload,
                        retries=retries,
                        backoff_base=backoff_base,
                        backoff_max=backoff_max,
                        rng=rng,
                    )
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not isinstance(choices, list) or not choices:
                        raise RuntimeError("Repair API response has no choices.")
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                    raw_text = _extract_chat_content(message.get("content"))
                    clean = _clean_repair_output(raw_text)
                    if clean:
                        out.append(
                            {
                                "id": job["id"],
                                "knowledge": job["knowledge"],
                                "question": job["question"],
                                "reference_answer": job.get("reference_answer"),
                                "sample_id": int(job["sample_id"]),
                                "answer": clean,
                                "generation_meta": {
                                    "backend": "api_repair",
                                    "model_name": model_name,
                                    "attempt": int(job["attempt"]),
                                },
                            }
                        )
                except Exception:
                    pass
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(max(1, max_concurrency))]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

    return out


def repair_hard_with_api(
    rows: Sequence[Dict[str, Any]],
    hard_cfg: HardAugmentConfig,
    answer_filter_cfg: AnswerFilterConfig,
    judge: AnswerJudge,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair hard-question negatives with a stronger API model and keep correct additions."""

    metrics: Dict[str, Any] = {
        "num_hard_questions": 0,
        "num_repair_jobs": 0,
        "num_api_candidates": 0,
        "num_judged_rows": 0,
        "num_kept_rows": 0,
        "num_positive_kept": 0,
    }
    if not hard_cfg.enable_api_repair:
        metrics["skipped"] = "enable_api_repair=false"
        return [], metrics

    api_key = os.getenv(hard_cfg.repair_api_key_env, "").strip()
    if not api_key:
        metrics["skipped"] = f"missing_env:{hard_cfg.repair_api_key_env}"
        return [], metrics

    hard_views = _hard_question_views(rows)
    metrics["num_hard_questions"] = len(hard_views)
    if not hard_views:
        return [], metrics

    jobs: List[Dict[str, Any]] = []
    for view in hard_views:
        negatives = list(view.get("negatives") or [])
        if not negatives:
            continue
        seed_negative = max(negatives, key=_support_score)
        rejected_answer = str(seed_negative.get("answer") or "").strip()
        if not rejected_answer:
            continue
        reference_answer = str(view.get("reference_answer") or "").strip() or None
        for attempt in range(hard_cfg.repair_attempts_per_question):
            prompt = build_repair_prompt(
                knowledge=view["knowledge"],
                question=view["question"],
                rejected_answer=rejected_answer,
                reference_answer=reference_answer,
            )
            jobs.append(
                {
                    "id": view["id"],
                    "knowledge": view["knowledge"],
                    "question": view["question"],
                    "reference_answer": reference_answer,
                    "attempt": attempt,
                    "sample_id": 200000 + attempt,
                    "prompt": prompt,
                }
            )

    metrics["num_repair_jobs"] = len(jobs)
    if not jobs:
        return [], metrics

    endpoint = f"{hard_cfg.repair_api_base_url.rstrip('/')}/chat/completions"
    api_candidates = asyncio.run(
        _run_repair_api(
            jobs=jobs,
            endpoint=endpoint,
            api_key=api_key,
            model_name=hard_cfg.repair_api_model_name,
            max_new_tokens=hard_cfg.repair_max_new_tokens,
            temperature=hard_cfg.repair_temperature,
            top_p=hard_cfg.repair_top_p,
            max_concurrency=hard_cfg.repair_api_max_concurrency,
            retries=hard_cfg.repair_api_max_retries,
            backoff_base=hard_cfg.repair_api_retry_backoff_base,
            backoff_max=hard_cfg.repair_api_retry_backoff_max,
            seed=42,
            timeout_seconds=hard_cfg.repair_api_timeout_seconds,
        )
    )
    metrics["num_api_candidates"] = len(api_candidates)
    if not api_candidates:
        return [], metrics

    judged_rows, judge_metrics = judge.judge(api_candidates)
    metrics["judge_metrics"] = judge_metrics
    metrics["num_judged_rows"] = len(judged_rows)

    filtered_rows, filter_metrics = apply_answer_filters(judged_rows, answer_filter_cfg)
    metrics["answer_filter_metrics"] = filter_metrics
    metrics["num_kept_rows"] = len(filtered_rows)

    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in filtered_rows:
        if not bool((row.get("judge") or {}).get("is_correct", False)):
            continue
        by_id.setdefault(str(row["id"]), []).append(row)

    selected: List[Dict[str, Any]] = []
    for rid, candidates in by_id.items():
        best = max(candidates, key=_support_score)
        best = dict(best)
        best["sample_id"] = 299999
        selected.append(best)

    metrics["num_positive_kept"] = len(selected)
    return selected, metrics
