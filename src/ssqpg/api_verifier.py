#!/usr/bin/env python3
"""
Minimal health check for Alibaba Cloud Model Studio (DashScope) OpenAI-compatible API.

- Verifies: network reachability, API key validity, and model availability.
- Prints: HTTP status, response snippet, and token usage if present.

Requires: Python 3.11+ (standard library only)
Env: DASHSCOPE_API_KEY
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3-0.6b"


def main() -> int:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 2

    payload = {
        "model": MODEL,
        "enable_thinking":False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Which city is the capital of France?"},
        ],
        "max_tokens": 128,
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}")
            print("Response (first 800 chars):")
            print(body[:800])

            # Try to parse usage info
            try:
                obj = json.loads(body)
                usage = obj.get("usage")
                if usage is not None:
                    print("Usage:")
                    print(json.dumps(usage, ensure_ascii=False))
                else:
                    print("Usage: not found in response.")
            except json.JSONDecodeError:
                print("WARNING: Response is not valid JSON.")
            return 0

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"HTTP ERROR {e.code}", file=sys.stderr)
        print(err_body[:1200], file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
