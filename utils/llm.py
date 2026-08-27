"""Async LLM client with semaphore-bounded concurrency, retries, and JSON parsing.

All pipeline LLM traffic goes through :func:`call_llm`. It:
  * uses the OpenAI Python SDK pointed at the vLLM-compatible endpoint in config.py
  * bounds in-flight requests with asyncio.Semaphore(config.MAX_CONCURRENT)
  * retries up to config.RETRY_ATTEMPTS times on any exception with
    exponential backoff (1s, 2s, 4s, ...)
  * enforces config.TIMEOUT_SECONDS per call
  * parses JSON responses (stripping markdown code fences) when expect_json=True
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call still fails after all retry attempts."""


_client: AsyncOpenAI | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None
_semaphore_capacity: int | None = None


def set_concurrency(concurrency: int) -> None:
    """Update the global concurrency cap and reset the semaphore."""
    global _semaphore, _semaphore_capacity
    config.MAX_CONCURRENT = concurrency
    _semaphore = None
    _semaphore_capacity = None


def get_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client (created lazily)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.TIMEOUT_SECONDS,
        )
    return _client


def get_semaphore() -> asyncio.Semaphore:
    """Return the concurrency semaphore, recreated if the event loop or cap changed.

    Stages are each driven by their own asyncio.run() invocation, so a semaphore
    bound to a previous (closed) loop would raise on acquire.
    """
    global _semaphore, _semaphore_loop, _semaphore_capacity
    running = asyncio.get_running_loop()
    if (
        _semaphore is None
        or _semaphore_loop is not running
        or _semaphore_capacity != config.MAX_CONCURRENT
    ):
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
        _semaphore_loop = running
        _semaphore_capacity = config.MAX_CONCURRENT
    return _semaphore


# ```lang\n ... \n```  (with or without a language tag)
_FENCED_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove a single wrapping markdown code fence, if present."""
    t = text.strip()
    match = _FENCED_RE.search(t)
    if match:
        return match.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*[ \t]*\r?", "", t)
    if t.endswith("```"):
        t = re.sub(r"\r?[ \t]*```$", "", t)
    return t.strip()


def extract_json(text: str) -> Any:
    """Parse a JSON document out of an LLM response.

    Order of attempts:
      1. parse the (code-fence-stripped) text as-is
      2. slice from the first '{' or '[' through the matching last '}' or ']'
    Raises ValueError when no valid JSON can be recovered.
    """
    if text is None:
        raise ValueError("empty LLM response")
    candidate = strip_code_fences(str(text))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON structure found in LLM response: {text[:200]!r}")
    start = min(starts)
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if end <= start:
        raise ValueError(f"unbalanced JSON structure in LLM response: {text[:200]!r}")
    snippet = candidate[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from LLM ({exc}): {snippet[:200]!r}") from exc


async def call_llm(prompt: str, expect_json: bool = True) -> dict | list | str:
    """Call the LLM (with retries) under the global concurrency cap.

    Makes one initial attempt plus up to config.RETRY_ATTEMPTS retries, waiting
    1s, 2s, 4s (exponential backoff) between attempts, retrying on any
    exception (including invalid JSON responses).

    Returns the parsed JSON document (dict or list) when expect_json is True,
    otherwise the raw response text. Raises LLMError when everything fails.
    """
    client = get_client()
    semaphore = get_semaphore()
    total_attempts = config.RETRY_ATTEMPTS + 1  # initial attempt + retries
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=config.TIMEOUT_SECONDS,
                )
                if not response.choices:
                    raise ValueError("LLM response contained no choices")
                content = response.choices[0].message.content
                if content is None or not str(content).strip():
                    raise ValueError("LLM returned empty content")
                if expect_json:
                    return extract_json(content)
                return str(content)
            except Exception as exc:  # noqa: BLE001 - spec: retry on any exception
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    total_attempts,
                    exc,
                )
        if attempt < total_attempts:
            wait_seconds = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
            logger.debug("retrying LLM call in %ds", wait_seconds)
            await asyncio.sleep(wait_seconds)

    raise LLMError(
        f"LLM call failed after {total_attempts} attempts "
        f"(1 + {config.RETRY_ATTEMPTS} retries): {last_error}"
    ) from last_error
