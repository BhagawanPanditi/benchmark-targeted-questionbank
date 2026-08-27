"""Stage 1 — Reasoning generation (grounded to the gold answer).

For each source problem, generate a step-by-step reasoning trace that an expert
would use to reach the gold answer. The final answer stated in the trace is
extracted and compared to the gold ``answer`` field; on mismatch the generation
is retried (up to MAX_ANSWER_ATTEMPTS total attempts). Records that still do
not match are kept with ``reasoning_status="failed"`` and are skipped by all
downstream stages.

Resumability: records whose ``problem_id`` already exists in the output file are
skipped. The output file is re-saved after every completed record.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from string import Template

from tqdm import tqdm

from prompts.stage1_reasoning import PROMPT_CODING, PROMPT_REASONING
from utils.io import load_json, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

MAX_ANSWER_ATTEMPTS = 3

# Matches the mandated final line, case-insensitively, at end-of-line.
FINAL_ANSWER_RE = re.compile(
    r"therefore,\s*the\s+answer\s+is:\s*(?P<answer>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison: case, whitespace, punctuation."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t.;,:\"'`")


def extract_final_answer(reasoning: str) -> str | None:
    """Return the last "Therefore, the answer is: ..." line, or None."""
    matches = FINAL_ANSWER_RE.findall(str(reasoning))
    if not matches:
        return None
    return matches[-1].strip()


def _base_record(problem: dict) -> dict:
    return {
        "benchmark": problem.get("benchmark"),
        "sub_benchmark": problem.get("sub_benchmark"),
        "problem_id": problem.get("problem_id"),
        "question": problem.get("question"),
        "answer": problem.get("answer"),
    }


async def _generate_one(problem: dict, prompt: Template) -> dict:
    """Generate a grounded reasoning trace for one problem (with answer check)."""
    pid = str(problem.get("problem_id"))
    question = str(problem.get("question", ""))
    gold = str(problem.get("answer", ""))
    record = _base_record(problem)
    prompt_text = prompt.safe_substitute(question=question, answer=gold)

    last_text = ""
    for attempt in range(1, MAX_ANSWER_ATTEMPTS + 1):
        try:
            text = await call_llm(prompt_text, expect_json=False)
        except LLMError as exc:
            logger.error(
                "Stage 1 [%s] problem %s: LLM call failed (attempt %d/%d): %s",
                record.get("benchmark"), pid, attempt, MAX_ANSWER_ATTEMPTS, exc,
            )
            break
        last_text = text
        stated = extract_final_answer(text)
        if stated is None:
            logger.warning(
                "Stage 1 [%s] problem %s: no 'Therefore, the answer is:' line found "
                "(attempt %d/%d)",
                record.get("benchmark"), pid, attempt, MAX_ANSWER_ATTEMPTS,
            )
        elif normalize_answer(stated) == normalize_answer(gold):
            record["reasoning"] = text
            record["reasoning_status"] = "ok"
            return record
        else:
            logger.warning(
                "Stage 1 [%s] problem %s: stated answer %r does not match gold %r "
                "(attempt %d/%d)",
                record.get("benchmark"), pid, stated, gold, attempt, MAX_ANSWER_ATTEMPTS,
            )

    record["reasoning"] = last_text
    record["reasoning_status"] = "failed"
    logger.info(
        "Stage 1 [%s] problem %s: marked reasoning_status=failed after %d attempts",
        record.get("benchmark"), pid, MAX_ANSWER_ATTEMPTS,
    )
    return record


async def run(input_path: Path, output_path: Path, domain: str) -> None:
    """Run Stage 1 for one domain."""
    require_file(input_path, f"(place {domain}.json in the project root)")
    problems = load_json(input_path)
    if not problems:
        logger.warning("Stage 1 [%s]: input %s contains no records", domain, input_path)

    existing = load_json(output_path)
    by_id: dict[str, dict] = {
        str(r["problem_id"]): r
        for r in existing
        if r.get("problem_id") is not None
    }
    existing_ids = set(by_id)
    pending = [
        p for p in problems
        if str(p.get("problem_id")) not in existing_ids
    ]
    prompt = PROMPT_CODING if domain == "coding" else PROMPT_REASONING

    logger.info(
        "Stage 1 [%s]: %d problem(s), %d already done, %d to process",
        domain, len(problems), len(existing_ids), len(pending),
    )

    lock = asyncio.Lock()
    counters = {"ok": 0, "failed": 0}
    pbar = tqdm(
        total=len(pending),
        desc=f"Stage 1 [{domain}] reasoning",
        unit="prob",
    )
    pbar.set_postfix(skip=len(existing_ids))

    async def worker(problem: dict) -> None:
        result = await _generate_one(problem, prompt)
        pid = str(problem.get("problem_id"))
        async with lock:
            by_id[pid] = result
            save_json(output_path, list(by_id.values()))
            counters["ok" if result["reasoning_status"] == "ok" else "failed"] += 1
            pbar.update(1)
            pbar.set_postfix(
                skip=len(existing_ids),
                ok=counters["ok"],
                failed=counters["failed"],
            )

    if pending:
        await asyncio.gather(*(worker(p) for p in pending))
    pbar.close()
    if not output_path.exists():
        save_json(output_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 1 [%s] complete: %d record(s) in output (%d ok, %d failed)",
        domain, len(by_id),
        sum(1 for r in by_id.values() if r.get("reasoning_status") == "ok"),
        sum(1 for r in by_id.values() if r.get("reasoning_status") == "failed"),
    )
