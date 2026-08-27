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

from utils.io import load_json, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

MAX_ANSWER_ATTEMPTS = 3

# Matches the mandated final line, case-insensitively, at end-of-line.
FINAL_ANSWER_RE = re.compile(
    r"therefore,\s*the\s+answer\s+is:\s*(?P<answer>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PROMPT_CODING = Template(r"""You are an expert software engineer and computer science educator.

You are given a coding problem and its correct solution. Write a step-by-step reasoning
trace that explains exactly how an expert would arrive at this solution from scratch.

Your trace must cover:
1. RESTATE: What is the problem actually asking? Restate the core task in your own words,
   stripping away flavor text and surface details.
2. CONSTRAINTS: What are the key constraints, input ranges, and edge cases that matter?
   Which constraints are load-bearing for the algorithm choice?
3. INSIGHT: What is the core algorithmic insight, trick, or pattern that unlocks this
   problem? This is the thing a failing solver would miss. Be specific — do not say
   "use dynamic programming," say what the recurrence is and why it holds.
4. APPROACH: Walk through the solution step by step. For each step, explain WHY this
   step, not just what it does. Connect every decision back to a constraint or insight.
5. EDGE CASES: Which edge cases does the solution handle, and how? Which edge cases
   would break a naive attempt?
6. FINAL: End with exactly this line: "Therefore, the answer is: ${answer}"

Problem:
${question}

Correct Answer:
${answer}

Write the reasoning trace now. Be specific to this problem, not generic.""")

PROMPT_REASONING = Template(r"""You are an expert mathematician and logician.

You are given a reasoning or math problem and its correct answer. Write a step-by-step
reasoning trace that leads to this answer.

Your trace must cover:
1. RESTATE: What is the problem asking? What quantity or object are we solving for?
   Restate precisely, stripping away narrative framing.
2. GIVEN: What information is provided? What are the constraints? Which ones are
   load-bearing for the solution?
3. INSIGHT: What is the key theorem, lemma, trick, or observation that makes this
   problem tractable? This is the thing a failing solver would miss or get wrong.
   Be specific — name the theorem, state the observation, explain why it applies here.
4. SOLUTION: Work through the solution completely. Show every logical step and explain
   WHY each step follows from the previous. Do not skip steps that seem obvious —
   obvious steps are often where wrong solvers make errors.
5. VERIFY: Does the answer make sense? Perform a sanity check: dimensional analysis,
   boundary case check, or substitution back into the original problem.
6. FINAL: End with exactly this line: "Therefore, the answer is: ${answer}"

Problem:
${question}

Correct Answer:
${answer}

Write the reasoning trace now. Be specific to this problem, not generic.""")


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
    require_file(input_path, f"(pass it via --{domain})")
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
