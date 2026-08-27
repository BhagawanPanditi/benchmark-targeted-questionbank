"""Stage 2 — Raw concept extraction (per problem, freeform).

For each problem with reasoning_status="ok", extract freeform dot-notation
concept tags. Normalization happens in Stage 3; this stage runs first so the
controlled vocabulary exists before Stage 4 tags failure modes against it.

Resumability: problem_ids already present in the output file are skipped; the
file is re-saved after every completed record.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from string import Template

from tqdm import tqdm

from utils.io import load_json, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

PROMPT_CODING = Template(r"""You are a programming education expert.

Analyze this problem, its reasoning trace, and its answer. Extract every distinct
concept, technique, data structure, algorithm, or pattern that a solver MUST know
or apply to solve this problem correctly.

Include concepts at ALL levels:
  - The primary algorithm or technique (e.g., algorithms.technique.two-pointer)
  - Data structures used (e.g., data-structures.mapping.dictionary)
  - Mathematical tools required (e.g., math.modular-arithmetic.modular-inverse)
  - Implicit prerequisites the solution relies on without stating them explicitly
    (e.g., if the solution uses binary search, include it even if not named)

Return concepts in taxonomical dot-notation: category.subcategory.specific-concept

Rules:
  - Each concept must be a reusable, teachable skill — not specific to this one problem
  - Use lowercase-with-hyphens within levels, dots between levels
  - Aim for 3-8 concepts per problem
  - Top-level categories should be chosen from:
    algorithms, data-structures, functionality, analytics, graph, dp, string, math

Examples:
  algorithms.technique.two-pointer
  data-structures.mapping.dictionary
  algorithms.sorting.merge-sort
  dp.technique.memoization
  math.modular-arithmetic.modular-inverse
  graph.traversal.bfs
  string.matching.kmp

Problem:
${question}

Reasoning Trace:
${reasoning}

Answer:
${answer}

Return ONLY valid JSON:
{"raw_concepts": ["category.subcategory.concept", ...]}""")

PROMPT_REASONING = Template(r"""You are a mathematics and logic education expert.

Analyze this problem, its reasoning trace, and its answer. Extract every distinct
concept, theorem, technique, or mathematical tool that a solver MUST know or apply
to solve this problem correctly.

Include concepts at ALL levels:
  - The primary method or theorem (e.g., combinatorics.counting.inclusion-exclusion)
  - Supporting mathematical tools (e.g., number-theory.divisibility.prime-factorization)
  - Logical structures used (e.g., logic.proof-technique.contradiction)
  - Implicit prerequisites the solution relies on without naming them

Return concepts in taxonomical dot-notation: category.subcategory.specific-concept

Rules:
  - Each concept must be a reusable, teachable skill — not specific to this one problem
  - Use lowercase-with-hyphens within levels, dots between levels
  - Aim for 3-8 concepts per problem
  - Top-level categories should be chosen from:
    algebra, combinatorics, number-theory, geometry, logic, probability, calculus,
    proof-technique

Examples:
  combinatorics.counting.inclusion-exclusion
  number-theory.divisibility.prime-factorization
  logic.constraint-satisfaction.backtracking
  algebra.inequalities.am-gm
  probability.expectation.linearity-of-expectation
  proof-technique.induction.strong-induction

Problem:
${question}

Reasoning Trace:
${reasoning}

Answer:
${answer}

Return ONLY valid JSON:
{"raw_concepts": ["category.subcategory.concept", ...]}""")


def _coerce_raw_concepts(data: object) -> list[str] | None:
    """Extract a clean list of concept strings from the LLM response, if valid."""
    if isinstance(data, dict):
        data = data.get("raw_concepts")
    if not isinstance(data, list):
        return None
    concepts: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            continue
        concept = item.strip().lower()
        if concept and concept not in seen:
            seen.add(concept)
            concepts.append(concept)
    return concepts or None


async def run(input_path: Path, output_path: Path, domain: str) -> None:
    """Run Stage 2 for one domain."""
    require_file(
        input_path,
        f"(run stage 1 first for domain '{domain}')",
    )
    records = load_json(input_path)
    ok_records = [r for r in records if r.get("reasoning_status") == "ok"]
    skipped_failed = len(records) - len(ok_records)

    existing = load_json(output_path)
    by_id: dict[str, dict] = {
        str(r["problem_id"]): r
        for r in existing
        if r.get("problem_id") is not None
    }
    existing_ids = set(by_id)
    pending = [
        r for r in ok_records if str(r.get("problem_id")) not in existing_ids
    ]
    prompt = PROMPT_CODING if domain == "coding" else PROMPT_REASONING

    logger.info(
        "Stage 2 [%s]: %d ok record(s), %d already done, %d to process, "
        "%d failed-record(s) excluded",
        domain, len(ok_records), len(existing_ids), len(pending), skipped_failed,
    )

    lock = asyncio.Lock()
    counters = {"success": 0, "failed": 0}
    pbar = tqdm(
        total=len(pending),
        desc=f"Stage 2 [{domain}] raw concepts",
        unit="prob",
    )
    pbar.set_postfix(skip=len(existing_ids))

    async def worker(record: dict) -> None:
        pid = str(record.get("problem_id"))
        prompt_text = prompt.safe_substitute(
            question=str(record.get("question", "")),
            reasoning=str(record.get("reasoning", "")),
            answer=str(record.get("answer", "")),
        )
        try:
            data = await call_llm(prompt_text, expect_json=True)
        except LLMError as exc:
            logger.error(
                "Stage 2 [%s] DISCARD problem %s: LLM call failed: %s", domain, pid, exc
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return
        concepts = _coerce_raw_concepts(data)
        if concepts is None:
            logger.error(
                "Stage 2 [%s] DISCARD problem %s: response missing a valid "
                "'raw_concepts' string list: %r",
                domain, pid, str(data)[:200],
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return
        out = {
            "benchmark": record.get("benchmark"),
            "sub_benchmark": record.get("sub_benchmark"),
            "problem_id": record.get("problem_id"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "reasoning": record.get("reasoning"),
            "reasoning_status": record.get("reasoning_status"),
            "raw_concepts": concepts,
        }
        async with lock:
            by_id[pid] = out
            save_json(output_path, list(by_id.values()))
            counters["success"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
        logger.debug("Stage 2 [%s] problem %s: %d raw concept(s)", domain, pid, len(concepts))

    if pending:
        await asyncio.gather(*(worker(r) for r in pending))
    pbar.close()
    if not output_path.exists():
        save_json(output_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 2 [%s] complete: %d record(s) with raw concepts in output",
        domain, len(by_id),
    )
