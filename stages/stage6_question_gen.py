"""Stage 6 — Diagnostic question generation (one per failure mode).

For each (problem, failure_mode) pair, one LLM call writes a new, standalone,
easier diagnostic question that specifically exposes that failure mode.

Coverage cap: before generating, the raw-questions file is checked for existing
questions with the same (concept_involved, failure_type) combination — if 3 or
more already exist, generation is skipped (sufficient coverage exists). The cap
counts questions from previous runs as well as ones generated earlier in this
run (both live in the same output file).

Context enrichment: the concept graph's transitive prerequisites of
concept_involved (up to depth 2) are passed as "related prerequisite concepts".

Resumability: each question is keyed by (source_problem_id, failure_index);
questions already present in the output file are skipped. Saved after every
record.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from prompts.stage6_question_gen import PROMPT_CODING, PROMPT_REASONING
from utils.io import load_json, load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

MAX_PER_CONCEPT_FAILURE_TYPE = 3
MAX_ANCESTOR_DEPTH = 2
DIFFICULTIES = ("beginner", "intermediate", "advanced")


def _ancestors_up_to(
    concept: str, adj: dict[str, list[str]], max_depth: int = MAX_ANCESTOR_DEPTH
) -> list[str]:
    """BFS ancestors of *concept* up to max_depth, in depth order (deduped)."""
    collected: list[str] = []
    seen = {concept}
    frontier = {concept}
    for _ in range(max_depth):
        level: set[str] = set()
        for current in frontier:
            for prereq in adj.get(current, []):
                if prereq not in seen:
                    seen.add(prereq)
                    level.add(prereq)
        if not level:
            break
        collected.extend(sorted(level))
        frontier = level
    return collected


def _why_seems_reasonable(fm: dict) -> str:
    """Best available explanation of why the wrong approach is tempting."""
    for field in ("why_it_seems_reasonable", "attempt_description"):
        value = str(fm.get(field) or "").strip()
        if value:
            return value
    return "(not provided — derive it from the failure description)"


async def run(
    failure_modes_path: Path, graph_path: Path, output_path: Path, domain: str
) -> None:
    """Run Stage 6 for one domain."""
    require_file(
        failure_modes_path,
        f"(run stage 4 first for domain '{domain}')",
    )
    records = load_json(failure_modes_path)
    graph = load_json_obj(graph_path) or {}
    adj: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("from") and edge.get("to"):
            adj.setdefault(str(edge["from"]), set()).add(str(edge["to"]))
    adj = {k: sorted(v) for k, v in adj.items()}

    existing = load_json(output_path)
    by_key: dict[tuple[str, int], dict] = {}
    pair_counts: Counter = Counter()
    for q in existing:
        if q.get("id") is None:
            continue
        key = (str(q.get("source_problem_id")), int(q.get("failure_index", 0)))
        by_key[key] = q
        pair = (str(q.get("concept_involved", "")).lower(), str(q.get("failure_type", "")).upper())
        pair_counts[pair] += 1

    pending: list[tuple[dict, int, dict]] = []
    for record in records:
        for idx, fm in enumerate(record.get("failure_modes", [])):
            key = (str(record.get("problem_id")), idx)
            if key in by_key:
                continue
            pending.append((record, idx, fm))

    prompt = PROMPT_CODING if domain == "coding" else PROMPT_REASONING
    logger.info(
        "Stage 6 [%s]: %d (problem, failure_mode) pair(s), %d already done, %d to process",
        domain,
        sum(len(r.get("failure_modes", [])) for r in records),
        len(by_key),
        len(pending),
    )

    lock = asyncio.Lock()
    counters = {"success": 0, "skip_coverage": 0, "failed": 0}
    pbar = tqdm(
        total=len(pending),
        desc=f"Stage 6 [{domain}] question generation",
        unit="q",
    )
    pbar.set_postfix(skip=len(by_key))

    async def worker(record: dict, idx: int, fm: dict) -> None:
        pid = str(record.get("problem_id"))
        concept = str(fm.get("concept_involved", "")).strip().lower()
        ftype = str(fm.get("failure_type", "")).strip().upper()
        pair = (concept, ftype)

        # Atomically reserve a coverage slot so the cap holds even under
        # concurrency; the slot is released if generation fails.
        async with lock:
            if pair_counts[pair] >= MAX_PER_CONCEPT_FAILURE_TYPE:
                covered = True
                seen_count = pair_counts[pair]
            else:
                covered = False
                pair_counts[pair] += 1
                seen_count = pair_counts[pair]
        if covered:
            logger.info(
                "Stage 6 [%s] SKIP problem %s failure_index %d: %d question(s) already "
                "exist for %s + %s (coverage cap reached)",
                domain, pid, idx, seen_count, concept, ftype,
            )
            counters["skip_coverage"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(by_key), **counters)
            return

        ancestors = _ancestors_up_to(concept, adj)
        prompt_text = prompt.safe_substitute(
            question=str(record.get("question", "")),
            failure_type=ftype,
            description=str(fm.get("description", "")),
            concept_involved=concept,
            what_correct_understanding_looks_like=str(
                fm.get("what_correct_understanding_looks_like", "")
            ),
            why_it_seems_reasonable=_why_seems_reasonable(fm),
            ancestor_concepts=(
                "\n  ".join(f"- {a}" for a in ancestors) if ancestors else "(none)"
            ),
        )
        async def release_slot() -> None:
            """Give back the reserved coverage slot (generation failed)."""
            async with lock:
                pair_counts[pair] = max(0, pair_counts[pair] - 1)

        try:
            data = await call_llm(prompt_text, expect_json=True)
        except LLMError as exc:
            logger.error(
                "Stage 6 [%s] DISCARD problem %s failure_index %d: LLM call failed: %s",
                domain, pid, idx, exc,
            )
            await release_slot()
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(by_key), **counters)
            return
        if not isinstance(data, dict) or not str(data.get("question", "")).strip():
            logger.error(
                "Stage 6 [%s] DISCARD problem %s failure_index %d: response missing "
                "'question' text: %r",
                domain, pid, idx, str(data)[:200],
            )
            await release_slot()
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(by_key), **counters)
            return

        difficulty = str(data.get("difficulty", "")).strip().lower()
        if difficulty not in DIFFICULTIES:
            difficulty = "intermediate"

        question_record = {
            "id": str(uuid.uuid4()),
            "source_problem_id": record.get("problem_id"),
            "source_benchmark": record.get("benchmark"),
            "source_sub_benchmark": record.get("sub_benchmark"),
            "domain": domain,
            "failure_index": idx,
            "failure_type": ftype,
            "failure_source": str(fm.get("source", "reasoning_anchored")).lower(),
            "failure_severity": str(fm.get("severity", "major")).strip().lower(),
            "failure_description": str(fm.get("description", "")),
            "what_correct_understanding_looks_like": str(
                fm.get("what_correct_understanding_looks_like", "")
            ),
            "concept_involved": concept,
            "question": str(data.get("question", "")).strip(),
            "what_it_tests": str(data.get("what_it_tests", "")).strip(),
            "trap": str(data.get("trap", "")).strip(),
            "why_trap_is_tempting": str(data.get("why_trap_is_tempting", "")).strip(),
            "difficulty": difficulty,
            "tags": [
                str(t).strip()
                for t in (data.get("tags") or [])
                if isinstance(t, (str, int, float)) and str(t).strip()
            ],
        }
        if domain == "reasoning":
            question_record["answer"] = str(data.get("answer", "")).strip()
            question_record["answer_explanation"] = str(
                data.get("answer_explanation", "")
            ).strip()

        async with lock:
            by_key[(pid, idx)] = question_record
            # (the coverage slot was reserved before generation; keep the count)
            save_json(output_path, list(by_key.values()))
            counters["success"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(by_key), **counters)
        logger.debug(
            "Stage 6 [%s] generated question %s for problem %s failure_index %d "
            "(%s + %s now has %d)",
            domain, question_record["id"], pid, idx, concept, ftype, pair_counts[pair],
        )

    if pending:
        await asyncio.gather(*(worker(*item) for item in pending))
    pbar.close()
    if not output_path.exists():
        save_json(output_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 6 [%s] complete: %d question(s) in raw output", domain, len(by_key)
    )
