"""Stage 4 — Failure mode extraction (per problem, TWO-PASS).

Pass A (reasoning-anchored): given the correct reasoning trace, identify what can
go wrong ALONG the correct solution path.

Pass B (anticipatory, wrong-solver simulation): WITHOUT the reasoning trace,
simulate plausible-but-wrong first attempts. This is the pass that catches
failures that never appear on the correct path at all (wrong frame, wrong
algorithm, pattern overfitting) — critical for hard benchmarks.

Both passes are tagged with the Stage 3 normalized vocabulary for
``concept_involved``. Results are merged and deduplicated on
(failure_type, concept_involved): higher severity wins; on equal severity the
anticipatory (Pass B) entry wins. New concepts (is_new_concept=true) are
registered into the taxonomy file so Stage 5's graph includes them.

Resumability: problem_ids already present in the output file are skipped.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompts.stage4_failure_modes import PROMPT_PASS_A, PROMPT_PASS_B
from utils.constants import SEVERITY_RANK, VALID_FAILURE_TYPES
from utils.io import load_json, load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm, set_concurrency

logger = logging.getLogger(__name__)

REQUIRED_FM_FIELDS = (
    "failure_type",
    "description",
    "concept_involved",
    "what_correct_understanding_looks_like",
)


def _sanitize(entries: Any, source: str) -> list[dict]:
    """Validate/normalize raw failure-mode entries from one pass."""
    if not isinstance(entries, list):
        if entries is not None:
            logger.debug("failure mode response was not a list: %r", str(entries)[:200])
        return []
    clean: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if any(not str(entry.get(field, "")).strip() for field in REQUIRED_FM_FIELDS):
            logger.debug("dropping malformed failure mode entry: %r", str(entry)[:200])
            continue
        fm = dict(entry)
        raw_type = str(fm["failure_type"]).strip().upper().replace(" ", "_").replace("-", "_")

        # Allow valid enum types; for OTHER or novel types, support proposed_new_type
        if raw_type == "OTHER" or raw_type not in VALID_FAILURE_TYPES:
            proposed = str(fm.get("proposed_new_type") or "").strip().upper().replace(" ", "_").replace("-", "_")
            if proposed and re.match(r"^[A-Z0-9_]+$", proposed):
                fm["failure_type"] = proposed
                fm["is_new_failure_type"] = True
                logger.info("Stage 4: discovered proposed new failure type: %s", proposed)
            elif re.match(r"^[A-Z0-9_]+$", raw_type) and raw_type != "OTHER":
                fm["failure_type"] = raw_type
                fm["is_new_failure_type"] = True
                logger.info("Stage 4: using non-standard failure type: %s", raw_type)
            else:
                fm["failure_type"] = "OTHER"
        else:
            fm["failure_type"] = raw_type

        fm["concept_involved"] = str(fm["concept_involved"]).strip().lower()
        severity = str(fm.get("severity", "major")).strip().lower()
        fm["severity"] = severity if severity in SEVERITY_RANK else "major"
        fm["is_new_concept"] = bool(fm.get("is_new_concept", False))
        fm["source"] = source
        clean.append(fm)
    return clean


def _merge(pass_a: list[dict], pass_b: list[dict]) -> list[dict]:
    """Merge both passes, deduplicating on (failure_type, concept_involved).

    Higher severity wins; on equal severity the anticipatory (Pass B) entry wins
    (anticipatory failures are harder to generate and more valuable).
    """
    best: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for fm in [*pass_a, *pass_b]:
        key = (fm["failure_type"], fm["concept_involved"])
        current = best.get(key)
        if current is None:
            best[key] = fm
            order.append(key)
            continue
        current_rank = SEVERITY_RANK[current["severity"]]
        challenger_rank = SEVERITY_RANK[fm["severity"]]
        if challenger_rank > current_rank:
            best[key] = fm
        elif challenger_rank == current_rank and fm["source"] == "anticipatory":
            best[key] = fm
    return [best[key] for key in order]


async def run(
    reasoning_path: Path,
    taxonomy_path: Path,
    output_path: Path,
    domain: str,
    concurrency: int | None = None,
) -> None:
    """Run Stage 4 (two-pass) for one domain."""
    if concurrency is not None:
        set_concurrency(concurrency)
    require_file(
        reasoning_path,
        f"(run stage 1 first for domain '{domain}')",
    )
    require_file(
        taxonomy_path,
        f"(run stage 3 first for domain '{domain}')",
    )
    records = load_json(reasoning_path)
    ok_records = [r for r in records if r.get("reasoning_status") == "ok"]

    taxonomy_doc = load_json_obj(taxonomy_path) or {}
    state: dict[str, Any] = {
        "taxonomy": [str(c).strip() for c in taxonomy_doc.get("taxonomy", []) if str(c).strip()],
        "merge_map": {str(k): str(v) for k, v in (taxonomy_doc.get("merge_map") or {}).items()},
        "doc": taxonomy_doc,
    }

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

    logger.info(
        "Stage 4 [%s]: %d ok record(s), %d already done, %d to process, "
        "%d taxonomy concept(s)",
        domain, len(ok_records), len(existing_ids), len(pending), len(state["taxonomy"]),
    )

    lock = asyncio.Lock()
    tax_lock = asyncio.Lock()
    counters = {"success": 0, "failed": 0}
    pbar = tqdm(total=len(pending), desc=f"Stage 4 [{domain}] failure modes (2-pass)", unit="prob")
    pbar.set_postfix(skip=len(existing_ids))

    async def vocabulary_string() -> str:
        async with tax_lock:
            if not state["taxonomy"]:
                return "(empty taxonomy)"
            return "\n".join(f"- {c}" for c in state["taxonomy"])

    async def register_new_concepts(fms: list[dict]) -> None:
        """Add is_new_concept=true concepts to the taxonomy file (Stage 5 needs them)."""
        new_names = [
            fm["concept_involved"]
            for fm in fms
            if fm.get("is_new_concept") and fm["concept_involved"]
        ]
        if not new_names:
            return
        added = 0
        async with tax_lock:
            for name in new_names:
                if name not in state["merge_map"]:
                    state["merge_map"][name] = name
                    state["taxonomy"].append(name)
                    added += 1
            if added:
                state["doc"]["taxonomy"] = state["taxonomy"]
                state["doc"]["merge_map"] = state["merge_map"]
                save_json(taxonomy_path, state["doc"])
        if added:
            logger.info(
                "Stage 4 [%s]: registered %d new concept(s) in taxonomy", domain, added
            )

    async def worker(record: dict) -> None:
        pid = str(record.get("problem_id"))
        vocabulary = await vocabulary_string()
        prompt_a = PROMPT_PASS_A.safe_substitute(
            question=str(record.get("question", "")),
            reasoning=str(record.get("reasoning", "")),
            answer=str(record.get("answer", "")),
            normalized_taxonomy=vocabulary,
        )
        prompt_b = PROMPT_PASS_B.safe_substitute(
            domain=domain,
            question=str(record.get("question", "")),
            answer=str(record.get("answer", "")),
            normalized_taxonomy=vocabulary,
        )
        try:
            raw_a, raw_b = await asyncio.gather(
                call_llm(prompt_a, expect_json=True),
                call_llm(prompt_b, expect_json=True),
            )
        except LLMError as exc:
            logger.error(
                "Stage 4 [%s] DISCARD problem %s: pass LLM call failed: %s",
                domain, pid, exc,
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return

        pass_a = _sanitize(raw_a, "reasoning_anchored")
        pass_b = _sanitize(raw_b, "anticipatory")
        merged = _merge(pass_a, pass_b)
        if not merged:
            logger.error(
                "Stage 4 [%s] DISCARD problem %s: both passes produced no usable "
                "failure modes (pass A: %d, pass B: %d entries)",
                domain, pid, len(pass_a), len(pass_b),
            )
            counters["failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
            return

        await register_new_concepts(merged)
        if not 4 <= len(merged) <= 8:
            logger.warning(
                "Stage 4 [%s] problem %s: %d failure mode(s) after merge (target 4-8)",
                domain, pid, len(merged),
            )

        out = {
            "benchmark": record.get("benchmark"),
            "sub_benchmark": record.get("sub_benchmark"),
            "problem_id": record.get("problem_id"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "reasoning": record.get("reasoning"),
            "reasoning_status": record.get("reasoning_status"),
            "failure_modes": merged,
        }
        async with lock:
            by_id[pid] = out
            save_json(output_path, list(by_id.values()))
            counters["success"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=len(existing_ids), **counters)
        logger.debug(
            "Stage 4 [%s] problem %s: %d merged failure mode(s)", domain, pid, len(merged)
        )

    if pending:
        await asyncio.gather(*(worker(r) for r in pending))
    pbar.close()
    if not output_path.exists():
        save_json(output_path, [])  # keep downstream stages runnable on empty input
    logger.info(
        "Stage 4 [%s] complete: %d record(s) with failure modes in output",
        domain, len(by_id),
    )
